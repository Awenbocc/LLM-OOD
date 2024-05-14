import os
# os.environ['HF_DATASETS_OFFLINE'] = '1'
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import argparse
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaTokenizer, GenerationConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn, AverageMeter, accuracy, DataCollatorForSeq2Seq, DataCollatorForSeq2Cls
from datasets import load_metric
from llama_modeling import LlamaForCausalLM
from evaluation import evaluate_ood
import warnings
from data import load, templates, task_to_label_dict
import json
import pandas as pd
import mlflow.pytorch
import umap
import umap.plot
import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")

task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20ng': 20,
    'trec': 6,
    'clinc150': 150,
    "bank": round(77 * 0.5),
    'rostd': 3
}

task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20ng': 'mnli',
    'trec': 'mnli',
    'clinc150': 'mnli',
    'bank': 'mnli',
    'rostd': 'mnli',
}


 


def train(args, model, tokenizer, train_dataset, dev_dataset, test_dataset, benchmarks):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer), shuffle=True,
                                  drop_last=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=DataCollatorForSeq2Cls(tokenizer=tokenizer))
    gradient_accumulation_steps = args.accumulation_step
    total_steps = int(len(train_dataloader) // gradient_accumulation_steps * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def detect_ood():
        model.prepare_ood(dev_dataloader)
        res = {}
        for tag, ood_features in benchmarks:
            print(f"**************star to evluate OOD dataset {tag}****************")
            results = evaluate_ood(args, model, tokenizer, test_dataset, ood_features, tag=tag)
            res = dict(res, **results)
            print(f"**************finishing evaluting OOD dataset {tag}*************")
        return res
    

    # Zero OODx
    if args.tunable_strategy == 'zero':
        epoch_dir = os.path.join(args.sub_exp_dir, f'epoch_zero')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        args.epoch_dir = epoch_dir
        plot(args, model, tokenizer, test_dataset, benchmarks, epoch_dir)
        ood_res = detect_ood()
        final_res = dict({'mode_name': args.model_name ,'sentence_emb': args.sentence_emb,'input_format': args.input_format, "tunable_strategy": args.tunable_strategy}, **ood_res)
        save_results(args, final_res)    
        return "Over for Zero-OOD Detection"
    
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, eps=1e-8, weight_decay= args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    


    with mlflow.start_run():
        mlflow.log_param("model_type", args.model_name_or_path.split('/')[-1])
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.num_train_epochs)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("warmup_ratio", args.warmup_ratio)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("tunable_strategy", args.tunable_strategy)
        
        
        best_eval = -float('inf')
        eval_fre = 5
        # epochs = 0
        patient = 0
        loss_avg = AverageMeter()
        acc_avg = AverageMeter()
        final_res = {}
        best_test = -float('inf')
        # eval_loss_avg.reset()
        result_pic = []
    
        # Zero test
        # change #1
        epoch_dir = os.path.join(args.sub_exp_dir, f'epoch_zero')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        args.epoch_dir = epoch_dir
        plot(args, model, tokenizer, test_dataset, benchmarks, epoch_dir)
        
        # change 2#
        tes_res = test(args, model, tokenizer, test_dataset, tag="test")
        print(f'zero test acc is {tes_res}')
        mlflow.log_metric("test_accuracy", tes_res, step=0)
        
        for epoch in range(int(args.num_train_epochs)):
            print("-"*20)
            if args.tunable_strategy =='lora':
                assert model.peft_config['default'].inference_mode == False
            model.zero_grad()
            model.train()
            loss_avg.reset()
            acc_avg.reset()
            epoch += 1
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {key: value.to("cuda") for key, value in batch.items()}
                outputs = model(**batch)
                loss, logits = outputs[0], outputs[1]
                loss = loss / gradient_accumulation_steps
                loss.backward()
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                # loss.backward()
                loss_avg.update(loss.item(), len(batch['labels']))
            
            
            print("train loss of epoch:", epoch, loss_avg.avg) 
            
            
            dev_acc = test(args, model, tokenizer, dev_dataset, tag="dev")
            print("dev acc:", dev_acc)
            
            mlflow.log_metric("train_loss", loss_avg.avg, step=epoch)
            mlflow.log_metric("dev_acc", dev_acc, step=epoch)
            
            tes_res = test(args, model, tokenizer, test_dataset, tag="test", print_=True if epoch==5 else False)
            print("test result:", tes_res)
            mlflow.log_metric("test_accuracy", tes_res, step=epoch)


            epoch_dir = os.path.join(args.sub_exp_dir, f'epoch_{epoch}')
            
            
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
                
            args.epoch_dir = epoch_dir
                
            # change 3#
            if len(benchmarks):
                plot(args, model, tokenizer, test_dataset, benchmarks, epoch_dir)
                ood_res = detect_ood()
                final_res = dict({'epoch': epoch, "test_acc": tes_res, 'eval_acc': dev_acc, 'sentence_emb': args.sentence_emb, 'seed': args.seed}, **ood_res)
                save_results(args, final_res)
            
                
            if tes_res > best_test:
                best_test = tes_res
                mlflow.log_metric("best_test_accuracy", best_test, step=epoch)
            
            
            if dev_acc >= best_eval:
                best_eval = dev_acc
                patient = 0

            else:
                patient +=1
            
            if patient>6 and epoch>15:
                break
    
@torch.no_grad()
def plot(args, model, tokenizer, id_dataset, ood_datasets, path, metric='cosine', min_dist=0.1, n_neighbors=10, densmap=True):
    
    print('start to draw...')
    # id test vs. ood test
    
    if args.task_name == 'sst2':
        id_name = 'ID: SST-2'
    elif args.task_name == '20ng':
        id_name = 'ID: 20 NewsGroups'
    elif args.task_name == 'bank':
        id_name = 'ID: Bank'
    else:
        id_name = 'ID: CLINC150'
    

    dataloader = DataLoader(id_dataset, batch_size=16, collate_fn=DataCollatorForSeq2Cls(tokenizer))
    in_sentences = []
    in_labels = np.array([id_name]*len(id_dataset))
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            sentence_embedding = model.get_sentence_embedding(**batch)
            in_sentences.append(sentence_embedding.to(dtype=torch.float32).cpu().numpy())
            
    in_sentences = np.concatenate(in_sentences)

    all_ood_label = []
    all_ood_sentences = []
    selected_set = ['ood_rte', 'ood_20ng', 'ood_trec', 'ood_bank_ood', 'ood_sst2','ood_clinc150_ood']
    for tag, ood_dataset in ood_datasets:
        if tag not in selected_set:
            continue
        if tag == 'ood_rte':
            tag = 'OOD: RTE'
        if tag == 'ood_20ng':
            tag = 'OOD: 20 NewsGroups'
        if tag == 'ood_trec':
            tag = 'OOD: TREC'
        if tag == 'ood_bank_ood':
            tag = 'OOD: Bank'
        if tag == 'ood_sst2':
            tag = 'OOD: SST-2'
        if tag == 'ood_clinc150_ood':
            tag = 'OOD: CLINC150'
        ood_label = np.array([tag]*len(ood_dataset))
        dataloader = DataLoader(ood_dataset, batch_size=16, collate_fn=DataCollatorForSeq2Cls(tokenizer))
        ood_sentences = []
        for batch in dataloader:
            model.eval()
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.no_grad():
                sentence_embedding = model.get_sentence_embedding(**batch)
                ood_sentences.append(sentence_embedding.to(dtype=torch.float32).cpu().numpy())

        ood_sentences = np.concatenate(ood_sentences)
        all_ood_label.append(ood_label)
        all_ood_sentences.append(ood_sentences)
        
    all_ood_label = np.concatenate(all_ood_label) 
    all_ood_sentences = np.concatenate(all_ood_sentences)
    
    y_test = np.concatenate((in_labels, all_ood_label), axis=0)
    x_test = np.concatenate((in_sentences, all_ood_sentences), axis=0)
    
    
    labels = list(set(y_test))
    # colours = ['deepskyblue', 'limegreen', 'orange', 'salmon']
    # colour_key = {labels[i]:colours[i] for i in range(len(labels))}
    colour_key = {'ID: SST-2':'deepskyblue', 'ID: 20 NewsGroups':'deepskyblue', 'OOD: TREC':'limegreen', 'OOD: RTE': 'orange', 'OOD: 20 NewsGroups':'salmon', 'OOD: SST-2':'salmon',
                  'ID: Bank': 'darkorange', 'OOD: Bank':'darkgrey',
                  'ID: CLINC150': 'darkorange', 'OOD: CLINC150':'darkgrey'}

    embedding = umap.UMAP(densmap=True, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(x_test)
    umap.plot.points(embedding, labels=y_test, color_key=colour_key)
    savename = os.path.join(path,'embedding.png')
    plt.title('ID and OOD Test Data Only')
    plt.savefig(savename, bbox_inches='tight', dpi=400)
    plt.show()
    print('finish drawing...')
    

def evaluate(args, model, tokenizer, eval_dataset, tag="train"):
    metric_name = task_to_metric[args.task_name]
    metric = load_metric("./metrics/glue", metric_name)
    loss_avg = AverageMeter()
    
    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result

    dataloader = DataLoader(eval_dataset, batch_size=args.val_batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer))

    label_list, logit_list = [], []
    model.eval()
    for step, batch in enumerate(tqdm(dataloader)):
        # labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to("cuda") for key, value in batch.items()}
        # batch["mode"] = tag
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs[0]
            loss_avg.update(loss.item(), len(batch['labels']))
    return loss_avg.avg


# Generative test
@torch.no_grad()
def test(args, model, tokenizer, eval_dataset, tag="train", print_=False):

    if 'instruct' in args.input_format:
        input_format = 'instruct'
    else:
        input_format = args.input_format
        
    task_split = templates[args.task_name][f'{input_format}_response_split']
    task_label = task_to_label_dict[args.task_name] #label dict
    dataloader = DataLoader(eval_dataset, batch_size=args.val_batch_size, collate_fn=DataCollatorForSeq2Cls(tokenizer=tokenizer))

    label_list = []
    generated_answer = []

    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        input_ids=batch['input_ids']
        # Greedy Search
        with torch.no_grad():
            generation_output = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            attention_mask = batch['attention_mask'].to('cuda'),
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=16,
            use_cache=True
        )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s,skip_special_tokens=True)
        answer = [i.split(task_split)[1].strip() for i in output]
        generated_answer += answer
        label_list += [task_label[i.item()] for i in batch['labels']]

        
    assert len(label_list) == len(generated_answer)
    acc = sum(np.array(label_list) == np.array(generated_answer)) / len(label_list)
    
    if print_:
        task_dir = args.sub_exp_dir
        with open(f'{task_dir}/predict.json','w') as file:
            results_ = {'truth':label_list, 'predict':generated_answer}
            json.dump(results_, file)
            
    return acc




def save_tunable_parameters(model, path, lora_training=True):
    if lora_training:
        peft_config = model.peft_config['default']
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True
        peft_config.save_pretrained(path)
        
        peft_config.inference_mode = inference_mode
    
    
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, os.path.join(path,"adapter_model.bin"))



def save_results(args, test_results):
    task_dir = args.sub_exp_dir

    var = [args.task_name, args.seed]
    names = ['dataset', 'seed']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = 'results.csv'
    results_path = os.path.join(task_dir, file_name)

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = pd.concat([df1, new], ignore_index=True)
        df1.to_csv(results_path, index=False)
    data_diagram = pd.read_csv(results_path)

    print('test_results')
    print(data_diagram)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="llama", type=str, help="llama_weights_path")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--task_name", default="sst2", type=str)
    parser.add_argument("--domain", default="banking", type=str)
    parser.add_argument("--input_format", default="instruct", type=str)
    parser.add_argument("--sentence_emb", default="avg", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--val_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--warmup_ratio", default=0.04, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulation_step", type=int, default=1)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--shot", default=1)
    parser.add_argument("--tunable_strategy", type=str, default='lora')
    parser.add_argument("--save_results_path", type=str, help="the path to save results")
    parser.add_argument("--pooling_way", type=str, default='final_token')

    args = parser.parse_args()
    set_seed(5)
    num_labels = task_to_labels[args.task_name]
    task_dir = os.path.join(args.save_results_path, args.task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        

    task_dir = os.path.join(task_dir, str(args.shot))
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    
    sub_task = args.input_format + '_'+ args.sentence_emb + '_seed' + str(args.seed)
    sub_exp_dir = os.path.join(task_dir, sub_task)
    if not os.path.exists(sub_exp_dir):
        os.makedirs(sub_exp_dir)
        
    args.sub_exp_dir = sub_exp_dir
    
    print(args.model_name_or_path)
    args.model_name = args.model_name_or_path.split('/')[-1]
    
    if 'llama' in args.model_name_or_path:
        config = LlamaConfig.from_pretrained(args.model_name_or_path)
        # config.gradient_checkpointing = True
        config.output_hidden_states= False
        config.use_cache = False
        config.sentence_emb = args.sentence_emb
        config.task = args.task_name
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        config.pad_token_id = 0
        tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
        )
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path, 
            config = config,
            torch_dtype = torch.float32,
            device_map = "auto"
        )
    else:
        pass
        # TODO: customized for large encoder-decoder architecture like Flan-T5
        
        
    if args.tunable_strategy =='lora':
        # Lora-tuning
        from peft import (
            LoraConfig,
            get_peft_model,
        )
        lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    elif args.tunable_strategy =='zero':
        model.requires_grad_(False)
    
    else:
        pass
        
    for name, para in model.named_parameters():
        if para.requires_grad:
            print(f'\033[0;33m {name} is trainable in this version. Please check if the setting is right! \033[0m')
    
    if args.task_name == '20ng':
        datasets = ['sst2', 'trec', 'mnli', 'rte', 'imdb', 'wmt16', 'multi30k', '20ng']
    elif args.task_name == 'sst2':
        datasets = ['sst2', 'trec', 'mnli', 'rte', 'wmt16', 'multi30k', '20ng']
    elif args.task_name == 'bank':
        datasets = ['bank', 'bank_ood']
    elif args.task_name == 'clinc150':
        datasets = ['clinc150', 'clinc150_ood']
    else:
        datasets = ['trec', 'imdb', 'wmt16', 'multi30k', 'rte', 'sst2', 'mnli', '20ng']

    benchmarks = ()
    for dataset in datasets:
        if dataset == args.task_name:
            train_dataset, dev_dataset, test_dataset = load(args, dataset, tokenizer, shot=args.shot, max_seq_length=args.max_seq_length,
                                                            is_id=True, input_format = args.input_format, generative=True )
            print(f"train size of {dataset} is {len(train_dataset)}, valid set is  {len(dev_dataset)}, test size is {len(test_dataset)}")
            
            # compute prob for each class
            if args.task_name == 'clinc150':
                label2dict = task_to_label_dict[args.task_name]
                print(label2dict)
                label_template = ['### Output:\n'+i for i in label2dict.values()]
                tokens = [tokenizer(i)['input_ids'][5] for i in label_template]
                print(tokens)
                print(tokenizer.convert_ids_to_tokens(tokens))
                assert len(tokens) == len(set(tokens)), 'wrong label name paraphrase'
                
                if args.tunable_strategy =='lora':
                    model.model.prob_loc = tokens
                else:
                    model.prob_loc = tokens

        else:
            _, _, ood_dataset = load(args, dataset, tokenizer, max_seq_length=args.max_seq_length, input_format = args.input_format, generative=True)
            benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks
            print("ood size "+dataset, len(ood_dataset))

    mlflow.set_experiment(f"Tunable: {args.tunable_strategy}, ID: {args.task_name}, Shot: {args.shot}")
    
    train(args, model, tokenizer, train_dataset, dev_dataset, test_dataset, benchmarks)


if __name__ == "__main__":
    main()
