import datasets
from datasets import load_dataset
import random
import numpy as np
import csv
import sys
import os
import json

datasets.logging.set_verbosity(datasets.logging.ERROR)

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    '20ng': ("text", None),
    'trec': ("text", None),
    'imdb': ("text", None),
    'wmt16': ("en", None),
    'multi30k': ("text", None),
    'clinc150': ("text", None),
    'clinc150_ood': ("text", None),
    'clinc150_full': ("text", None),
    'bank': ("text", None),
    'bank_ood': ("text", None),
    'rostd': ("text", None)
}

task_to_label_dict = {
    "mnli": {0:'Entailment', 1:'Neutral', 2:'Contradiction', -1: ''},
    "rte": {0:'Entailment', 1:'Not Entailment', -1: ''},
    "sst2": {0: 'negative', 1:'positive'},
    'trec': {0: 'Abbreviation', 1:'Entity', 2:'Description and abstract concept', 3: 'Human being', 4: 'Location', 5: 'Numeric value'},
    'imdb': {0: 'Negative', 1:'Positive'},
    'wmt16': {},
    'multi30k': {},
    'clinc150': {},
    'clinc150_ood': {},
    'clinc150_full': {},
    'bank': {},
    'bank_ood': {},
    # '20ng': {
    #     0:'alt.atheism', 1:'comp.graphics', 2:'comp.os.ms-windows.misc', 3:'comp.sys.ibm.pc.hardware',
    #     4:'comp.sys.mac.hardware', 5:'comp.windows.x', 6:'misc.forsale', 7:'rec.autos',
    #     8:'rec.motorcycles', 9:'rec.sport.baseball', 10:'rec.sport.hockey', 11:'sci.crypt',
    #     12:'sci.electronics', 13:'sci.med', 14:'sci.space', 15:'soc.religion.christian',
    #     16:'talk.politics.guns', 17:'talk.politics.mideast', 18:'talk.politics.misc',
    #     19:'talk.religion.misc'}
    '20ng': {
        0:'atheism', 1:'computer graphics', 2:'ms windows', 3:'ibm pc hardware',
        4:'mac hardware', 5:'windows x', 6:'forsale', 7:'autos',
        8:'motorcycles', 9:'baseball sport', 10:'hockey sport', 11:'cryptography',
        12:'electronics', 13:'medicine', 14:'space science', 15:'christian',
        16:'guns politics talk', 17:'mideast politics talk', 18:'miscellaneous political talk',
        19:'diverse religious talk'}
}


# clinc150_bank_para = {'bill_due': 'due_bill', 'transfer':'money_transfer'}

templates = json.load(open("./template.json",'r'))
task_template = templates['20ng']

def load(args, task_name, tokenizer, shot=1000000000, max_seq_length=512, is_id=False, input_format = 'instruct', generative=True):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    global task_template
    task_template = templates[args.task_name]
    
    print("Loading {}".format(task_name))
    if task_name in ('mnli', 'rte'):
        datasets = load_glue(task_name, input_format = input_format, generative = generative)
    elif task_name == 'sst2':
        datasets = load_sst2(args, shot, is_id, input_format = input_format, generative = generative)
    elif task_name == '20ng':
        datasets = load_20ng(args, shot, is_id, input_format = input_format,  generative = generative)
    elif task_name == 'trec':
        datasets = load_trec(shot, is_id, input_format = input_format, generative = generative)
    elif task_name == 'imdb':
        datasets = load_imdb(shot, is_id, input_format = input_format, generative = generative)
    elif task_name == 'wmt16':
        datasets = load_wmt16(generative = generative, input_format = input_format)
    elif task_name == 'multi30k':
        datasets = load_multi30k(generative = generative, input_format = input_format)
    elif task_name == 'clinc150':
        datasets = load_clinc(args, is_id=True, shot=shot, input_format = input_format, generative = generative)
    elif task_name == 'clinc150_full':
        datasets = load_clinc_full(args, is_id=True, shot=shot, input_format = input_format, generative = False)
    elif task_name == 'clinc150_ood':
        datasets = load_clinc(args, is_id=False, shot='full', input_format = input_format, generative = generative)
    elif task_name == 'bank':
        datasets = load_uood(is_id=True, seed=args.seed, known_cls_ratio = args.ratio, shot=shot, input_format = input_format)
    elif task_name == 'bank_ood':
        datasets = load_uood(is_id=False, seed=args.seed, known_cls_ratio = args.ratio, shot=shot, input_format = input_format)
        
        
    def tokenize(prmopt, add_eos_token=True):
        
        length = max_seq_length
        if task_name == 'imdb' or task_name == '20ng':
            length = 600
        result = tokenizer(
            prmopt,
            truncation=True,
            max_length=length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result  
    
    
    def preprocess_function_dis(examples, add_eos_token=True):
        # inputs = examples[sentence1_key] if sentence2_key is None else examples[sentence1_key] + " " + examples[sentence2_key]
        
        # # sentence = examples['sentence']
        # # inputs = sentence
        # result = tokenize(inputs)  
        # result["labels"] = examples["label"] if 'label' in examples else 0
        
        # return result
        sentence = examples['sentence']
        inputs = sentence
        result = tokenize(inputs)  
        result['labels'] = examples['label'] 
        return result
        

    def preprocess_function_gen_train(examples, predicts_label_only=True):
        
        # inputs = (
        #     (examples[sentence1_key],) if sentence2_key is None else (
        #         examples[sentence1_key] + " " + examples[sentence2_key],)
        # )
        sentence = examples['sentence']
        label = examples['label'] 
        inputs = f'{sentence}{label}'
        result = tokenize(inputs)
        
        if predicts_label_only:
            tokenized_input_prompt = tokenize(sentence, add_eos_token=False)      
            input_prompt_len = len(tokenized_input_prompt["input_ids"])
            result["labels"] = [
                    -100
                ] * input_prompt_len + result["labels"][
                    input_prompt_len:
                ]  # could be sped up, probably    
        return result
    
    def preprocess_function_gen_val_test(examples, predicts_label_only=True):
        
        sentence = examples['sentence']
        inputs = sentence
        result = tokenize(inputs, add_eos_token=False)  
        result['labels'] = examples['label'] 
        return result

    if generative:
        train_dataset = list(map(preprocess_function_gen_train, datasets['train'])) if 'train' in datasets and is_id else None
        dev_dataset = list(map(preprocess_function_gen_val_test, datasets['validation'])) if 'validation' in datasets and is_id else None
        test_dataset = list(map(preprocess_function_gen_val_test, datasets['test'])) if 'test' in datasets else None
    else:
        train_dataset = list(map(preprocess_function_dis, datasets['train'])) if 'train' in datasets and is_id else None
        dev_dataset = list(map(preprocess_function_dis, datasets['validation'])) if 'validation' in datasets and is_id else None
        test_dataset = list(map(preprocess_function_dis, datasets['test'])) if 'test' in datasets else None
        
    return train_dataset, dev_dataset, test_dataset


def load_glue(task, input_format='instruct',  generative=True,):
    pre_datasets = load_dataset("glue", task)
    ### MNIL
    if task == 'mnli':
        label2name = task_to_label_dict['mnli']
        train_flag = True
        # task_template = templates['mnli']
        print(task_template[input_format])
        def template_mnli(item):
            label = label2name[item['label']] if train_flag and generative else item['label']
            premise = item['premise']
            hypothesis = item['hypothesis']
            sentence = premise + " " + hypothesis
            sentence = sentence.strip()
            input = task_template[input_format].format(sentence = sentence)
            return dict(sentence=input, label=label)
        
        train_dataset = pre_datasets['train']
        dev_dataset = [d for d in pre_datasets['validation_matched']] + [d for d in pre_datasets['validation_mismatched']]
        test_dataset = [d for d in pre_datasets['test_matched']] + [d for d in pre_datasets['test_mismatched']]
        # datasets['test'] = test_dataset

        train_dataset = list(map(template_mnli, train_dataset)) 
        train_flag = False
        dev_dataset = list(map(template_mnli, dev_dataset)) 
        test_dataset = list(map(template_mnli, test_dataset)) 
        datasets = {'train':train_dataset, 'validation':dev_dataset, 'test': test_dataset}
    
    ### RTE
    if task == 'rte':
        label2name = task_to_label_dict['rte']
        train_flag = True
        # task_template = templates['rte']
        print(task_template[input_format])
        def template_rte(item):
            label = label2name[item['label']] if train_flag and generative else item['label']
            premise = item['sentence1']
            hypothesis = item['sentence2']
            sentence = premise + " " + hypothesis
            sentence = sentence.strip()
            input = task_template[input_format].format(sentence = sentence)
            return dict(sentence=input, label=label)

        train_dataset = pre_datasets['train']
        dev_dataset = pre_datasets['validation']
        test_dataset = pre_datasets['test']
        
        train_dataset = list(map(template_rte, train_dataset)) 
        train_flag = False
        dev_dataset = list(map(template_rte, dev_dataset)) 
        test_dataset = list(map(template_rte, test_dataset)) 
        
        datasets = {'train':train_dataset, 'validation':dev_dataset, 'test': test_dataset}

    return datasets


def load_clinc_full(args, is_id, shot=100, known_cls_ratio = 0.5, input_format = 'normal', generative = False, data_dir="./data/clinc_full"):
    # domain 
    # domain = args.domain
    # domain_map = json.load(open(os.path.join(data_dir, 'domains.json'),'r'))
    # # label_list = list(map(lambda x: clinc150_bank_para.get(x,x),domain_map[domain])) 
    # label_list = domain_map[domain]
    if args.domain is not None:
        domain_map = json.load(open(os.path.join(data_dir, 'domains.json'),'r'))
    # label_list = list(map(lambda x: clinc150_bank_para.get(x,x),domain_map[domain])) 
        all_label_list_pos = domain_map[args.domain]
    else:  
        all_label_list_pos = get_labels(data_dir)
        
    print(len(all_label_list_pos))
    label_map = {}
    inverse_label_map = {}
    for i, label in enumerate(all_label_list_pos):
        label_map[label] = i
        inverse_label_map[i] = label
        
        
    task_to_label_dict['clinc150'] = inverse_label_map

    train_flag = True
    print(task_template[input_format])
    
    
    def template(item):
        label = inverse_label_map[item['label']] if train_flag and generative else item['label']
        intent = item['text'].strip()
        input = task_template[input_format].format(sentence=intent)
        return dict(sentence=input, label=label)

    if is_id:
        train_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "train.tsv")), label_map, all_label_list_pos)
        dev_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "valid.tsv")), label_map, all_label_list_pos)
        test_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "test.tsv")), label_map, all_label_list_pos)
        
        if shot != 'full':
            train_dataset = select_few_shot(shot, train_dataset, "clinc150", args.seed)
            dev_dataset = select_few_shot(shot, dev_dataset, "clinc150", args.seed)
            print(f'few-shot setting : train {shot}, val {shot}')
        
        # template
        train_dataset = list(map(template, train_dataset)) 
        train_flag = False
        dev_dataset = list(map(template, dev_dataset)) 
        test_dataset = list(map(template, test_dataset)) 
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    else:
        
        train_flag = False
        test_dataset = [template(i) for i in _get_ood(
            _read_tsv(os.path.join(data_dir, "test_para.tsv")), all_label_list_pos)]
        datasets = {'test': test_dataset}
        
    return datasets

def load_clinc(args, is_id, shot=100, known_cls_ratio = 0.5, input_format = 'normal', generative = True, data_dir="./data/clinc_full"):
    # domain 
    domain = args.domain
    domain_map = json.load(open(os.path.join(data_dir, 'domains_para.json'),'r'))
    # label_list = list(map(lambda x: clinc150_bank_para.get(x,x),domain_map[domain])) 
    label_list = domain_map[domain]
    
    n_known_cls = round(len(label_list) * known_cls_ratio)
    np.random.seed(args.seed)
    known_label_list = list(
        np.random.choice(np.array(label_list), n_known_cls, replace=False))
    ood_labels = list(set(label_list) - set(known_label_list))
    print(f'ID Classes: {known_label_list}')
    print(f'OOD Classes: {ood_labels}')

    label_map = {}
    inverse_label_map = {}
    for i, label in enumerate(known_label_list):
        label_map[label] = i
        inverse_label_map[i] = label
        
        
    task_to_label_dict['clinc150'] = inverse_label_map
    task_to_label_dict['clinc150_ood'] = ood_labels
    
    train_flag = True
    print(task_template[input_format])
    
    
    def template(item):
        label = inverse_label_map[item['label']] if train_flag and generative else item['label']
        intent = item['text'].strip()
        input = task_template[input_format].format(sentence=intent)
        return dict(sentence=input, label=label)

    if is_id:
        train_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "train_para.tsv")), label_map, known_label_list)
        dev_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "valid_para.tsv")), label_map, known_label_list)
        test_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "test_para.tsv")), label_map, known_label_list)
        
        if shot != 'full':
            train_dataset = select_few_shot(shot, train_dataset, "clinc150", args.seed)
            dev_dataset = select_few_shot(shot, dev_dataset, "clinc150", args.seed)
            print(f'few-shot setting : train {shot}, val {shot}')
        
        # template
        train_dataset = list(map(template, train_dataset)) 
        train_flag = False
        dev_dataset = list(map(template, dev_dataset)) 
        test_dataset = list(map(template, test_dataset)) 
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    else:
        
        train_flag = False
        test_dataset = [template(i) for i in _get_ood(
            _read_tsv(os.path.join(data_dir, "test_para.tsv")), ood_labels)]
        datasets = {'test': test_dataset}
        
    return datasets


def load_uood(is_id, shot=100000000, data_dir="./data/banking", seed=42, known_cls_ratio=0.50, dataname='bank', input_format = 'instruct'):
    all_label_list_pos = get_labels(data_dir)
    n_known_cls = round(len(all_label_list_pos) * known_cls_ratio)
    np.random.seed(seed)
    known_label_list = list(
        np.random.choice(np.array(all_label_list_pos), n_known_cls, replace=False))
    ood_labels = list(set(all_label_list_pos) - set(known_label_list))
    label_map = {}
    inverse_label_map = {}
    for i, label in enumerate(known_label_list):
        label_map[label] = i
        inverse_label_map[i] = label

    
    task_to_label_dict['bank'] = inverse_label_map
    task_to_label_dict['bank_ood'] = ood_labels
    train_flag = True
    task_template = templates['bank']

    def template(item):
        label = inverse_label_map[item['label']] if train_flag else item['label']
        utterance = item['text']
        input = task_template[input_format].format(utterance=utterance)
        return dict(sentence=input, label=label)
    
    if is_id:
        
        train_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "train.tsv")), label_map, known_label_list)
        dev_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "dev.tsv")), label_map, known_label_list)
        test_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "test.tsv")), label_map, known_label_list)
        
        if shot < 1:
            train_dataset = select_few_shot(shot, train_dataset, "bank", seed)
            dev_dataset = select_few_shot(shot, dev_dataset, "bank", seed)
            
            
        train_flag = True
        train_dataset = [template(i) for i in train_dataset]
        
        train_flag = False
        dev_dataset = [template(i) for i in dev_dataset]
        test_dataset = [template(i) for i in test_dataset]
        # train_dataset = select_few_shot(shot, train_dataset, dataname)
        # dev_dataset = select_few_shot(shot, dev_dataset, dataname)
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
        
        
    else:
        train_flag = False
        test_dataset =[template(i) for i in _get_ood(
            _read_tsv(os.path.join(data_dir, "test.tsv")), ood_labels) ]
        datasets = {'test': test_dataset}
    return datasets



def load_20ng(args, shot, is_id, generative=True, input_format='instruct'):
    all_subsets = (
        '18828_alt.atheism', '18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.ibm.pc.hardware',
        '18828_comp.sys.mac.hardware', '18828_comp.windows.x', '18828_misc.forsale', '18828_rec.autos',
        '18828_rec.motorcycles', '18828_rec.sport.baseball', '18828_rec.sport.hockey', '18828_sci.crypt',
        '18828_sci.electronics', '18828_sci.med', '18828_sci.space', '18828_soc.religion.christian',
        '18828_talk.politics.guns', '18828_talk.politics.mideast', '18828_talk.politics.misc',
        '18828_talk.religion.misc')
    label2name = task_to_label_dict['20ng']
    train_flag = True
    print(task_template[input_format])
    def template(item):
        label = label2name[item['label']] if train_flag and generative else item['label']
        sentence = item['text'].strip()
        input = task_template[input_format].format(sentence=sentence)
        return dict(sentence=input, label=label)
    
    if 'opt' in args.model_name_or_path:
        print('opt dataset...')
        datasets = json.load(open('./data/20ng/opt_dataset.json','r'))
        
    elif 'llama' in args.model_name_or_path:
        print('llama dataset...')
        datasets = json.load(open('./data/20ng/dataset.json', 'r'))
        
    
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for i, subset in enumerate(all_subsets):
        # dataset = load_dataset('newsgroup', subset)['train']
        # dataset = dataset.shuffle()
        dataset = datasets[subset]
        examples = [{'text': d['text'], 'label': i} for d in dataset]
        num_train = int(0.8 * len(dataset))
        num_dev = int(0.1 * len(dataset))
        train_dataset += examples[:num_train]
        dev_dataset += examples[num_train: num_train + num_dev]
        test_dataset += examples[num_train + num_dev:]
        
    if shot != 'full' and is_id:
        train_dataset = select_few_shot(shot, train_dataset, "20ng", args.seed)
        dev_dataset = select_few_shot(shot, dev_dataset, "20ng", args.seed)
        print(f'few-shot setting : train {shot}, val {shot}')

    train_dataset = list(map(template, train_dataset)) 
    train_flag = False
    dev_dataset = list(map(template, dev_dataset)) 
    test_dataset = list(map(template, test_dataset)) 
    
    
    # train_dataset += [template(j['text'], j['label']) for j in train_dataset]
    # dev_dataset += [template(j['text'], j['label']) for j in dev_dataset]
    # test_dataset += [template(j['text'], j['label']) for j in test_dataset]
    

    # if is_id:
    #     train_dataset = select_few_shot(shot, train_dataset, "20ng")
    #     dev_dataset = select_few_shot(shot, dev_dataset, "20ng")
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_trec(shot, is_id, generative=True, input_format = "instruct"):
    datasets = load_dataset('trec')
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    
    label2name = task_to_label_dict['trec']
    train_flag = True
    # task_template = templates['trec']
    print(task_template[input_format])
    def template(item):
        question = item['text'].strip()
        category = label2name[item['coarse_label']] if train_flag and generative else item['coarse_label']
        input = task_template[input_format].format(sentence=question)
        return dict(sentence=input, label=category)
    
    idxs = list(range(len(train_dataset)))
    random.seed(42)
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    

    train_dataset_new = [template(train_dataset[i]) for i in
                    idxs[:-num_reserve]]
    
    train_flag = False
    
    dev_dataset_new = [template(train_dataset[i]) for i in
                idxs[-num_reserve:]]

    test_dataset = [template(d) for d in test_dataset]
    # else:
    #     dev_dataset_new = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['coarse_label']} for i in
    #                 idxs[-num_reserve:]]
    #     train_dataset_new = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['coarse_label']} for i in
    #                     idxs[:-num_reserve]]
    #     test_dataset = [{'text': d['text'], 'label': d['coarse_label']} for d in test_dataset]
    # if is_id:
    #     train_dataset = select_few_shot(shot, train_dataset, "trec")
    #     dev_dataset = select_few_shot(shot, dev_dataset, "trec")
    datasets = {'train': train_dataset_new, 'validation': dev_dataset_new, 'test': test_dataset}
    return datasets


def load_imdb(shot, is_id, generative=True, input_format='instruct'):
    # datasets = load_dataset('imdb')
    # train_dataset_all = datasets['train']
    
    train_dataset_all = json.load(open('./data/imdb/train.json','r'))
    test_dataset_all = json.load(open('./data/imdb/test.json','r'))
    
    label2name = task_to_label_dict['imdb']
    train_flag = True
    # task_template = templates['imdb']
    print(task_template[input_format])
    
    def template(item):
        review = item['text'].strip()
        label = label2name[item['label']] if train_flag and generative else item['label']
        
        # pre = tokenizer(review, truncation=True,
        #     max_length=512,
        #     padding=False,
        #     return_tensors=None)['input_ids']
        # after = tokenizer.decode(pre, skip_special_tokens=True)
        input = task_template[input_format].format(sentence=review)
        return dict(sentence=input, label=label)
    
    idxs = list(range(len(train_dataset_all)))
    random.seed(42)
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset_all) * 0.1)


    train_dataset = [template(train_dataset_all[i]) for i in
                    idxs[:-num_reserve]]
    
    train_flag = False
    dev_dataset = [template(train_dataset_all[i]) for i in idxs[-num_reserve:]]
    test_dataset =  [template(d) for d in test_dataset_all]
    # else:
    #     train_dataset = [train_dataset_all[i] for i in idxs[:-num_reserve]]
    #     dev_dataset = [train_dataset_all[i] for i in idxs[-num_reserve:]]
    #     test_dataset =  test_dataset_all
    # if is_id:
    #     train_dataset = select_few_shot(shot, train_dataset, "imdb")
    #     dev_dataset = select_few_shot(shot, dev_dataset, "imdb")
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_wmt16(generative=True,input_format='instruct'):
    datasets = load_dataset('wmt16', 'de-en')
    # task_template = templates['wmt16']
    print(task_template[input_format])
    def template(item):
        english = item['en'].strip()
        label = item['de']
        input = task_template[input_format].format(sentence=english)
        return dict(sentence=input, label=0)
    
    test_dataset = [template(d['translation']) for d in datasets['test']]

        
    datasets = {'test': test_dataset}
    return datasets


def load_multi30k(generative=True, input_format='instruct'):
    test_dataset = []
    # task_template = templates['multi30k']
    print(task_template[input_format])
    def template(line):
        input = task_template[input_format].format(sentence=line)
        return dict(sentence=input, label=0)
    
    for file_name in ('./data/multi30k/test_2016_flickr.en', './data/multi30k/test_2017_mscoco.en',
                      './data/multi30k/test_2018_flickr.en'):
        with open(file_name, 'r') as fh:
            for line in fh:
                line = line.strip()
                if len(line) > 0:
                    example = template(line)
                    test_dataset.append(example)
    datasets = {'test': test_dataset}
    return datasets


def load_sst2(args, shot, is_id, generative=True, input_format = "instruct"):
    label2name = task_to_label_dict['sst2']
    train_flag = True
    # task_template = templates['sst2']
    print(task_template[input_format])
    def process(file_name):
        examples = []
        with open(file_name, 'r') as fh:
            for line in fh:
                splits = line.split()
                label = splits[0]
                text = " ".join(splits[1:])
                examples.append(
                    {'sentence': text, 'label': int(label)}
                )
        return examples

    def template(item):
        label = label2name[item['label']] if train_flag and generative else item['label']
        sentence = item['sentence'].strip()
        input = task_template[input_format].format(sentence=sentence)
        return dict(sentence=input, label=label)
    
    datasets = load_dataset('glue', 'sst2')
    train_dataset = datasets['train']
    dev_dataset = datasets['validation']
    test_dataset = process('./data/sst2/test.data')
    
    if shot != 'full' and is_id:
        train_dataset = select_few_shot(shot, train_dataset, "sst2", args.seed)
        dev_dataset = select_few_shot(shot, dev_dataset, "sst2", args.seed)
        print(f'few-shot setting : train {shot}, val {shot}')
    

    train_dataset = list(map(template, train_dataset)) 
    train_flag = False
    dev_dataset = list(map(template, dev_dataset)) 
    test_dataset = list(map(template, test_dataset)) 
    # if is_id:
    #     train_dataset = select_few_shot(shot, train_dataset, "sst2")
    #     dev_dataset = select_few_shot(shot, dev_dataset, "sst2")
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


# train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} for i in
#                      idxs[:-num_reserve]]
def select_few_shot(shot, trainset, task_name, seed = 42):
    # examples = []
    shot = float(shot)
    few_examples = []
    sentence1_key, sentence2_key = task_to_keys[task_name]
    from collections import defaultdict
    sorted_examples = defaultdict(list)

    for example in trainset:
        # if example.label in self.known_label_list and np.random.uniform(0, 1) <= args.labeled_ratio:
        #     examples.append(example)
        sorted_examples[example["label"]] = sorted_examples[example["label"]] + [example[sentence1_key]]
        
    for k, v in sorted_examples.items():
        arr = np.array(v)
        if shot < 1:
            len_ = int(len(arr)*shot)
        else:
            len_ = int(shot)
        np.random.seed(seed)
        np.random.shuffle(arr)
        for elems in arr[:len_]:
            few_examples.append({sentence1_key: elems, 'label': k})

    return few_examples


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def _create_examples(lines, label_map, know_labels):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if len(line) != 2:
            continue
        # guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if label in know_labels:
            examples.append(
                {'text': text_a, 'label': label_map[label]})
    return examples


def _get_ood(lines, ood_labels):
    out_examples = []

    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if len(line) != 2:
            continue
        # guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if label in ood_labels:
            out_examples.append(
                {'text': text_a, 'label': 0})

    return out_examples


def get_labels(data_dir):
    """See base class."""
    import pandas as pd
    test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
    labels = np.unique(np.array(test['label']))

    return labels
    