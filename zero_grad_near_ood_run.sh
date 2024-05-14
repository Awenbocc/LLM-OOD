for SEED in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python run_gen.py  \
        --model_name_or_path /home/bossjobai/LLM_Projects/weights/llama-7b \
        --task_name 'clinc150'  \
        --domain 'banking' \
        --shot 'full' \
        --seed ${SEED} \
        --learning_rate 5e-5 \
        --batch_size 16 \
        --accumulation_step 1 \
        --num_train_epochs 50 \
        --tunable_strategy 'zero' \
        --input_format 'instruct_banking' \
        --sentence_emb 'last' \
        --save_results_path './results/zero/llama-7b'\

done