for SEED in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python run_gen.py  \
        --model_name_or_path /home/bossjobai/LLM_Projects/weights/llama-7b \
        --task_name 'sst2'  \
        --shot 10 \
        --seed ${SEED} \
        --learning_rate 1e-4 \
        --batch_size 8 \
        --accumulation_step 1 \
        --num_train_epochs 50 \
        --tunable_strategy 'lora' \
        --input_format 'normal' \
        --sentence_emb 'last' \
        --save_results_path './results/tuned/llama-7b'\

done