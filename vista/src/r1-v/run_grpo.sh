\export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"



torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ./results/output_table_Qwen2-VL-7B_chartqa \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name HuggingFaceM4/ChartQA \
    --deepspeed ./local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CHARTQA \
    --save_steps 30 \
    --save_only_model true \
    --num_generations 4