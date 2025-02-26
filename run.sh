export HF_ENDPOINT=https://hf-mirror.com

DATA_PATH=/ceph2/user-data/linzhentao/code/ASR_training/wenet/examples/seewo/v3/data/edu_datas/zh_0403_in_wer/wer_0_5
LLM_MODEL=Qwen/Qwen2-1.5B-Instruct
# LLM_MODEL=Qwen/Qwen2.5-0.5B
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
    --llm_model_name_or_path ${LLM_MODEL} \
    --whisper_model_name_or_path tiny \
    --data_path ${DATA_PATH} \
    --bf16 True \
    --output_dir ${LLM_MODEL}-whisper-tiny \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 512 \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 10 \
    --deepspeed ds_config_zero3.json