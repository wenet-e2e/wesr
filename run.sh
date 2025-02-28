export HF_ENDPOINT=https://hf-mirror.com

DATA_PATH=/ceph2/user-data/linzhentao/code/ASR_training/wenet/examples/seewo/v3/data/edu_datas/zh_0403_in_wer/wer_0_5
DATA_PATH=/ceph2/user-data/chenzhongliang/west/aishell/train.jsonl
LLM_MODEL=Qwen/Qwen2-1.5B-Instruct
# LLM_MODEL=Qwen/Qwen2.5-0.5B
PER_DEVICE_TRAIN_BATCH_SIZE=48
stage=$1
stop_stage=$1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
        --projector_model_path Qwen-1.5B-Instruct-whisper-tiny/checkpoint-1170/model.safetensors \
        --llm_model_name_or_path ${LLM_MODEL} \
        --whisper_model_name_or_path tiny \
        --data_path ${DATA_PATH} \
        --bf16 True \
        --output_dir ${LLM_MODEL}-whisper-tiny \
        --num_train_epochs 5 \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
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
        --dataloader_num_workers 16 \
        --dataloader_prefetch_factor 64 \
        --deepspeed ds_config_zero3.json

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
        --grpo \
        --projector_model_path Qwen-1.5B-Instruct-whisper-tiny/checkpoint-1170/model.safetensors \
        --llm_model_name_or_path ${LLM_MODEL} \
        --whisper_model_name_or_path tiny \
        --data_path ${DATA_PATH} \
        --bf16 True \
        --output_dir ${LLM_MODEL}-whisper-tiny \
        --temperature 0.5 \
        --num_train_epochs 5 \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1 \
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
        --dataloader_num_workers 16 \
        --dataloader_prefetch_factor 64 \
        --deepspeed ds_config_zero3.json


fi

TEST_MODEL=Qwen-1.5B-Instruct-whisper-tiny/checkpoint-1170/model.safetensors
TEST_MODEL=Qwen/Qwen2-1.5B-Instruct-whisper-tiny/checkpoint-3/model.safetensors
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python recognize.py \
        --llm_model_name_or_path  ${LLM_MODEL} \
        --whisper_model_name_or_path tiny \
        --projector_model_path ${TEST_MODEL} \
        --data_path test.jsonl \
        --result_path result.txt
fi