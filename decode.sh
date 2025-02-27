python recognize.py \
    --llm_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --whisper_model_name_or_path tiny \
    --projector_model_path /ceph2/user-data/chenzhongliang/west/Qwen-1.5B-Instruct-whisper-tiny/checkpoint-1170/model.safetensors \
    --data_path /ceph2/user-data/chenzhongliang/west/aishell/test.jsonl \
    --result_path result.txt
`