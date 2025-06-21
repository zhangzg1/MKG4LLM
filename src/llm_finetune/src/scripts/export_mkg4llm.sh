CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
    --model_name_or_path ../../model/RoG/ \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --export_dir ../merge_mgk4llm_model/ \
    --adapter_name_or_path ../saves/mgk4llm-model/checkpoint-660/
