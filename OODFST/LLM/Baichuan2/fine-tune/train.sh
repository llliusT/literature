hostfile=""
deepspeed --include="localhost:2" --master_port=25641 fine-tune.py  \
    --use_lora True \
    --report_to none \
    --data_path /home/gch/project/yiming/Out-of-Domain/railway_data/baichuan_data/train_top_8.json \
    --model_name_or_path /home/gch/project/plm_cache/baichuan2_chat/baichuan2_chat \
    --output_dir /home/gch/project/biye_final/OODFST/LLM/Baichuan2/baichuan_model \
    --model_max_length 2048 \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True
    
