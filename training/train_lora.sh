MODEL_NAME="./weights/Kolors"
INSTANCE_DIR='dataset/dog'
OUTPUT_DIR="./trained_models/dog_lora"
cfg_file=training/default_config.yaml

accelerate launch --config_file ${cfg_file} training/train.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="polynomial" \
    --lr_warmup_steps=500 \
    --rank=32 \
    --resolution=1024 \
    --max_train_steps=1000 \
    --checkpointing_steps=1000 \
    --center_crop \
    --mixed_precision='fp16' \
    --seed=1 \
    --img_repeat_nums=1 \
    --sample_batch_size=2 \
    --gradient_checkpointing \
    --adam_weight_decay=1e-02 \
    --checkpoints_total_limit=1 \
    --report_to="wandb" \
    --train_lora 
