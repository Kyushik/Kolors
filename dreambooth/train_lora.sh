MODEL_NAME="./weights/Kolors"
INSTANCE_DIR='dataset/EkuUekura'
OUTPUT_DIR="./trained_models/EkuUekura_lora"
cfg_file=dreambooth/default_config.yaml

accelerate launch --config_file ${cfg_file} dreambooth/train.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --class_data_dir=$CLASS_DIR \
    --instance_prompt=""\
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5 \
    --text_encoder_lr=5e-5 \
    --lr_scheduler="polynomial" \
    --lr_warmup_steps=500 \
    --rank=16 \
    --resolution=768 \
    --max_train_steps=5000 \
    --checkpointing_steps=1000 \
    --center_crop \
    --mixed_precision='fp16' \
    --seed=0 \
    --img_repeat_nums=1 \
    --sample_batch_size=2 \
    --gradient_checkpointing \
    --adam_weight_decay=1e-02 \
    --train_text_encoder \
    --checkpoints_total_limit=1\
    --report_to="wandb"
