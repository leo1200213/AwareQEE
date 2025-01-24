#!/bin/bash

# Activate the conda environment
conda activate lgvit

# Ensure that Hugging Face Datasets will cache to /work/u7946530/hf_dataset_cache
export HF_DATASETS_CACHE="/work/u7946530/hf_dataset_cache"
export HUGGINGFACE_HUB_CACHE="/work/u7946530/hf_cache/hub"

# Set paths to necessary directories
project_root="/home/u7946530/LGViT"
model_path="$project_root/models/deit_highway"

# Ensure the correct paths are included in PYTHONPATH without duplication
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
    export PYTHONPATH="$project_root:$PYTHONPATH"
fi

if [[ ":$PYTHONPATH:" != *":$model_path:"* ]]; then
    export PYTHONPATH="$model_path:$PYTHONPATH"
fi

# Debugging info
echo "---- PYTHONPATH ----"
echo $PYTHONPATH
echo "--------------------"
echo "---- Checking Python ----"
which python3
echo "--------------------"
echo "HF_DATASETS_CACHE = $HF_DATASETS_CACHE"

# Define dataset and other variables
DATASET="imagenet-1k"
DATASET_PATH="/work/u7946530"
BACKBONE="DeiT"
MODEL_NAME="facebook/deit-base-distilled-patch16-224"
MODEL_TYPE="DeiT"

# Run the Python script using torchrun
python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29557 --nnodes=1 ../examples/run_base_deit.py \

    --report_to wandb \
    --run_name "DeiT-base-run" \
    --dataset_name $DATASET \
    --dataset_path $DATASET_PATH \
    --backbone $BACKBONE \
    --model_name_or_path $MODEL_NAME \
    --output_dir ../saved_models/$MODEL_TYPE/$DATASET/base \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 777 \
    --ignore_mismatched_sizes=True
