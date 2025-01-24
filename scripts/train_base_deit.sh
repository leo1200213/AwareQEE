
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

# Debugging: Print PYTHONPATH to ensure the paths are set correctly
echo "---- PYTHONPATH ----"
echo $PYTHONPATH
echo "--------------------"

# Ensure correct Python is being used and visible in the current environment
echo "---- Checking Python ----"
which python3
echo "--------------------"

# Define dataset and other variables
DATASET="cifar100"
BACKBONE="DeiT"
MODEL_NAME="facebook/deit-base-distilled-patch16-224"
MODEL_TYPE="DeiT"

# Run the Python script using torchrun
python3 -m torch.distributed.run  --nproc_per_node=1 --nnodes=1 ../examples/run_base_deit.py \
    --report_to wandb \
    --run_name "DeiT-base-run" \
    --dataset_name $DATASET \
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
