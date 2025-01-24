

# Set paths to necessary directories
project_root="/home/u7946530/LGViT"
model_path="$project_root/models/deit_highway"

export HF_DATASETS_CACHE="/work/u7946530/hf_dataset_cache"
export HUGGINGFACE_HUB_CACHE="/work/u7946530/hf_cache/hub"
# Ensure the correct paths are included in PYTHONPATH without duplication
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
    export PYTHONPATH="$project_root:$PYTHONPATH"
fi

if [[ ":$PYTHONPATH:" != *":$model_path:"* ]]; then
    export PYTHONPATH="$PYTHONPATH:$model_path"
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
BACKBONE="ViT"
MODEL_TYPE="${BACKBONE}-base"
MODEL_NAME="facebook/deit-base-distilled-patch16-224"
DATASET="imagenet-1k" # cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k

EXIT_STRATEGY="confidence" # entropy, confidence, patience, patient_and_confident
TRAIN_STRATEGY="distillation_LGViT" # distillation_LGViT, distillation
HIGHWAY_TYPE="LGViT"

# Set CUDA devices and WandB project
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="${BACKBONE}_${DATASET}"

# Run the Python script using torch distributed launch







python -m torch.distributed.run --nproc_per_node=2 --master_port=29577 --nnodes=1 ../examples/run_distillation_qdeit.py \
    --report_to wandb \
    --threshold 0.8 \
    --run_name "${HIGHWAY_TYPE}_${TRAIN_STRATEGY}" \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path "$project_root/saved_models/ViT-base/imagenet-1k/LGViT/alternating_weighted" \
    --dataset_name $DATASET \
    --output_dir "$project_root/saved_models/$MODEL_TYPE/$DATASET/${HIGHWAY_TYPE}/stage2_${TRAIN_STRATEGY}/deit/quantized" \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --train_highway True \
    --num_early_exits 8 \
    --position_exits "[3,4,5,6,7,8,9,10]" \
    --exit_strategy $EXIT_STRATEGY \
    --train_strategy $TRAIN_STRATEGY \
    --loss_coefficient 0.3 \
    --homo_coefficient 0.01 \
    --hete_coefficient 0.01 \
    --learning_rate 5e-5 \
    --output_hidden_states False \
    --do_train \
    --do_eval \
    --num_train_epochs 30 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 1000 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --save_total_limit 3 \
    --seed 777 \
    --quant_bits 7 \
    --quantize_model True  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --per_layer_bits "[7, 7, 8, 7, 7, 7, 7, 7, 7, 6, 6, 8]"\
    --dataloader_num_workers 8

