conda activate lgvit

# Set paths to necessary directories
project_root="/home/u7946530/LGViT"
model_path="$project_root/models/swin_highway"

# Ensure the correct paths are included in PYTHONPATH without duplication
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
    export PYTHONPATH="$project_root:$PYTHONPATH"
fi

if [[ ":$PYTHONPATH:" != *":$model_path:"* ]]; then
    export PYTHONPATH="$model_path:$PYTHONPATH"
fi

BACKBONE=Swin
MODEL_TYPE=${BACKBONE}-baseline
MODEL_NAME=microsoft/swin-base-patch4-window7-224
# cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k
DATASET=cifar100

EXIT_STRATEGY=confidence # entropy, confidence, patience, patient_and_confident
TRAIN_STRATEGY=alternating # normal, weighted, alternating, distillation, alternating_weighted
HIGHWAY_TYPE=linear # linear, LGViT, vit,
ADD_INFO=BERxiT     # SDN, PABEE, PCEE, BERxiT, ViT-EE, LGViT
NUM_EXITS="[0, 1, 6, 1]"
POSITION_EXITS='[[], [2], [3, 6, 9, 12, 15, 18], [1]]'

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=${BACKBONE}_${DATASET}
export WANDB_WATCH=all


python -m torch.distributed.run --nproc_per_node=2 --master_port=29517 --nnodes=1 ../examples/run_highway_swin.py \
    --report_to wandb \
    --threshold 0.8 \
    --run_name "${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${ADD_INFO}" \
    --image_processor_name "$MODEL_NAME" \
    --config_name "$MODEL_NAME" \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET" \
    --output_dir "$project_root/saved_models/$MODEL_TYPE/$DATASET/$ADD_INFO" \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --exit_strategy "$EXIT_STRATEGY" \
    --train_strategy "$TRAIN_STRATEGY" \
    --num_early_exits "$NUM_EXITS" \
    --position_exits "$POSITION_EXITS" \
    --highway_type "$HIGHWAY_TYPE" \
    --learning_rate 5e-5 \
    --output_hidden_states False \
    --do_train \
    --do_eval \
    --num_train_epochs 100 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --save_total_limit 3 \
    --seed 777 \
    --ignore_mismatched_sizes True \
    --use_auth_token true
