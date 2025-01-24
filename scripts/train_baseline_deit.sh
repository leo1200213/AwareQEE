#!/bin/bash

# Activate the conda environment
conda activate lgvit

# Set paths to necessary directories
project_root="/home/u7946530/LGViT"
model_path="$project_root/models/deit_highway"

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

BACKBONE=ViT
MODEL_TYPE=${BACKBONE}-base
MODEL_NAME=facebook/deit-base-distilled-patch16-224
DATASET=cifar100 # cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k

EXIT_STRATEGY=confidence # entropy, confidence, patience, patient_and_confident
TRAIN_STRATEGY=alternating_weighted # normal, weighted, alternating, distillation, alternating_weighted
HIGHWAY_TYPE=LGViT # linear, LGViT, vit, self_attention, conv_normal
ADDTIONAL_INFO=LGViT   # SDN, PABEE, PCEE, BERxiT, ViT-EE, LGViT

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=${BACKBONE}_${DATANAME}

python -m torch.distributed.run --nproc_per_node=2 --master_port=29557 --nnodes=1 ../examples/run_highway_deit.py \
    --report_to wandb \
    --threshold 0.8 \
    --run_name ${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${ADDTIONAL_INFO}_threshold${threshold} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../saved_models/$MODEL_TYPE/$DATASET/${ADDTIONAL_INFO}/$TRAIN_STRATEGY \
    --overwrite_output_dir False \
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --train_highway True \
    --exit_strategy $EXIT_STRATEGY \
    --train_strategy $TRAIN_STRATEGY \
    --num_early_exits 8 \
    --position_exits "[4,5,6,7,8,9,10,11]" \
    --highway_type $HIGHWAY_TYPE \
    --learning_rate 5e-5 \
    --output_hidden_states False \
    --do_train \
    --do_eval \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --save_total_limit 3 \
    --seed 777 \
    --ignore_mismatched_sizes True \
    --homo_loss_coefficient 0.01 \
    --hete_loss_coefficient 0.01
    


