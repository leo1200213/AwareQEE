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
DATASET=imagenet-1k      # cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k
ADD_INFO=BERxiT 
EXIT_STRATEGY=confidence # entropy, confidence, patience, patient_and_confident
TRAIN_STRATEGY=distillation # distillation_LGViT, distillation
HIGHWAY_TYPE=LGViT

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=${BACKBONE}_${DATANAME}

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_distillation_swin.py \
    --report_to wandb \
    --run_name ${HIGHWAY_TYPE}_${TRAIN_STRATEGY} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path "$project_root/saved_models/$MODEL_TYPE/$DATASET/$ADD_INFO" \
    --dataset_name $DATASET \
    --output_dir "$project_root/saved_models/$MODEL_TYPE/$DATASET/${HIGHWAY_TYPE}/stage2_${TRAIN_STRATEGY}/swin/" \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --exit_strategy $EXIT_STRATEGY \
    --train_strategy $TRAIN_STRATEGY \
    --loss_coefficient 0.3 \
    --homo_loss_coefficient 0.01 \
    --hete_loss_coefficient 0.01 \
    --learning_rate 5e-5 \
    --do_train \
    --do_eval \
    --num_train_epochs 50 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --save_total_limit 3 \
    --seed 777 \
