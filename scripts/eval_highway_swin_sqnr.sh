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
DATASET=cifar100      # cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k

NUM_EXITS="[0, 1, 6, 1]"
EXIT_STRATEGY=confidence # entropy, confidence, patience, patient_and_confident
PAPER_NAME=BERxiT     # base, SDN, PABEE, PCEE, BERxiT, ViT-EE, LGViT
TRAIN_STRATEGY=distillation
export CUDA_VISIBLE_DEVICES=0,1
# export WANDB_PROJECT=${BACKBONE}_${DATANAME}eval

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_swin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/${HIGHWAY_TYPE}/stage2_${TRAIN_STRATEGY}/swin/" \
    --dataset_name $DATASET \
    --output_dir  ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/  \
    --remove_unused_columns False \
    --exit_strategy $EXIT_STRATEGY \
    --num_early_exits $NUM_EXITS \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --seed 777 \
    --report_to wandb \
    --use_auth_token False \
    --sqnr True\
    --quant_bits 8\
    --output_hidden_states True
'''
python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_swin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/${HIGHWAY_TYPE}/stage2_${TRAIN_STRATEGY}/swin/" \
    --dataset_name $DATASET \
    --output_dir  ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/  \
    --remove_unused_columns False \
    --exit_strategy $EXIT_STRATEGY \
    --num_early_exits $NUM_EXITS \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --seed 777 \
    --report_to wandb \
    --use_auth_token False \
    --sqnr True\
    --quant_bits 7\
    --output_hidden_states True

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_swin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/${HIGHWAY_TYPE}/stage2_${TRAIN_STRATEGY}/swin/" \
    --dataset_name $DATASET \
    --output_dir  ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/  \
    --remove_unused_columns False \
    --exit_strategy $EXIT_STRATEGY \
    --num_early_exits $NUM_EXITS \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --seed 777 \
    --report_to wandb \
    --use_auth_token False \
    --sqnr True\
    --quant_bits 6\
    --output_hidden_states True
'''