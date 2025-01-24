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
TRAIN_STRATEGY=distillation
NUM_EXITS="[0, 1, 6, 1]"
EXIT_STRATEGY=confidence # entropy, confidence, patience, patient_and_confident
PAPER_NAME=BERxiT     # base, SDN, PABEE, PCEE, BERxiT, ViT-EE, LGViT

export CUDA_VISIBLE_DEVICES=0,1
# export WANDB_PROJECT=${BACKBONE}_${DATANAME}eval



  
'''
python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_6_conf_0.5_bits_7_7_7_7_7_7_6_6_6_5_5_5_5_5_6_6_6_6_7_6_5_5_5_7" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 6 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 0.5\
    --dythreshold False\
    --per_w_layer_bits "[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 6, 5, 5, 5, 7]"\
    --per_a_layer_bits "[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 6, 5, 5, 5, 7]"

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_6_conf_0.7_bits_7_7_7_7_7_6_6_6_6_5_5_5_5_5_7_6_6_6_7_6_5_5_5_7" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 6 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 0.7\
    --dythreshold False\
    --per_w_layer_bits "[7, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 7, 6, 6, 6, 7, 6, 5, 5, 5, 7]"\
    --per_a_layer_bits "[7, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 7, 6, 6, 6, 7, 6, 5, 5, 5, 7]"

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_6_conf_0.8_bits_6_7_6_7_6_6_6_6_6_5_5_5_5_5_7_7_6_7_7_7_5_5_5_7" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 6 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 0.8\
    --dythreshold False\
    --per_w_layer_bits "[6, 7, 6, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 7, 7, 6, 7, 7, 7, 5, 5, 5, 7]"\
    --per_a_layer_bits "[6, 7, 6, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 7, 7, 6, 7, 7, 7, 5, 5, 5, 7]"

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_6_conf_0.9_bits_6_6_6_7_6_6_6_6_5_5_5_5_5_6_7_7_7_7_7_7_5_5_5_7" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 6 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 0.9\
    --dythreshold False\
    --per_w_layer_bits "[6, 6, 6, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 5, 5, 5, 7]"\
    --per_a_layer_bits "[6, 6, 6, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 5, 5, 5, 7]"

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_6_conf_1.0_bits_5_6_6_6_6_5_5_5_5_5_6_6_6_6_7_7_7_7_7_7_7_5_5_7" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 6 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 1.0\
    --dythreshold False\
    --per_w_layer_bits "[5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7]"\
    --per_a_layer_bits "[5, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7]"

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_7_conf_0.5_bits_8_8_8_8_7_7_7_7_7_6_6_6_6_7_8_7_8_8_7_6_6_6_6_8" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 7 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 0.5\
    --dythreshold False\
    --per_w_layer_bits "[8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7, 8, 7, 8, 8, 7, 6, 6, 6, 6, 8]"\
    --per_a_layer_bits "[8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7, 8, 7, 8, 8, 7, 6, 6, 6, 6, 8]"

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530//Swin_qae_quant_7_conf_0.7_bits_7_8_8_8_7_7_7_7_7_6_6_6_6_7_8_8_8_8_7_6_6_6_6_8" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 7 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 0.7\
    --dythreshold False\
    --per_w_layer_bits "[7, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7, 8, 8, 8, 8, 7, 6, 6, 6, 6, 8]"\
    --per_a_layer_bits "[7, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7, 8, 8, 8, 8, 7, 6, 6, 6, 6, 8]"

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_7_conf_0.9_bits_7_8_8_8_7_7_7_7_7_6_6_7_6_7_8_8_8_8_6_6_6_6_6_8" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 7 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 0.9\
    --dythreshold False\
    --per_w_layer_bits "[7, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 7, 6, 7, 8, 8, 8, 8, 6, 6, 6, 6, 6, 8]"\
    --per_a_layer_bits "[7, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 7, 6, 7, 8, 8, 8, 8, 6, 6, 6, 6, 6, 8]"

python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_7_conf_1.0_bits_6_7_7_7_7_6_6_6_6_6_7_7_7_8_8_8_8_8_8_8_7_6_6_8" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 7 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 1.0\
    --dythreshold False\
    --per_w_layer_bits "[6, 7, 7, 7, 7, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 7, 6, 6, 8]"\
    --per_a_layer_bits "[6, 7, 7, 7, 7, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 7, 6, 6, 8]"

'''
python -m torch.distributed.run --nproc_per_node=2 --master_port=29527 --nnodes=1 ../examples/run_highway_qswin.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path  "/work/u7946530/Swin_qae_quant_7_conf_0.5_bits_8_8_8_8_7_7_7_7_7_6_6_6_6_7_8_7_8_8_7_6_6_6_6_8" \
    --full_model_name_or_path  "$project_root/saved_models/$MODEL_TYPE/$DATASET/$PAPER_NAME" \
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
    --sqnr False \
    --quant_bits 7 \
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --threshold 1.0\
    --dythreshold True\
    --per_w_layer_bits "[8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7, 8, 7, 8, 8, 7, 6, 6, 6, 6, 8]"\
    --per_a_layer_bits "[8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7, 8, 7, 8, 8, 7, 6, 6, 6, 6, 8]"