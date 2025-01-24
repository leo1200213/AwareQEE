







# Activate conda environment
#!/bin/bash




# Activate conda environment
conda activate lgvit




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




BACKBONE="ViT"
MODEL_TYPE="${BACKBONE}-base"
MODEL_NAME="facebook/deit-base-distilled-patch16-224"
DATASET="imagenet-1k" # cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k




EXIT_STRATEGY="confidence" # entropy, confidence, patience, patient_and_confident
TRAIN_STRATEGY="distillation_LGViT" # distillation_LGViT, distillation
HIGHWAY_TYPE="LGViT"




QUANTIZE_MODEL=True
# Set CUDA device
#export CUDA_VISIBLE_DEVICES=0,1




# Uncomment if using wandb for logging
# export WANDB_PROJECT=${BACKBONE}_${DATANAME}_eval




# Run evaluation script
export CUDA_VISIBLE_DEVICES=0,1




# Uncomment if using wandb for logging
# export WANDB_PROJECT=${BACKBONE}_${DATANAME}_eval




# Run evaluation script


'''

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_1.0_bits_6_6_6_6_6_6_6_6_6_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.5  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_1.0_bits_6_6_6_6_6_6_6_6_6_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.6  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_1.0_bits_6_6_6_6_6_6_6_6_6_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.7  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_1.0_bits_6_6_6_6_6_6_6_6_6_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.8  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_1.0_bits_6_6_6_6_6_6_6_6_6_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.9  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]" \


python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_7_6_5_6_6_5_5_5"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.5  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 7, 6, 5, 6, 6, 5, 5, 5]" \
  
python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_7_6_5_6_6_5_5_5"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.6  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 7, 6, 5, 6, 6, 5, 5, 5]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_7_6_5_6_6_5_5_5"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.7  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 7, 6, 5, 6, 6, 5, 5, 5]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_7_6_5_6_6_5_5_5"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.9  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 7, 6, 5, 6, 6, 5, 5, 5]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_7_6_5_6_6_5_5_5"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 1.0  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 7, 6, 5, 6, 6, 5, 5, 5]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_6_6_5_6_5_5_5_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.5  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 6, 6, 5, 6, 5, 5, 5, 7]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_6_6_5_6_5_5_5_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.6  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 6, 6, 5, 6, 5, 5, 5, 7]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_6_6_5_6_5_5_5_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.7  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 6, 6, 5, 6, 5, 5, 5, 7]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_6_conf_0.8_bits_6_7_7_7_6_6_5_6_5_5_5_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.9  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  6\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[6, 7, 7, 7, 6, 6, 5, 6, 5, 5, 5, 7]" \







  python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_1.0_bits_7_7_7_7_7_7_7_7_7_7_7_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.5  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]" \
'''

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_1.0_bits_7_7_7_7_7_7_7_7_7_7_7_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.6  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]" \
    
'''
  python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_1.0_bits_7_7_7_7_7_7_7_7_7_7_7_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.7  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]" \

  python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_1.0_bits_7_7_7_7_7_7_7_7_7_7_7_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.8  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]" \

  python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_1.0_bits_7_7_7_7_7_7_7_7_7_7_7_7"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.9  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_8_8_8_8_7_6_7_7_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.5  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 8, 8, 8, 8, 7, 6, 7, 7, 6, 6, 6]" \


python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_8_8_8_8_7_6_7_7_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.6  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 8, 8, 8, 8, 7, 6, 7, 7, 6, 6, 6]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_8_8_8_8_7_6_7_7_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.7  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 8, 8, 8, 8, 7, 6, 7, 7, 6, 6, 6]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_8_8_8_8_7_6_7_7_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 1.0  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 8, 8, 8, 8, 7, 6, 7, 7, 6, 6, 6]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/work/u7946530/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_8_8_8_8_7_6_7_7_6_6_6"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.9  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 8, 8, 8, 8, 7, 6, 7, 7, 6, 6, 6]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/home/u7946530/LGViT/scripts/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_7_8_7_7_7_7_7_7_6_6_8"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.5  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 8, 7, 7, 7, 7, 7, 7, 6, 6, 8]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/home/u7946530/LGViT/scripts/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_7_8_7_7_7_7_7_7_6_6_8"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.6  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 8, 7, 7, 7, 7, 7, 7, 6, 6, 8]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/home/u7946530/LGViT/scripts/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_7_8_7_7_7_7_7_7_6_6_8"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.7  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 8, 7, 7, 7, 7, 7, 7, 6, 6, 8]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/home/u7946530/LGViT/scripts/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_7_8_7_7_7_7_7_7_6_6_8"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 0.9  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 8, 7, 7, 7, 7, 7, 7, 6, 6, 8]" \

python -m torch.distributed.run --nproc_per_node=2 --master_port=29566 --nnodes=1 ../examples/run_highway_qdeit.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --model_name_or_path "/home/u7946530/LGViT/scripts/Deit_qae_quant_imagenet-1k_7_conf_0.8_bits_7_7_8_7_7_7_7_7_7_6_6_8"\
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --threshold 1.0  \
    --seed 777 \
    --report_to wandb \
    --quant_bits  7\
    --quantize_model $QUANTIZE_MODEL  \
    --calibration_num_samples 32 \
    --calibration_batch_size 4 \
    --calibration_steps 100 \
    --per_layer_bits "[7, 7, 8, 7, 7, 7, 7, 7, 7, 6, 6, 8]" \
 #--model_name_or_path "$project_root/saved_models/$MODEL_TYPE/$DATASET/${HIGHWAY_TYPE}/stage2_${TRAIN_STRATEGY}/"  \

 #--model_name_or_path  "/home/u7946530/LGViT/scripts/Deit_qae_quant_7_conf_0.8_bits_7_8_8_8_7_7_6_7_6_6_6_8" \

# --model_name_or_path "$project_root/saved_models/ViT-base/cifar100/LGViT/alternating_weighted/"\

'''