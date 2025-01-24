# MAQEE

Official PyTorch implementation of "MAQEE: Mutual Adaptive Quantization with Early Existing"

## Usage

First, download the repository locally.


Then, install PyTorch and [transformers 4.26.0](https://github.com/huggingface/transformers)

```bash
conda create -n maqee python=3.9.13
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.26.0 datasets==2.9.0 evaluate==0.4.0 timm==0.6.13 wandb==0.14.2 ipykernel scikit-learn
```

Enter the `scripts` folder to execute the scripts for training and evaluation

```bash
cd ./scripts
```

- **train_base_deit.sh / train_base_swin.sh**

  This is for fine-tuning base models.

- **train_baseline_deit.sh / train_baseline_swin.sh**

  This is for fine-tuning *1st stage MAQEE* models and unquantized baseline models, which is our main .

- **train_distillation_deit.sh / train_distillation_swin.sh**

  This is for fine-tuning *2nd stage MAQEE* unquantized models.

- **train_distillation_qdeit.sh / train_distillation_qswin.sh**

  This is for fine-tuning *2nd stage MAQEE* quantized models, which is our main method.

- **eval_highway_deit.sh / eval_highway_swin.sh**

​		This is for evaluating fine-tuned unquantized models.

- **eval_highway_deit.sh / eval_highway_swin.sh**

​		This is for evaluating fine-tuned quantized models.

Before running the script, modify the `path` and `model_path` in the script to be appropriate.

### Training

To fine-tune a MAQEE backbone, run:

```bash
source train_base_deit.sh
```

To fine-tune a MAQEE models, run:

```bash
#no qunaitzed
source train_baseline_deit.sh
source train_distillation_deit.sh
#qunaitzed
source train_baseline_qdeit.sh
source train_distillation_qdeit.sh
```

### Evaluation

To evaluate a fine-tuned ViT, run:

```bash
#no qunaitzed
source eval_highway_deit.sh
#qunaitzed
source eval_highway_qdeit.sh
```



### Some Hyperparameters Settings






## Acknowledgments

This repository is built upon the [LGViT](https://github.com/falcon-xu/LGViT) and [PTQ4ViT](https://github.com/hahnyuan/PTQ4ViT). Thanks for these awesome open-source projects!


