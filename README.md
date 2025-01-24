# MAQEE

This is the official PyTorch implementation of the paper **"MAQEE: Mutual Adaptive Quantization with Early Exiting"**.




## Requirements

### Installation

1. Clone the repository to your local machine.

2. Install PyTorch and the required dependencies:

```bash
conda create -n maqee python=3.9.13
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.26.0 datasets==2.9.0 evaluate==0.4.0 timm==0.6.13 wandb==0.14.2 ipykernel scikit-learn
```

3. Enter the `scripts` folder to execute the scripts for training and evaluation

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





---

## Hyperparameter Settings

In the scripts `train_distillation_qdeit.sh` and `train_distillation_qswin.sh`, the following hyperparameters can be adjusted:

- **`calibration_num_samples`**: Number of samples used during quantization calibration.
- **`calibration_batch_size`**: Batch size used during quantization calibration.
- **`per_layer_bits`**: Bit-width assigned to specific layers.
- **`dythreshold`**: Dynamically adjusts thresholds for each layer based on PGR and SQNR.

---





## Acknowledgments

This repository builds upon the foundational work of [LGViT](https://github.com/falcon-xu/LGViT) and [PTQ4ViT](https://github.com/hahnyuan/PTQ4ViT). We extend our sincere gratitude to the creators of these outstanding open-source projects for their invaluable contributions!








