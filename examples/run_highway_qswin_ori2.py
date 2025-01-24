#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from sqnr_calculation import calculate_sqnr, quantize
from conv import MinMaxQuantConv2d
from linear import MinMaxQuantLinear
from matmul import MinMaxQuantMatMul

import evaluate
import numpy as np
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
import torch.distributed as dist  # Added import for distributed operations
from quant_calib_swin import QuantCalibrator, HessianQuantCalibrator
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

#from models.swin_highway import SwinConfig, SwinHighwayForImageClassification

from models.swin_highway.configuration_qswin import SwinConfig

from models.swin_highway.modeling_highway_qswin import SwinHighwayForImageClassification
import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    denumpify_detensorize,
    EvalLoopOutput,
    get_last_checkpoint,
    has_length,
)

from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.utils import (
    ENV_VARS_TRUE_VALUES,
    check_min_version,
    send_example_telemetry,
    is_sagemaker_mp_enabled,
)
from transformers.data.data_collator import DataCollator
from transformers.integrations import WandbCallback, rewrite_logs, is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.versions import require_version
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch.nn.functional as F

def performance_gap_risk(highway_logits, final_layer_logits, labels, loss_fn=F.cross_entropy):
    """
    Calculate the performance gap risk as the difference in loss between an early exit and the final model.
    Args:
        highway_logits: Logits from the early exit.
        final_layer_logits: Logits from the final layer of the full model.
        labels: Ground truth labels.
        loss_fn: Loss function (default: Cross-Entropy Loss).
    Returns:
        Risk value: Positive value indicating the gap in performance.
    """
    loss_exit = loss_fn(highway_logits, labels)
    loss_full = loss_fn(final_layer_logits, labels)
    return loss_exit - loss_full

def consistency_risk(highway_logits, final_layer_logits):
    """
    Calculate the consistency risk based on KL divergence between softmax outputs of early exit and final model.
    Args:
        highway_logits: Logits from the early exit.
        final_layer_logits: Logits from the final layer of the full model.
    Returns:
        KL divergence value as risk.
    """
    softmax_exit = F.log_softmax(highway_logits, dim=-1)
    softmax_full = F.softmax(final_layer_logits, dim=-1)
    return F.kl_div(softmax_exit, softmax_full, reduction="batchmean")


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    exit_strategy: str = field(
        default='entropy',
        metadata={
            "help": "choose one exit_strategy: entropy, confidence, patience"
        }
    )

    train_strategy: str = field(
        default='normal',
        metadata={
            "help": "choose one train_strategy: normal, weighted, alternating"
        }
    )

    threshold: float = field(
        default=0.8,
        metadata={
            "help": "threshold"
        }
    ) 
    
    num_early_exits: str = field(
        default='[0,1,6,1]',
        metadata={
            "help": "number of exits"
        }
    )

    position_exits: Optional[str] = field(
        default=None,
        metadata={"help": "The position of the exits"}
    )

    highway_type: Optional[str] = field(
        default='linear',
        metadata={
            "help": "choose one highway_type: linear, conv1_1, conv1_2, conv1_3, conv2_1, attention"
        }
    )


    output_hidden_states: bool = field(
        default=False,
        metadata={"help": "whether output_hidden_states and use feature distillation" }
    )


    model_name_or_path: str = field(
        default="microsoft/swin-base-patch4-window7-224",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    full_model_name_or_path: str = field(
        default="microsoft/swin-base-patch4-window7-224",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        }
    )
    quantize_model: bool = field(
        default=False,
        metadata={"help": "Perform post-training quantization."}
    )
    calibration_num_samples: int = field(
        default=1024,
        metadata={"help": "Number of samples for calibration."}
    )
    calibration_batch_size: int = field(
        default=32,
        metadata={"help": "Calibration batch size."}
    )
    per_w_layer_bits: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of weight quantization bits per layer, e.g., '[8,8,8,8,8,8]'."
        }
    )
    per_a_layer_bits: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of activation quantization bits per layer, e.g., '[8,8,8,8,8,8]'."
        }
    )
    quant_bits: int = field(default=8, metadata={"help": "quant_bits for SQNR test."}),
    sqnr: bool =field(default=False, metadata={"help": "Whether execute SQNR test or not."}),
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["fine_label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def swin_block_forward_hook(module, input, output):
    print(f"swin_block_forward_hook called for module {module}")
    if not hasattr(module, 'raw_input'):
        module.raw_input = []
    if not hasattr(module, 'raw_out'):
        module.raw_out = []
    module.raw_input.append(input[0].detach())
    if isinstance(output, tuple):
        output = output[0]
    module.raw_out.append(output.detach())
    
def linear_forward_hook(module, input, output):
    print(f"linear_forward_hook called for module {module}")
    if not hasattr(module, 'raw_input'):
        module.raw_input = []
    if not hasattr(module, 'raw_out'):
        module.raw_out = []
    module.raw_input.append(input[0].detach())
    module.raw_out.append(output.detach())
    
def conv2d_forward_hook(module, input, output):
    print(f"conv2d_forward_hook called for module {module}")
    if not hasattr(module, 'raw_input'):
        module.raw_input = []
    if not hasattr(module, 'raw_out'):
        module.raw_out = []
    module.raw_input.append(input[0].detach())
    module.raw_out.append(output.detach())

def matmul_forward_hook(module, input, output):
    print(f"matmul_forward_hook called for module {module}")
    if not hasattr(module, 'raw_input'):
        module.raw_input = []
    if not hasattr(module, 'raw_out'):
        module.raw_out = []
    module.raw_input.append(input[0].detach())
    module.raw_out.append(output.detach())


def create_calibration_dataset(dataset, num_samples=1024):
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    calibration_data = dataset.select(indices)
    calibration_dataset = DatasetDict({'calibration': calibration_data})
    return calibration_dataset

def set_quant_mode(model, mode):
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            if 'classifier' in name or 'highway' in name or 'early_exit' in name:
                continue  # Do not set mode for excluded layers
            module.mode = mode
            
def compute_risks(model, full_model, dataloader, candidate_thresholds, loss_fn=F.cross_entropy):
    model.eval()
    full_model.eval()
    device = next(model.parameters()).device
    full_model.to(device)

    # Dictionary to store total risk and counts for each lambda
    total_risk_map = {lam: 0.0 for lam in candidate_thresholds}
    count_map = {lam: 0 for lam in candidate_thresholds}

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["pixel_values"], batch["labels"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Compute the full model logits once (final layer)
            final_outputs = full_model(pixel_values=inputs, disable_early_exits=True)
            final_logits = final_outputs.logits

            # For each candidate threshold
            for lam in candidate_thresholds:
                # Set the global threshold in the model
                model.set_global_threshold(lam)

                # Run the model in normal mode (with early exits triggered if confidence ≥ lam)
                early_outputs = model(pixel_values=inputs)
                early_logits = early_outputs.logits

                # Compute performance gap risk = E[ℓ(o_λ, y) - ℓ(o_L, y)]
                # Here ℓ is cross-entropy by default
                risk_diff = (loss_fn(early_logits, labels) - loss_fn(final_logits, labels)).item()
                batch_size = labels.size(0)
                total_risk_map[lam] += risk_diff * batch_size
                count_map[lam] += batch_size

    # Compute average risk
    risks = {}
    for lam in candidate_thresholds:
        if count_map[lam] > 0:
            risks[lam] = total_risk_map[lam] / count_map[lam]
        else:
            risks[lam] = 0.0  # No samples processed (unlikely), default to 0.

    return risks

  
def pick_threshold_empirical(risks, epsilon):
    # Sort thresholds by value
    for lam in sorted(risks.keys()):
        print("lam",lam)
        if risks[lam] <= epsilon:
            return lam
    # If none found, default to lam=1
    return 1.0

class TrainerwithExits(Trainer):
    def __init__(self, *args, sqnr =False,  target_block=None, quant_bits=8, model_args=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.quant_bits = quant_bits      # 設置量化的位元數
        self.sqnr = sqnr 
        self.model_args = model_args  
    def compute_bops_swin(self, model_config, eval_exit_counts):
        """
        Compute total BOPs and average BOPs per sample based on the Swin model configuration and evaluation exit counts.

        Args:
            model_config: A config object containing Swin model parameters.
            eval_exit_counts: A dict mapping exit layer indices (1-based) to counts.
                            For example: {3: 121, 4: 69, ..., 12: 3487}

        Returns:
            total_bops: Total BOPs across all samples.
            avg_bops_per_sample: Average BOPs per sample.
            bops_per_exit_layer: Dict mapping exit layer indices to cumulative BOPs up to that layer.
        """

        # Extract necessary parameters from model_config
        image_size = model_config.image_size  # e.g., 224
        patch_size = model_config.patch_size  # e.g., 4
        num_channels = model_config.num_channels  # e.g., 3
        embed_dim = model_config.embed_dim  # e.g., 96
        depths = model_config.depths  # List of depths per stage, e.g., [2, 2, 6, 2]
        num_heads = model_config.num_heads  # List of num_heads per stage, e.g., [3, 6, 12, 24]
        window_size = model_config.window_size  # e.g., 7
        mlp_ratio = model_config.mlp_ratio  # e.g., 4.0
        per_w_layer_bits = model_config.per_w_layer_bits  # List of bit-widths per layer
        per_a_layer_bits = model_config.per_a_layer_bits
        # Compute the number of patches
        H = W = image_size
        patch_H = H // patch_size
        patch_W = W // patch_size
        num_patches = patch_H * patch_W

        # Compute MACs for Patch Embedding Layer
        # Convolution parameters
        C_in = num_channels
        C_out = embed_dim
        K = patch_size
        H_out = H // patch_size
        W_out = W // patch_size

        # MACs for Patch Embedding
        macs_patch_embed = H_out * W_out * C_out * (C_in * K * K)

        # Assume full precision (32 bits) for patch embedding layer
        w_bit_patch_embed = 32
        a_bit_patch_embed = 32
        bops_patch_embed = macs_patch_embed * (w_bit_patch_embed * a_bit_patch_embed)

        # Initialize cumulative BOPs
        cumulative_bops = bops_patch_embed
        cumulative_bops_per_layer = [cumulative_bops]  # Index corresponds to exit layer index

        layer_idx = 0  # To keep track of per_layer_bits index

        # Loop over stages
        for stage_idx, depth in enumerate(depths):
            D = embed_dim * (2 ** stage_idx)
            H_out = H_out // (2 ** max(stage_idx - 1, 0))
            W_out = W_out // (2 ** max(stage_idx - 1, 0))
            N = H_out * W_out  # Number of tokens

            # Number of windows
            window_area = window_size * window_size
            num_windows = (H_out // window_size) * (W_out // window_size)

            for block_idx in range(depth):
                # Get per-layer bit-widths
                w_bit = per_w_layer_bits[layer_idx]
                a_bit = per_a_layer_bits[layer_idx]

                # **Compute MACs for Attention Layers**

                # QKV projections: 3 Linear layers
                macs_qkv = N * D * D * 3

                # Attention scores: matmul of Q and K^T within each window
                macs_attn_scores = num_windows * (window_area * D // num_heads[stage_idx]) * (window_area)

                # Attention context: matmul of attention_probs and V
                macs_attn_context = macs_attn_scores  # Same as attention scores

                # Output projection
                macs_output = N * D * D

                macs_attention_total = macs_qkv + macs_attn_scores + macs_attn_context + macs_output

                bops_attention = macs_attention_total * (w_bit * a_bit)

                # **Compute MACs for MLP Layers**

                macs_intermediate = N * D * int(D * mlp_ratio)
                macs_output_mlp = N * int(D * mlp_ratio) * D

                macs_mlp_total = macs_intermediate + macs_output_mlp

                bops_mlp = macs_mlp_total * (w_bit * a_bit)

                # **Total BOPs for this block**

                bops_block = bops_attention + bops_mlp

                cumulative_bops += bops_block
                cumulative_bops_per_layer.append(cumulative_bops)

                layer_idx += 1  # Move to the next bit-width

            # **Compute MACs for Patch Merging Layer (if not the last stage)**

            if stage_idx < len(depths) - 1:
                # After downsampling, spatial dimensions are halved
                H_out = H_out // 2
                W_out = W_out // 2
                N = H_out * W_out

                D_in = D * 4  # Due to concatenation of 2x2 neighboring patches
                D_out = D * 2

                # Linear layer in Patch Merging
                macs_patch_merging = N * D_in * D_out

                # Use the same bit-width as the last block
                w_bit = per_w_layer_bits[layer_idx - 1]
                a_bit = per_a_layer_bits[layer_idx - 1]

                bops_patch_merging = macs_patch_merging * (w_bit * a_bit)

                cumulative_bops += bops_patch_merging
                cumulative_bops_per_layer.append(cumulative_bops)

        # Now, use eval_exit_counts to compute total BOPs
        total_samples = sum(eval_exit_counts.values())
        total_bops = 0

        for exit_layer_idx, count in eval_exit_counts.items():
            # exit_layer_idx is 1-based, cumulative_bops_per_layer is 0-based (includes initial bops_patch_embed)
            cumulative_bops_up_to_exit = cumulative_bops_per_layer[exit_layer_idx]
            total_bops += count * cumulative_bops_up_to_exit

        avg_bops_per_sample = total_bops / total_samples if total_samples > 0 else 0

        # For reporting, create a dict mapping exit layers to cumulative BOPs
        bops_per_exit_layer = {}
        for exit_layer_idx in eval_exit_counts.keys():
            cumulative_bops_up_to_exit = cumulative_bops_per_layer[exit_layer_idx]
            bops_per_exit_layer[exit_layer_idx] = cumulative_bops_up_to_exit

        return total_bops, avg_bops_per_sample, bops_per_exit_layer
    def compute_loss(self, model, inputs, return_outputs=False):

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs, disable_early_exits=False)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, modified to handle only one early exit scenario per sample.
        This records the single early exit prediction and label whenever a sample exits early.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if args.deepspeed and not self.deepspeed:
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if self.model_args.per_w_layer_bits:
            logger.info(f"  Quantization setting = {self.model_args.per_w_layer_bits}")

        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Initialize dictionaries to store predictions and labels per exit layer
        ee_preds_per_layer = {}
        ee_labels_per_layer = {}
        # Lists to store exit layers for samples that took an early exit
        all_exit_layers = []
        # Lists to store predictions and labels for samples that successfully exited early
        ee_preds = []
        ee_labels = []

        observed_num_examples = 0

        for step, inputs in enumerate(dataloader):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size

            loss, logits, labels, exit_layer, all_exit_logits = self.prediction_step(
                self.model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

            # If a sample exits early, record its prediction and label
            if exit_layer is not None and labels is not None:
                #print("exit_layer =", exit_layer)
                num_labels = self.model.config.num_labels
                _, index = logits.view(-1, num_labels).max(dim=-1)
                pred_class = index.item()
                true_label = labels.item()
                
                    # Ensure the exit_layer key is initialized
                if exit_layer not in ee_preds_per_layer:
                    ee_preds_per_layer[exit_layer] = []
                    ee_labels_per_layer[exit_layer] = []

                ee_preds_per_layer[exit_layer].append(pred_class)
                ee_labels_per_layer[exit_layer].append(true_label)
                all_exit_layers.append(exit_layer)

            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            # Host accumulation
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Accumulation steps
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        # Final host accumulation
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        eval_dataset = getattr(dataloader, "dataset", None)
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Compute main metrics
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        if len(all_exit_layers) > 0:
            all_exit_layers = np.array(all_exit_layers)
            for i in range(24):
                metrics[f"counts_exit_layer_{i+1}"] = 0
            unique, counts = np.unique(all_exit_layers, return_counts=True)
            for i in range(len(unique)):
                metrics[f"counts_exit_layer_{unique[i]}"] = int(counts[i])

            metrics["average_exit_layer"] = all_exit_layers.mean().item()
            metrics["speed-up"] = 24 / all_exit_layers.mean().item()
        else:
            all_exit_layers = np.array([])

        for layer, preds in ee_preds_per_layer.items():
            labels = ee_labels_per_layer[layer]
            preds_array = np.array(preds)
            labels_array = np.array(labels)
            accuracy = (preds_array == labels_array).mean()
            if layer !=24 :
                metrics[f"{metric_key_prefix}_exit_{layer}_accuracy"] = float(accuracy)

        exit_counts = {}
        for i in range(1, 25):
            key = f"counts_exit_layer_{i}"
            if key in metrics:
                exit_counts[i] = metrics[key]
            else:
                exit_counts[i] = 0

        total_bops, avg_bops_per_sample, bops_per_exit_layer = self.compute_bops_swin(self.model.config, exit_counts)
        metrics[f"{metric_key_prefix}_total_bops"] = total_bops
        metrics[f"{metric_key_prefix}_avg_bops_per_sample"] = avg_bops_per_sample

        # Prefix all keys
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], List[torch.Tensor]]:
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        # labels may be popped when computing the loss
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs, disable_early_exits=False)

        # Extract final logits and exit_layer
        if isinstance(outputs, (dict, ModelOutput)):
            logits = outputs.logits
            exit_layer = outputs.exit_layer
            all_highway_exits = outputs.all_highway_exits
        else:
            # In case of exception or different structure
            logits = outputs[0]
            exit_layer = outputs.exit_layer if hasattr(outputs, 'exit_layer') else None
            all_highway_exits = outputs.all_highway_exits if hasattr(outputs, 'all_highway_exits') else None

        # Gather all exit logits
        all_exit_logits = []
        all_exit_layers = []
        if all_highway_exits is not None:
            # all_highway_exits is a tuple of tuples, each like (logits, ...)
            for exit_tuple in all_highway_exits:
                exit_logits = exit_tuple[0]
                #exit_layer = exit_tuple[-1]  # Assuming you added it as the last element
                all_exit_logits.append(exit_logits)
                #all_exit_layers.append(exit_layer)
        # Detach final logits as well
        logits = logits.detach() if logits is not None else None
        labels = labels.detach() if labels is not None and isinstance(labels, torch.Tensor) else labels

        if prediction_loss_only:
            return (loss, None, None, exit_layer, all_exit_logits)

        return (loss, logits, labels, exit_layer, all_exit_logits)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification_highway", model_args, data_args)
    
    quant_bits = model_args.quant_bits
    confidence = model_args.threshold  # Assuming confidence is stored in threshold
    per_a_layer_bits = model_args.per_a_layer_bits
    per_w_layer_bits = model_args.per_w_layer_bits
    if per_a_layer_bits is not None:
        per_a_layer_bits_str = per_a_layer_bits.strip("[]").replace(",", "_").replace(" ", "")
        per_w_layer_bits_str = per_w_layer_bits.strip("[]").replace(",", "_").replace(" ", "")
    else:
        per_a_layer_bits_str = "default"  # Use a fallback if per_layer_bits is not set

    # Construct the output directory name
    output_dir_name = f"Swin_quant_{quant_bits}_conf_{confidence}_w_bits_{per_w_layer_bits_str}_a_bits_{per_a_layer_bits_str}"

    # Update the output_dir in TrainingArguments
    training_args.output_dir = output_dir_name
    #training_args.evaluation_strategy = 'no'
    training_args.eval_accumulation_steps = 1
    training_args.dataloader_num_workers = 0
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            task="image-classification",
        )

    # If we don't have a validation split, split off a percentage of train as validation.
    # data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    sqnr = model_args.sqnr
    quant_bits = model_args.quant_bits

    if "validation" in dataset.keys():
        data_args.train_val_split = None
    elif "valid" in dataset.keys():
        data_args.train_val_split = None
        dataset["validation"] = dataset["valid"]
    elif "test" in dataset.keys():
        data_args.train_val_split = None
        dataset["validation"] = dataset["test"]
    else:
        pass


    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["fine_label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    # breakpoint()

    if model_args.per_w_layer_bits is not None:
        # Remove brackets and whitespace
        per_w_layer_bits_str = model_args.per_w_layer_bits.strip('[]').replace(' ', '')
        per_w_layer_bits = [int(bit) for bit in per_w_layer_bits_str.split(',')]
        per_a_layer_bits_str = model_args.per_a_layer_bits.strip('[]').replace(' ', '')
        per_a_layer_bits = [int(bit) for bit in per_a_layer_bits_str.split(',')]

    else:
        per_w_layer_bits = [model_args.quant_bits] * config.num_hidden_layers
        per_a_layer_bits = [model_args.quant_bits] * config.num_hidden_layers

    if training_args.do_train:
        config = SwinConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
            finetuning_task="image-classification",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            threshold=model_args.threshold,
            exit_strategy=model_args.exit_strategy,
            train_strategy=model_args.train_strategy,
            num_early_exits=model_args.num_early_exits,
            position_exits=model_args.position_exits,
            highway_type=model_args.highway_type,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = SwinConfig.from_pretrained(
            model_args.model_name_or_path,
            quant_bits=model_args.quant_bits,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
            finetuning_task="image-classification",
            threshold=model_args.threshold,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            exit_strategy=model_args.exit_strategy,
            num_early_exits=model_args.num_early_exits,
            per_a_layer_bits=per_a_layer_bits,
            per_w_layer_bits=per_w_layer_bits,

        )

    total_optimization_steps = int(len(dataset['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs)

    model = SwinHighwayForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    full_model = SwinHighwayForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    
    full_model.eval()
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    # Define torchvision transforms to be applied to each image.
    image_processor.size['shortest_edge'] = 224
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["img"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["img"]]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    # Initalize our trainer
    if model_args.quantize_model:
        # Create Calibration Dataset
        calibration_dataset = create_calibration_dataset(
            dataset["train"], model_args.calibration_num_samples
        )
        calibration_dataset = calibration_dataset["calibration"]
        calibration_dataset.set_transform(val_transforms)

        # Create DataLoader for Calibration
        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=model_args.calibration_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Collect Quantization Modules
        wrapped_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, (MinMaxQuantConv2d, MinMaxQuantLinear, MinMaxQuantMatMul)):
                wrapped_modules[name] = module
        
        # Move Model to Device
        model = model.to(device=training_args.device)

        # Perform Calibration
        quant_calibrator = QuantCalibrator(
            net=model,
            wrapped_modules=wrapped_modules,
            calib_loader=calibration_dataloader,
            sequential=False,
            batch_size=model_args.calibration_batch_size
        )
                #quant_calibrator = QuantCalibrator(net=model,wrapped_modules=wrapped_modules,calib_loader=calibration_dataloader,sequential=False,batch_size=model_args.calibration_batch_size) # 16 is too big for ViT-L-16

        quant_calibrator.batching_quant_calib()

        # Set Quantization Mode
        set_quant_mode(model, 'quant_forward')
        candidate_thresholds = np.linspace(0, 1, 21)  # 21 thresholds: 0.0, 0.05, 0.1, ... 1.0
        epsilon = 0.05  # your chosen tolerated risk
        risks = compute_risks(model, full_model, calibration_dataloader, candidate_thresholds, loss_fn=F.cross_entropy)
        chosen_lambda = pick_threshold_empirical(risks, epsilon)
        print(f"Chosen threshold λ = {chosen_lambda}")

        # Set the chosen threshold in the model
        model.set_global_threshold(chosen_lambda)
        
    print('do_eval:', training_args.do_eval)
    


    # After calibrating thresholds, we have a flat list of thresholds for all exits in all stages.

        
    trainer = TrainerwithExits(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        sqnr=sqnr,
        quant_bits=quant_bits,
        data_collator=collate_fn,
        model_args=model_args,
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    """
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, MinMaxQuantLinear):
            hooks.append(module.register_forward_hook(linear_forward_hook))
        elif isinstance(module, MinMaxQuantConv2d):
            hooks.append(module.register_forward_hook(conv2d_forward_hook))
        elif isinstance(module, MinMaxQuantMatMul):
            hooks.append(module.register_forward_hook(matmul_forward_hook))
        elif isinstance(module, SwinLayer):
            hooks.append(module.register_forward_hook(swin_block_forward_hook))
        else:
            pass
    """

        
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()