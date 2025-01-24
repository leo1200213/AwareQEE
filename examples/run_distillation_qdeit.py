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

import evaluate
import numpy as np
import torch
import torch.nn as nn
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

#from models.deit_highway import DeiTImageProcessor, DeiTConfig, DeiTHighwayForImageClassification, DeiTHighwayForImageClassification_distillation
from conv import MinMaxQuantConv2d
from linear import MinMaxQuantLinear
from matmul import MinMaxQuantMatMul
from quant_calib import QuantCalibrator, HessianQuantCalibrator

from models.deit_highway.modeling_highway_qdeit import  DeiTHighwayForImageClassification, DeiTHighwayForImageClassification_distillation
from models.deit_highway import DeiTImageProcessor
from models.deit_highway.configuration_qdeit import DeiTConfig

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

class EvalEveryNEpochsCallback(TrainerCallback):
    def __init__(self, eval_every_n_epochs):
        self.eval_every_n_epochs = eval_every_n_epochs

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch is not None and (state.epoch + 1) % self.eval_every_n_epochs == 0:
            control.should_evaluate = True
            
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
    backbone: str = field(
        default='ViT',
        metadata={"help": "Choose one backbone: ViT, DeiT"}
    )

    train_highway: bool = field(
        default=False,
        metadata={"help": "Train highway"}
    )

    encoder_ensemble: bool = field(
        default=False,
        metadata={"help": "Whether to use encoder ensemble"}
    )

    exit_strategy: str = field(
        default='entropy',
        metadata={"help": "Choose one exit strategy: entropy, confidence, patience"}
    )

    train_strategy: str = field(
        default='normal',
        metadata={"help": "Choose one training strategy: normal, weighted, alternating"}
    )

    num_early_exits: int = field(
        default=4,
        metadata={"help": "Number of early exits"}
    )

    position_exits: Optional[str] = field(
        default=None,
        metadata={"help": "The position of the exits"}
    )

    highway_type: Optional[str] = field(
        default='linear',
        metadata={"help": "Choose one highway type: linear, conv1_1, conv1_2, conv1_3, conv2_1, attention"}
    )

    loss_coefficient: float = field(
        default=0.6,
        metadata={"help": "Coefficient of the distillation loss"}
    )

    homo_coefficient: float = field(
        default=0.01,
        metadata={"help": "Coefficient of the homogeneous distillation loss"}
    )

    hete_coefficient: float = field(
        default=0.01,
        metadata={"help": "Coefficient of the heterogeneous distillation loss"}
    )

    output_hidden_states: bool = field(
        default=False,
        metadata={"help": "Whether to output hidden states and use feature distillation"}
    )

    threshold: float = field(
        default=0.8,
        metadata={"help": "Threshold for early exit"}
    )

    model_name_or_path: str = field(
        default="facebook/deit-base-distilled-patch16-224",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from s3"}
    )

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )

    image_processor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    quantize_model: bool = field(
        default=False,
        metadata={"help": "Perform post-training quantization."}
    )
    calibration_num_samples: int = field(
        default=500,
        metadata={"help": "Number of samples for calibration."}
    )
    calibration_batch_size: int = field(
        default=32,
        metadata={"help": "Calibration batch size."}
    )
    quant_bits: int = field(default=8, metadata={"help": "quant_bits for SQNR test."}),
    per_layer_bits: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of quantization bits per layer, e.g., '[7,7,7,7,6,6,6,6,5,5,5,5]'."}
    )
    use_auth_token: bool = field(
        default=True,
        metadata={"help": "Use the token generated when running `huggingface-cli login` (necessary for private models)."}
    )

    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Enable loading a pretrained model whose head dimensions are different."}
    )

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
def collate_fn_img(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def set_quant_mode(model, mode):
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            if 'classifier' in name or 'highway' in name or 'early_exit' in name:
                continue  # Do not set mode for excluded layers
            module.mode = mode
            
def create_calibration_dataset(dataset, num_samples=32):
    print("------------------------------------start calibration------------------------------------")
    import numpy as np
    from datasets import DatasetDict

    # Randomly select num_samples indices from the dataset
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    # Select the calibration data
    calibration_data = dataset.select(indices)

    calibration_dataset = DatasetDict({'calibration': calibration_data})
    print("------------------------------------finish calibration------------------------------------\n", len(calibration_dataset['calibration']))

    return calibration_dataset

class TrainerwithExits(Trainer):
    def __init__(self, *args, sqnr =False,  target_block=None, quant_bits=8,model_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_bits = quant_bits      # 設置量化的位元數
        self.model_args = model_args  # Save model_args for use in other methods
    def compute_bops(self, model_config, eval_exit_counts):
        """
        Compute total BOPs and average BOPs per sample based on the model configuration and evaluation exit counts.

        Args:
            model_config: A config object containing model parameters.
            eval_exit_counts: A dict mapping exit layer indices (1-based) to counts.
                            For example: {3: 121, 4: 69, ..., 12: 3487}

        Returns:
            total_bops: Total BOPs across all samples.
            avg_bops_per_sample: Average BOPs per sample.
            bops_per_exit_layer: Dict mapping exit layer indices to cumulative BOPs up to that layer.
        """

        # Extract necessary parameters from model_config
        hidden_size = model_config.hidden_size  # e.g., 768
        num_hidden_layers = model_config.num_hidden_layers  # e.g., 12
        num_attention_heads = model_config.num_attention_heads  # e.g., 12
        intermediate_size = model_config.intermediate_size  # e.g., 3072
        per_layer_bits = model_config.per_layer_bits  # List of bit-widths per layer
        image_size = model_config.image_size  # e.g., 224
        patch_size = model_config.patch_size  # e.g., 16
        num_channels = model_config.num_channels  # e.g., 3

        # Compute the number of patches
        num_patches = (image_size // patch_size) * (image_size // patch_size)  # e.g., 14 * 14 = 196
        seq_length = num_patches + 2  # Adding 2 for cls and distillation tokens, so seq_length = 198

        # Attention head size
        head_dim = hidden_size // num_attention_heads  # e.g., 768 // 12 = 64

        # BOPs per layer will be stored here
        bops_per_layer = []

        # Compute BOPs for patch embedding layer
        # Assuming Conv2d with kernel_size=patch_size, stride=patch_size, in_channels=num_channels, out_channels=hidden_size
        # Number of MACs in patch embedding layer:
        #macs_patch_embed = (image_size // patch_size) * (image_size // patch_size) * \
                        #(patch_size * patch_size * num_channels) * hidden_size

        # Assume full precision (32 bits) for patch embedding layer
        #w_bit_patch_embed = 32
        #a_bit_patch_embed = 32
        #bops_patch_embed = macs_patch_embed * (w_bit_patch_embed * a_bit_patch_embed)

        # Now, for each transformer layer
        #cumulative_bops = bops_patch_embed  # Start with BOPs from patch embedding
        cumulative_bops = 0
        cumulative_bops_per_layer = [cumulative_bops]  # Store cumulative BOPs up to each layer (including previous layers)

        for layer_idx in range(num_hidden_layers):
            # Get per-layer bit-widths
            w_bit = per_layer_bits[layer_idx]
            a_bit = per_layer_bits[layer_idx]

            # Compute MACs and BOPs for attention sublayer

            # QKV projections: 3 Linear layers
            macs_qkv = 3 * seq_length * hidden_size * hidden_size

            # Attention scores: matmul of Q and K^T
            macs_attn_scores = num_attention_heads * seq_length * seq_length * head_dim

            # Attention context: matmul of attention_probs and V
            macs_attn_context = macs_attn_scores  # Same as attention scores

            # Output projection
            macs_output = seq_length * hidden_size * hidden_size

            macs_attention_total = macs_qkv + macs_attn_scores + macs_attn_context + macs_output

            bops_attention = macs_attention_total * (w_bit * a_bit)

            # Compute MACs and BOPs for MLP sublayer

            # First Linear layer (intermediate)
            macs_intermediate = seq_length * hidden_size * intermediate_size

            # Second Linear layer (output)
            macs_output_mlp = seq_length * intermediate_size * hidden_size

            macs_mlp_total = macs_intermediate + macs_output_mlp

            bops_mlp = macs_mlp_total * (w_bit * a_bit)

            # Total BOPs for this layer
            bops_layer = bops_attention + bops_mlp

            cumulative_bops += bops_layer
            cumulative_bops_per_layer.append(cumulative_bops)

            bops_per_layer.append(bops_layer)

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

        outputs = model(**inputs)
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
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
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

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_exit_layers = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels, exit_layer = self.prediction_step(model, inputs, prediction_loss_only,
                                                                    ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            # Update containers on host
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
            if exit_layer is not None:
                exit_layer = np.array([exit_layer])
                all_exit_layers = exit_layer if all_exit_layers is None else np.concatenate(
                    [all_exit_layers, exit_layer])

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
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

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
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
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if all_exit_layers is not None:
            for i in range(model.config.num_hidden_layers):
                metrics[f"counts_exit_layer_{i + 1}"] = 0
            unique, counts = np.unique(all_exit_layers, return_counts=True)
            for i in range(len(unique)):
                metrics[f"counts_exit_layer_{unique[i]}"] = int(counts[i])

            metrics["average_exit_layer"] = all_exit_layers.mean().item()

        metrics["speed-up"] = 12 / all_exit_layers.mean().item()
        
        exit_counts = {}
        for i in range(model.config.num_hidden_layers):
            key = f"counts_exit_layer_{i + 1}"
            if key in metrics:
                exit_counts[i + 1] = metrics[key]
            else:
                exit_counts[i + 1] = 0
        
        #Compute BOPs
        total_bops, avg_bops_per_sample, bops_per_exit_layer = self.compute_bops(model.config, exit_counts)

        # Add BOPs to metrics
        metrics[f"{metric_key_prefix}_total_bops"] = total_bops
        metrics[f"{metric_key_prefix}_avg_bops_per_sample"] = avg_bops_per_sample
        
        # Prefix all keys with metric_key_prefix + '_'
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
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference",
                                      ['entropies', 'exit_layer', 'hidden_states', 'attentions'])
            else:
                ignore_keys = ['entropies', 'exit_layer', 'hidden_states', 'attentions']

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():  # false
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                        print(outputs['exit_layer'])
                    else:
                        logits = outputs[1]
                        exit_layer = outputs[-1]
                # else:
                #     loss = None
                #     with self.compute_loss_context_manager():
                #         outputs = model(**inputs)
                #     if isinstance(outputs, dict):
                #         logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                #     else:
                #         logits = outputs
                #     # TODO: this needs to be fixed and made cleaner later.
                #     if self.args.past_index >= 0:
                #         self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1 and type(logits) == tuple:
            logits = logits[0]

        return (loss, logits, labels, exit_layer)


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
    per_layer_bits = model_args.per_layer_bits
    
    if per_layer_bits is not None:
        per_layer_bits_str = per_layer_bits.strip("[]").replace(",", "_").replace(" ", "")
    else:
        per_layer_bits_str = "default"  # Use a fallback if per_layer_bits is not set

    # Construct the output directory name
    output_dir_name = f"Deit_qae_quant_{data_args.dataset_name}_{quant_bits}_conf_{confidence}_bits_{per_layer_bits_str}"

    # Update the output_dir in TrainingArguments
    training_args.output_dir = output_dir_name
    training_args.evaluation_strategy = 'no'
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
        """dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            task="image-classification",
            use_auth_token=True if model_args.use_auth_token else None,
        )"""
        dataset = load_dataset(data_args.dataset_name)
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

# Check if the dataset has a validation set or needs to split from the train set
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

    # Split training data into train and validation if needed
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(test_size=data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Print available features to determine the correct label key
    print("Available features in train dataset:", dataset["train"].features)

    # Prepare label mappings by checking the available keys
    if "label" in dataset["train"].features:
        labels = dataset["train"].features["label"].names
    elif "labels" in dataset["train"].features:
        labels = dataset["train"].features["labels"].names
    else:
        # If 'label' or 'labels' are not present, try another approach
        # Assume there is a class feature in the dataset and iterate over possible features
        for key, feature in dataset["train"].features.items():
            if hasattr(feature, "names"):
                labels = feature.names
                break
        else:
            raise KeyError("The dataset does not contain any feature with label information.")

    # Create label2id and id2label mappings
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
    
    if model_args.per_layer_bits is not None:
        # Remove brackets and whitespace
        per_layer_bits_str = model_args.per_layer_bits.strip('[]').replace(' ', '')
        per_layer_bits = [int(bit) for bit in per_layer_bits_str.split(',')]
    else:
        per_layer_bits = [model_args.quant_bits] * config.num_hidden_layers
        
    config = DeiTConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        backbone=model_args.backbone,
        encoder_ensemble=model_args.encoder_ensemble,
        exit_strategy=model_args.exit_strategy,
        train_strategy=model_args.train_strategy,
        num_early_exits=model_args.num_early_exits,
        position_exits=model_args.position_exits,
        highway_type=model_args.highway_type,
        loss_coefficient=model_args.loss_coefficient,
        homo_coefficient=model_args.homo_coefficient,
        hete_coefficient=model_args.hete_coefficient,
        output_hidden_states=model_args.output_hidden_states,
        per_layer_bits=per_layer_bits,
        threshold=model_args.threshold,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    print('num_early_exits=================',model_args.num_early_exits) 
    print('position_exits=================',model_args.position_exits) 
    total_optimization_steps = int(
        len(dataset['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs)
    config.total_optimization_steps = total_optimization_steps

    model = DeiTHighwayForImageClassification_distillation.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        train_highway=model_args.train_highway,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    image_processor = DeiTImageProcessor.from_pretrained(
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
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch
    
    def train_transforms_img(example_batch):
        """Apply _train_transforms across a batch."""
        try:
            example_batch["pixel_values"] = [
                _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
            ]
        except Exception as e:
            logger.error(f"Error in train_transforms: {e}")
            logger.error(f"example_batch['img']: {example_batch['image']}")
            raise e
        return example_batch
    
    def val_transforms_img(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch
    
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        if data_args.dataset_name=='cifar100':
            dataset["train"].set_transform(train_transforms)
        else:
            dataset["train"].set_transform(train_transforms_img)
    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        if data_args.dataset_name=='cifar100':
            print('data_args.dataset_name:', data_args.dataset_name)
            dataset["validation"].set_transform(val_transforms)
        else:
            dataset["validation"].set_transform(val_transforms_img)
    # Initalize our trainer

    print('do_eval:', training_args.do_eval)
    
    if model_args.quantize_model:
        model.eval()
        """
        calibration_dataset = create_calibration_dataset(train_dataset, num_samples=1024)
        from torch.utils.data import DataLoader
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=32,  # Adjust the batch size as needed
            shuffle=False,
            num_workers=4,  # Adjust the number of workers as needed
            collate_fn=collate_fn,
        )
        """
        calibration_dataset = create_calibration_dataset(dataset['train'], model_args.calibration_num_samples)
        calibration_dataset = calibration_dataset["calibration"]
        calibration_dataset.set_transform(val_transforms)

        from torch.utils.data import DataLoader
        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=model_args.calibration_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    
        wrapped_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, (MinMaxQuantConv2d, MinMaxQuantLinear, MinMaxQuantMatMul)):
                wrapped_modules[name] = module
        #print('----------------------------------------------------------------------calibration_step1----------------------------------------------------------------------')
        model = model.to(device=training_args.device)
        
       # quant_calibrator = HessianQuantCalibrator(net=model,wrapped_modules=wrapped_modules,calib_loader=calibration_dataloader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
  
        """
        quant_calibrator = QuantCalibrator(
            net=model,
            wrapped_modules=wrapped_modules,
            calib_loader=calibration_dataloader,
            sequential=True  
        )
        """
        quant_calibrator = QuantCalibrator(net=model,wrapped_modules=wrapped_modules,calib_loader=calibration_dataloader,sequential=False,batch_size=model_args.calibration_batch_size) # 16 is too big for ViT-L-16

        #quant_calibrator = HessianQuantCalibrator(net=model,wrapped_modules=wrapped_modules,calib_loader=calibration_dataloader,sequential=False,batch_size=model_args.calibration_batch_size) # 16 is too big for ViT-L-16
       
        quant_calibrator.batching_quant_calib()
        #print('----------------------------------------------------------------------calibration_step1----------------------------------------------------------------------')
        
                

        # Set to Quant Forward Mode
        set_quant_mode(model, 'quant_forward')
    if data_args.dataset_name=='cifar100':
        trainer = TrainerwithExits(
            model=model,
            args=training_args,
            train_dataset=dataset["train"] if training_args.do_train else None,
            eval_dataset=dataset["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=image_processor,
            data_collator=collate_fn,
            quant_bits=quant_bits,
            model_args=model_args,
            callbacks=[EvalEveryNEpochsCallback(eval_every_n_epochs=10)]
        )
    else :
        trainer = TrainerwithExits(
            model=model,
            args=training_args,
            train_dataset=dataset["train"] if training_args.do_train else None,
            eval_dataset=dataset["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=image_processor,
            data_collator=collate_fn_img,
            quant_bits=quant_bits,
            model_args=model_args,
            callbacks=[EvalEveryNEpochsCallback(eval_every_n_epochs=10)]
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

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics[model.exit_strategy] = model.deit.encoder.early_exit_threshold[0]
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()