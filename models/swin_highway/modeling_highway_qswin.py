from dataclasses import dataclass
from collections import Iterable
from typing import Optional, Set, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F


from transformers.modeling_outputs import BaseModelOutputWithPooling

from modeling_qswin import SwinEmbeddings, SwinLayer, SwinPatchMerging, SwinPreTrainedModel, SwinPatchEmbeddings
from models.swin_highway.highway import SwinHighway, SwinHighway_v2, ViT_EE_Highway
from configuration_qswin import SwinConfig
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Any

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


@dataclass
class SwinHighwayOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    all_highway_exits: Optional[Any] = None
    exit_layer: Optional[int] = None
    block_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SwinModelOutput(BaseModelOutputWithPooling):
    """
    Swin Model's outputs that also contain block hidden states.

    Args:
        block_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` containing the hidden states from each block.
    """
    block_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
def CrossEntropy(outputs, targets, temperature):
    log_softmax_outputs = F.log_softmax(outputs / temperature, dim=1)
    softmax_targets = F.softmax(targets / temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    x = torch.softmax(x, dim=-1)  # softmax normalized prob distribution
    return -torch.sum(x * torch.log(x), dim=-1)  # entropy calculation on probs: -\sum(p \ln(p))



def confidence(x):
    # x: torch.Tensor, logits BEFORE softmax
    softmax = torch.softmax(x, dim=-1)
    return torch.max(softmax, dim=1)[0]  # Returns tensor of shape (batch_size,)


def prediction(x):
    # x: torch.Tensor, logits BEFORE softmax
    softmax = torch.softmax(x, dim=-1)
    return torch.argmax(softmax)


class SwinStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample, stage_index, global_layer_counter):
        super().__init__()
        self.config = config
        self.dim = dim
        self.stage_index = stage_index
        
        self.num_early_exits = eval(config.num_early_exits)[stage_index]
        self.exit_strategy = config.exit_strategy
        self.train_strategy = config.train_strategy
        self.global_layer_counter = global_layer_counter
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = SwinLayer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                layer_idx=self.global_layer_counter[0],  # Assign the current global layer index
            )
            self.blocks.append(layer)
            self.global_layer_counter[0] += 1  # Increment the global layer counter
        

        self.init_highway()
        self.set_early_exit_positon()
        self.set_early_exit_threshold(self.config.threshold)
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def init_highway(self):
        config = self.config
        if config.highway_type == 'linear':
            self.highway = nn.ModuleList([SwinHighway(config, stage=self.stage_index+1) for _ in range(self.num_early_exits) ])
        elif config.highway_type == 'vit':
            self.highway = nn.ModuleList(
                [ViT_EE_Highway(config, stage=self.stage_index + 1) for _ in range(self.num_early_exits)])
        elif config.highway_type == 'LGViT':
            if self.stage_index == 1:
                self.highway = nn.ModuleList(
                    [SwinHighway_v2(config, stage=self.stage_index + 1, highway_type='conv1_1') for _ in
                     range(self.num_early_exits)])
            elif self.stage_index == 2:
                self.highway = nn.ModuleList(
                    [SwinHighway_v2(config, stage=self.stage_index + 1, highway_type='conv1_1'),
                     SwinHighway_v2(config, stage=self.stage_index + 1, highway_type='conv2_1'),
                     SwinHighway_v2(config, stage=self.stage_index + 1, highway_type='conv2_1'),
                     SwinHighway_v2(config, stage=self.stage_index + 1, highway_type='attention_r1'),
                     SwinHighway_v2(config, stage=self.stage_index + 1, highway_type='attention_r1'),
                     SwinHighway_v2(config, stage=self.stage_index + 1, highway_type='attention_r2'),
                     ])
            elif self.stage_index == 3:
                self.highway = nn.ModuleList(
                    [SwinHighway_v2(config, stage=self.stage_index + 1, highway_type='attention_r2')])

    def set_early_exit_threshold(self, x=None):
        
        if self.exit_strategy == 'entropy':
            self.early_exit_threshold = [0.65 for _ in range(self.num_early_exits)]
        elif self.exit_strategy == 'confidence':
            self.early_exit_threshold = [0.75 for _ in range(self.num_early_exits)]
        elif self.exit_strategy == 'patience':
            self.early_exit_threshold = (3,)
        elif self.exit_strategy == 'patient_and_confident':
            self.early_exit_threshold = [0.8 for _ in range(self.num_early_exits)]
            self.early_exit_threshold.append(2)

        if x is not None:
            if (type(x) is float) or (type(x) is int):
                for i in range(len(self.early_exit_threshold)):
                    self.early_exit_threshold[i] = x
            else:
                self.early_exit_threshold = x

    def set_early_exit_positon(self):
        position_exits = self.config.position_exits
        stage = self.stage_index
        
        if position_exits is not None and isinstance(position_exits, Iterable):
            self.position_exits = eval(self.config.position_exits)[stage]
            if len(self.position_exits) != self.num_early_exits:
                raise ValueError(
                    "Lengths of config.position_exits and num_early_exits do not match, which can lead to poor training results!")

        print(f'Stage{stage+1}: the exits are in position: ', self.position_exits)
        self.position_exits = {int(position) - 1: index for index, position in enumerate(self.position_exits)}
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        pct=None,
        pred=None,
        disable_early_exits: bool = False,
        final_layer_logits=None,  # Pass the final logits for risk comparison
        labels = None

    ) -> Tuple[torch.Tensor]:
        
        height, width = input_dimensions
        all_highway_exits = ()
        all_exits_logits = [] 
        all_block_hidden_states = []
        
        if self.exit_strategy == 'patience':
            # store the number of times that the predictions remain “unchanged”
            pct = pct
            pred = pred
        elif self.exit_strategy == 'patient_and_confident':
            # store the number of times that the predictions remain confident in consecutive layers
            pct = pct
        
        for i, layer_module in enumerate(self.blocks):

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            hidden_states = layer_outputs[0]
            all_block_hidden_states.append(hidden_states)
            #print('------------------------------------------------layer_outputs------------------------------------------------\n',layer_outputs[1:])
            #print('------------------------------------------------layer_outputs_size------------------------------------------------\n',len(layer_outputs))

            current_outputs = (hidden_states,)
        
            if i in self.position_exits:
                highway_exit = self.highway[self.position_exits[i]](current_outputs)
                
            # logits, pooled_output
            
            # inference stage
            #disable_early_exits= False
            #print('------------------disable_early_exits-----------------\n',disable_early_exits)
            if i in self.position_exits and not disable_early_exits:
                if not self.training:
                    highway_logits = highway_exit[0]
                    # * entropy strategy
                    if self.exit_strategy == 'entropy':
                        highway_entropy = entropy(highway_logits)
                        highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                        all_highway_exits = all_highway_exits + (highway_exit,)
                        if highway_entropy < self.early_exit_threshold[self.position_exits[i]]:
                            new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                            raise HighwayException(new_output, self.stage_index, i)
                    
                    # * confidence strategy
                    
                    elif self.exit_strategy == 'confidence':
                        highway_confidence = confidence(highway_logits)  # Tensor of shape (batch_size,)
                        highway_exit = highway_exit + (highway_confidence,)
                        all_highway_exits = all_highway_exits + (highway_exit,)
                        # Compute risks and decide whether to exit
                        
                        #if total_risk is not None  (total_risk > self.early_exit_threshold[self.position_exits[i]]).all():
                        if (highway_confidence > self.early_exit_threshold[self.position_exits[i]]).all():
                       # if (highway_confidence > self.config.global_threshold).all():
                             #(highway_confidence > self.early_exit_threshold[self.position_exits[i]]).all()
                            #print(f'i={i},layer_module={layer_module}\nhighway logits:{highway_logits},highway conf. :{highway_confidence}\n',)
                            new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                            
                            raise HighwayException(new_output, self.stage_index, i)
                        
                    # * patience strategy
                    elif self.exit_strategy == 'patience':
                        highway_prediction = prediction(highway_logits)
                        highway_exit = highway_exit + (highway_prediction,)
                        all_highway_exits = all_highway_exits + (highway_exit,)

                        if pct == 0:
                            pred = highway_prediction
                            pct += 1
                        else:
                            if pred == highway_prediction:
                                pct +=1
                            else:
                                pct = 1
                                pred = highway_prediction

                        if pct == self.early_exit_threshold[0]:
                            new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                            raise HighwayException(new_output, self.stage_index, i)
                    
                    # * patient and confident strategy
                    elif self.exit_strategy == 'patient_and_confident':
                        highway_entropy = entropy(highway_logits)
                        highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                        all_highway_exits = all_highway_exits + (highway_exit,)

                        if highway_entropy < self.early_exit_threshold[self.position_exits[i]]:
                            pct += 1
                        else:
                            pct = 0

                        if pct == self.early_exit_threshold[-1]:
                            new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                            raise HighwayException(new_output, self.stage_index, i)
                else:
                    all_highway_exits = all_highway_exits + (highway_exit,)
        
        # downsampling
        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]

        if self.exit_strategy == 'patience' or self.exit_strategy == 'patient_and_confident':
            stage_outputs = stage_outputs + (all_highway_exits,) + ((pct, pred),)
        else:
            stage_outputs = stage_outputs + (all_highway_exits,)
        stage_outputs = stage_outputs + (all_block_hidden_states,)
        # hidden_states, hidden_states_before_downsampling, output_dimensions, (attentions), highway_exits, all_block_hidden_states
        return stage_outputs

class SwinEncoder(nn.Module):

    def __init__(self, config:SwinConfig, grid_size):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.global_layer_counter = [0]  # Start counting from 1
        self.layers = nn.ModuleList(
            [
                SwinStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=SwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                    stage_index=i_layer,
                    global_layer_counter=self.global_layer_counter,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        if config.exit_strategy == 'patience' or config.exit_strategy == 'patient_and_confident':
            self.pct = 0
            self.pred = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        #head_mask_sqnr: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        disable_early_exits: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_early_exits = []
        all_block_hidden_states = []  # List to store hidden states from all blocks

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            #for j, block in enumerate(layer_module.blocks):
            #    layer_head_mask = head_mask_sqnr[i][j]
            #    layer_outputs = block(
            #       hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition, self.pct, self.pred
            #    )
            layer_head_mask = head_mask[i]

            if self.config.exit_strategy == 'patience' or self.config.exit_strategy == 'patient_and_confident':
                layer_outputs = layer_module(
                    hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition, self.pct, self.pred,disable_early_exits =disable_early_exits
                )
                self.pct, self.pred = layer_outputs[-2]
                layer_outputs = layer_outputs[:-2]
            else:
                layer_outputs = layer_module(
                        hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition, disable_early_exits =disable_early_exits
                    )
        
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            
            stage_block_hidden_states = layer_outputs[-1]
            all_block_hidden_states.extend(stage_block_hidden_states)
            
            all_early_exits.append(layer_outputs[-2])

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:-1]

        return hidden_states, all_hidden_states, all_early_exits, all_block_hidden_states
        # return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)


SWIN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SwinConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SWIN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class SwinModel(SwinPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = SwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        disable_early_exits: bool = False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))
        #print("------------------------------ len(self.config.depths)------------------------------\n",len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            disable_early_exits= disable_early_exits,
        )

        sequence_output = encoder_outputs[0]
        all_hidden_states = encoder_outputs[1] if output_hidden_states else None
        all_early_exits = encoder_outputs[-2]
        all_block_hidden_states = encoder_outputs[-1]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)
            
        
        return SwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=None,  # Add attentions if you handle them
            block_hidden_states=all_block_hidden_states,
        )
       # head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)

        #return head_outputs + encoder_outputs[1:]


class HighwayException(Exception):
    def __init__(self, message, stage, exit_layer):
        self.message = message
        self.stage = stage
        self.exit_layer = exit_layer  # start form 1!
        



class SwinHighwayForImageClassification(SwinPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.num_labels = config.num_labels
        self.depths = config.depths
        self.exit_strategy = config.exit_strategy
        self.train_strategy = config.train_strategy
        self.loss_coefficient = config.loss_coefficient
        self.feature_loss_coefficient = config.feature_loss_coefficient
        self.position_exits = [4, 7, 10, 13, 16, 19, 22, 23]
        self.global_threshold = 1.0  # default no early exit
        self.config.global_threshold = 1.0
        #print(f'position_exits:{self.position_exits}')
        
        self.step = 0
        
        self.swin = SwinModel(config)
        
        # Classifier head
        self.classifier = (
            nn.Linear(self.swin.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()
    def set_exit_thresholds(self, thresholds):
        """
        Assign normalized thresholds to each early exit in the model.

        normalized_thresholds: a list of floats (0.0 to 1.0), one per exit in the order they appear.
        """
        exit_counter = 0
        for stage_idx, layer in enumerate(self.swin.encoder.layers):
            num_exits = len(getattr(layer, 'highway', []))
            if num_exits > 0:
                # Ensure layer.early_exit_threshold is defined and matches num_exits
                if not hasattr(layer, 'early_exit_threshold'):
                    layer.early_exit_threshold = [0.0 for _ in range(num_exits)]
                for ex_idx in range(num_exits):
                    # Map from [0, 1] to [0.5, 1.0]
                    mapped_threshold = (thresholds[exit_counter] )
                    layer.early_exit_threshold[ex_idx] = mapped_threshold
                    exit_counter += 1        
    def set_exit_norm_thresholds(self, normalized_thresholds):
        """
        Assign normalized thresholds to each early exit in the model.

        normalized_thresholds: a list of floats (0.0 to 1.0), one per exit in the order they appear.
        """
        exit_counter = 0
        for stage_idx, layer in enumerate(self.swin.encoder.layers):
            num_exits = len(getattr(layer, 'highway', []))
            if num_exits > 0:
                # Ensure layer.early_exit_threshold is defined and matches num_exits
                if not hasattr(layer, 'early_exit_threshold'):
                    layer.early_exit_threshold = [0.0 for _ in range(num_exits)]
                for ex_idx in range(num_exits):
                    # Map from [0, 1] to [0.5, 1.0]
                    mapped_threshold = 0.5 + (normalized_thresholds[exit_counter] * 0.4)
                    #mapped_threshold = (normalized_thresholds[exit_counter] )
                    layer.early_exit_threshold[ex_idx] = mapped_threshold
                    exit_counter += 1

        
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        disable_early_exits: bool = False,
    ):
        exit_layer = None
        
        try:
            outputs = self.swin(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                disable_early_exits=disable_early_exits,
                
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
            all_block_hidden_states=outputs.block_hidden_states,

            logits = self.classifier(pooled_output)
            exit_layer = sum(self.depths)
        except HighwayException as e:
            outputs = e.message
            exit_stage = e.stage
            exit_layer = e.exit_layer
            #print('exit_stage---------------------',exit_stage)
            #print('exit_layer---------------------',exit_layer)
            exit_layer = sum(self.depths[:exit_stage]) + exit_layer
            logits = outputs[0]
            hidden_states = None  # Adjust as necessary
            total_loss = None  # Set total_loss to None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    
                    max_value, max_index = logits.view(-1, self.num_labels).max(dim=-1)
                    
                    #print('max_value=',max_value,"max_index=",max_index," labels.view(-1)=", labels.view(-1))
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
                total_loss = loss  # Assign loss to total_loss
            else:
                total_loss = None  # Ensure total_loss is defined
            all_highway_exits = outputs[-2]  # Assuming it's the second[] last element in the tuple
            all_block_hidden_states=outputs[-1]
            return SwinHighwayOutput(
                loss=total_loss,  # loss might be None
                logits=logits,
                hidden_states=hidden_states,
                attentions=None,
                all_highway_exits=all_highway_exits,
                exit_layer=exit_layer,
                block_hidden_states=all_block_hidden_states,
            )

        # Compute loss
        loss = None
        total_loss = None
        if labels is not None:
            # Compute main loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            # Compute highway losses if training
            if self.training:
                highway_losses = []
                for stage, all_highway_exit in enumerate(outputs.all_highway_exits):
                    if not all_highway_exit:
                        continue
                    for index, highway_exit in enumerate(all_highway_exit):
                        highway_logits = highway_exit[0]
                        # Compute highway_loss
                        if self.config.problem_type == "regression":
                            loss_fct = MSELoss()
                            highway_loss = loss_fct(highway_logits.squeeze(), labels.squeeze())
                        elif self.config.problem_type == "single_label_classification":
                            loss_fct = CrossEntropyLoss()
                            highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                        elif self.config.problem_type == "multi_label_classification":
                            loss_fct = BCEWithLogitsLoss()
                            highway_loss = loss_fct(highway_logits, labels)
                        highway_losses.append(highway_loss)
                # Aggregate losses
                if self.train_strategy == 'normal':
                    total_loss = (sum(highway_losses) + loss) / (len(highway_losses) + 1)
                elif self.train_strategy == 'weighted':
                    highway_losses = [highway_losses[i] * coeff for i, coeff in enumerate(self.position_exits)]
                    total_loss = (sum(highway_losses) + loss * 24) / (sum(self.position_exits) + 24)
                elif self.train_strategy == 'alternating':
                    if self.step % 2 == 0:
                        total_loss = loss
                    else:
                        total_loss = (sum(highway_losses) + loss) / (len(highway_losses) + 1)
                    self.step += 1
                elif self.train_strategy == 'alternating_weighted':
                    if self.step % 2 == 0:
                        total_loss = loss
                    else:
                        highway_losses = [highway_losses[i] * coeff for i, coeff in enumerate(self.position_exits)]
                        total_loss = (sum(highway_losses) + loss * 24) / (sum(self.position_exits) + 24)
                    self.step += 1
                else:
                    total_loss = loss
            else:
                total_loss = loss

        # Prepare the outputs
        #print('exit_layer=',exit_layer)
        return SwinHighwayOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=None,
            all_highway_exits=outputs.all_highway_exits if hasattr(outputs, 'all_highway_exits') else None,
            exit_layer=exit_layer,
            block_hidden_states=all_block_hidden_states,
            
        )


class SwinHighwayForImageClassification_distillation(SwinPreTrainedModel):
    def __init__(self, config: SwinConfig):
        super(SwinHighwayForImageClassification_distillation, self).__init__(config)

        self.config = config
        self.num_labels = config.num_labels
        self.depths = config.depths
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.exit_strategy = config.exit_strategy
        self.train_strategy = config.train_strategy
        self.loss_coefficient = config.loss_coefficient
        self.feature_loss_coefficient = config.feature_loss_coefficient

        self.stage = 0

        self.swin = SwinModel(config)

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1)

        self.classifier = (
            nn.Linear(self.swin.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        self.position_exits = [4, 7, 10, 13, 16, 19, 22, 23]

        self.post_init()

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_layer=-1,
            disable_early_exits: bool = False
    ):

        embedding_output = self.swin.embeddings(pixel_values)
        # Common code for both training and inference
        outputs = None
        logits = None
        loss = None
        exit_layer = None
        all_highway_exits = None
        if self.training:
            hidden_states, input_dim = embedding_output
            hidden_list = []
            with torch.no_grad():
                for i, stage_module in enumerate(self.swin.encoder.layers):
                    height, width = input_dim
                    for j, layer_module in enumerate(stage_module.blocks):
                        layer_head_mask = head_mask[j] if head_mask is not None else None
                        layer_outputs = layer_module(hidden_states, input_dim, layer_head_mask)
                        hidden_states = layer_outputs[0]
                        if j in stage_module.position_exits:
                            hidden_list.append((hidden_states,))
                    hidden_states_before_downsampling = hidden_states
                    if stage_module.downsample is not None:
                        height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
                        output_dim = (height, width, height_downsampled, width_downsampled)
                        hidden_states = stage_module.downsample(hidden_states_before_downsampling, input_dim)
                    else:
                        output_dim = (height, width, height, width)
                    input_dim = (output_dim[-2], output_dim[-1])
                if self.config.train_strategy == 'distillation':
                    sequence_output = self.layernorm(hidden_states)
                    pooled_output = self.pooler(sequence_output.transpose(1, 2))
                    pooled_output = torch.flatten(pooled_output, 1)
                    teacher_logits = self.classifier(pooled_output)

            if self.config.train_strategy == 'distillation':
                highway_losses = []
                distillation_losses = []
                all_highway_exits = ()
                n = 0
                for i, stage_module in enumerate(self.swin.encoder.layers):
                    for j, layer_module in enumerate(stage_module.blocks):
                        if j in stage_module.position_exits:
                            index = stage_module.position_exits[j]
                            highway_exit = stage_module.highway[index](hidden_list[n])
                            all_highway_exits = all_highway_exits + (highway_exit,)
                            highway_logits = highway_exit[0]

                            loss_fct = CrossEntropyLoss()
                            highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                            highway_losses.append(highway_loss)

                            # * soft distillation
                            T = 2
                            highway_distill_loss = F.kl_div(
                                F.log_softmax(highway_logits / T, dim=1),
                                F.log_softmax(teacher_logits / T, dim=1),
                                reduction='sum',
                                log_target=True
                            ) * (T * T) / highway_logits.numel()

                            distillation_losses.append(highway_distill_loss)

                            n += 1

                distill_coef = self.loss_coefficient
                highway_losses = [highway_losses[index] * coeff for index, coeff in enumerate(self.position_exits)]

                loss_all = (1 - distill_coef) * sum(highway_losses) / sum(self.position_exits) + distill_coef * sum(
                    distillation_losses) / len(distillation_losses)
                #outputs = (loss_all,)
                outputs = SwinHighwayOutput(
                    loss=loss_all,
                    logits=teacher_logits,
                    hidden_states=None,
                    attentions=None,
                    all_highway_exits=all_highway_exits,
                    exit_layer=None,
                    block_hidden_states=None,
                )


        else:
            #exit_layer = None

            try:
                outputs = self.swin(
                    pixel_values,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    disable_early_exits=disable_early_exits
                )

                #pooled_output = outputs[0]
                pooled_output = outputs.pooler_output
                logits = self.classifier(pooled_output)
                #outputs = (logits,) + outputs[2:-1]  # logits, all_highway_exits
                #outputs = (logits,) + outputs[2:]
                outputs = SwinHighwayOutput(
                    loss=None,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    all_highway_exits=all_highway_exits,
                    exit_layer=None,
                    block_hidden_states=outputs.block_hidden_states,
                )

            except HighwayException as e:
                outputs = e.message
                exit_stage = e.stage
                
                
                exit_layer = e.exit_layer
                exit_layer = sum(self.depths[:exit_stage]) + exit_layer
                logits = outputs[0]
                #logits = outputs.logits
                all_highway_exits = outputs[-1]
                #all_highway_exits = outputs.all_highway_exits
                """
                print(f"Type of logits: {type(logits)}")
                if isinstance(logits, torch.Tensor):
                    print(f"Shape of logits: {logits.shape}")
                else:
                    print(f"Contents of logits: {logits}")
                """
                outputs = SwinHighwayOutput(
                    loss=None,
                    logits=logits,
                    hidden_states=None,  # Adjust as necessary
                    attentions=None,
                    all_highway_exits=all_highway_exits,
                    exit_layer=exit_layer,
                    block_hidden_states=None,
                )

                    

            if self.exit_strategy == 'confidence':
                original_score = confidence(logits)
            else:
                raise ValueError(
                    "Please select one of the exit strategies:entropy, confidence, patience, patient_and_confident")

            highway_score = []
            highway_logits_all = []
            loss = None


            
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs.loss = loss
                #outputs = (loss,) + outputs
            outputs.exit_layer = sum(self.depths) if exit_layer is None else exit_layer
            outputs.logits = logits
            #exit_layer = sum(self.depths) if exit_layer == None else exit_layer
            #outputs = outputs[:-1] + ((original_score, highway_score), exit_layer)

        return outputs
    
