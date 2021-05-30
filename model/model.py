"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
import json
import logging
from io import open
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
import torch.nn.functional as F
from .layer import BertLayer, BertPooler
from utils.heatmap import plot_attention_headmap
import pdb
import sys
import numpy as np
import traceback

logger = logging.getLogger(__name__)


class UniterConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = UniterConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UniterPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, UniterConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                model.__class__.__name__,
                "\n\t".join(error_msgs)))
        return model


class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    def get_attention_probs(self, layer_module, hidden_states, attn_mask, row_b, row_l, col_b=None, col_l=None):
        attn = layer_module.attention.self.get_attention_probs(hidden_states,
                                                               attn_mask)  # [sample_num, attn_head_num, ?, ?]
        attn = torch.mul(attn, attn_mask)
        attn = torch.narrow(attn, 2, row_b, row_l)
        if col_b is None and col_l is None:
            col_b, col_l = row_b, row_l
        attn = torch.narrow(attn, 3, col_b, col_l)
        attn = torch.mean(attn, dim=1)
        return attn

    def iais_distributed(self, txt_attn, img_attn, t2i_attn, i2t_attn, modal):
        if modal == 'L':
            pseudo_txt_attn = torch.matmul(t2i_attn, i2t_attn)
            iais_loss = self.KLDivLoss(torch.log(txt_attn + 1e-6), pseudo_txt_attn) + self.KLDivLoss(
                torch.log(pseudo_txt_attn + 1e-6), txt_attn)
        elif modal == 'V':
            pseudo_img_attn = torch.matmul(i2t_attn, t2i_attn)
            iais_loss = self.KLDivLoss(torch.log(img_attn + 1e-6), pseudo_img_attn) + self.KLDivLoss(
                torch.log(pseudo_img_attn + 1e-6), img_attn)
        else:
            raise ValueError('error modal')
        return iais_loss

    def iais_singular(self, txt_attn, img_attn, cross_attn, length, modal):
        index = cross_attn.argmax(-1).detach().cpu().numpy().tolist()
        rows = [[i] * length for i in index]
        cols = [index] * length
        if modal == 'L':
            pseudo_txt_attn = nn.Softmax(dim=-1)(img_attn[rows, cols])
            iais_loss = self.KLDivLoss(txt_attn.log(), pseudo_txt_attn) + self.KLDivLoss(pseudo_txt_attn.log(),
                                                                                         txt_attn)
        elif modal == 'V':
            pseudo_img_attn = nn.Softmax(dim=-1)(txt_attn[rows, cols])
            iais_loss = self.KLDivLoss(img_attn.log(), pseudo_img_attn) + self.KLDivLoss(pseudo_img_attn.log(),
                                                                                         img_attn)
        else:
            raise ValueError('error modal')
        return iais_loss

    def forward(self, input_, attention_mask, txt_attn_mask=None, img_attn_mask=None,
                t2i_attn_mask=None, i2t_attn_mask=None, max_tl=0, max_nbb=0,
                output_all_encoded_layers=True, IAIS=False, pairs_num=3):
        all_encoder_layers = []
        self_attn_loss_per_layer = {}
        hidden_states = input_
        for i, layer_module in enumerate(self.layer):  # every layer_module is a bert_layer
            if IAIS and i == len(self.layer) - 1:
                gt_indices = torch.tensor(list(range(0, hidden_states.size(0), pairs_num)),
                                          dtype=torch.long, device=hidden_states.device)
                hidden_states_gt = hidden_states.index_select(0, gt_indices)
                txt_attn = self.get_attention_probs(layer_module, hidden_states_gt, txt_attn_mask, 1,
                                                    max_tl - 2)  # remove [cls] and [sep]
                img_attn = self.get_attention_probs(layer_module, hidden_states_gt, img_attn_mask, max_tl, max_nbb)
                t2i_attn = self.get_attention_probs(layer_module, hidden_states_gt, t2i_attn_mask, 1, max_tl - 2,
                                                    max_tl,
                                                    max_nbb)  # [sample_num, max_tl-2, max_nbb]
                i2t_attn = self.get_attention_probs(layer_module, hidden_states_gt, i2t_attn_mask, max_tl, max_nbb, 1,
                                                    max_tl - 2)  # [sample_num, max_nbb, max_tl-2]

                self_attn_loss_layer_i = torch.tensor(0, dtype=hidden_states.dtype, device=hidden_states.device)
                for j, (input_len, nbb) in enumerate(
                        zip(txt_attn_mask[:, 0, 1, :].sum(1), img_attn_mask[:, 0, max_tl, :].sum(1))):
                    input_len, nbb = int(input_len.item()), int(nbb.item())
                    if IAIS == 'L-singular':
                        iais_loss = self.iais_singular(txt_attn[j, :input_len, :input_len], img_attn[j, :nbb, :nbb],
                                                   t2i_attn[j, :input_len], input_len, 'L')
                    elif IAIS == 'V-singular':
                        iais_loss = self.iais_singular(txt_attn[j, :input_len, :input_len], img_attn[j, :nbb, :nbb],
                                                   i2t_attn[j, :nbb], nbb, 'V')
                    elif IAIS == 'L-distributed':
                        iais_loss = self.iais_distributed(txt_attn[j, :input_len, :input_len], img_attn[j, :nbb, :nbb],
                                                   t2i_attn[j, :input_len, :nbb], i2t_attn[j, :nbb, :input_len], 'L')
                    elif IAIS == 'V-distributed':
                        iais_loss = self.iais_distributed(txt_attn[j, :input_len, :input_len], img_attn[j, :nbb, :nbb],
                                                   t2i_attn[j, :input_len, :nbb], i2t_attn[j, :nbb, :input_len], 'V')
                    else:
                        raise ValueError("IAIS must in ['L-distributed', 'V-distributed', 'L-singular', 'V-singular']")

                    self_attn_loss_layer_i += iais_loss
                self_attn_loss_per_layer['self_attn_loss/layer_%s' % i] = self_attn_loss_layer_i / gt_indices.size(0)
                self_attn_loss_per_layer['self_attn_loss'] = self_attn_loss_per_layer['self_attn_loss/layer_%s' % i]
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        if IAIS:
            return all_encoder_layers, self_attn_loss_per_layer
        else:
            return all_encoder_layers


class UniterModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """

    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = UniterTextEmbeddings(config)
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(  # [sample_num, token_num, 768]
            input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(  # [sample_num, bb_max_num, 768]
            img_feat, img_pos_feat, img_masks, img_type_ids)
        if gather_index is not None:  # evaluation
            # align back to most compact input
            gather_index = gather_index.unsqueeze(-1).expand(  # [sample_num, ?, 768]
                -1, -1, self.config.hidden_size)
            embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),  # [sample_num, ?, 768]
                                            dim=1, index=gather_index)
        else:
            embedding_output = torch.cat([txt_emb, img_emb], dim=1)
        return embedding_output

    def extend_self_attn_mask(self, attention_mask):
        '''note this attention is 0-1'''
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = torch.matmul(attention_mask.permute(0, 1, 3, 2), attention_mask)
        return attention_mask

    def extend_cross_attn_mask(self, txt_attn_mask, img_attn_mask):
        txt_attn_mask = txt_attn_mask.unsqueeze(1).unsqueeze(2)
        txt_attn_mask = txt_attn_mask.to(dtype=next(self.parameters()).dtype)
        img_attn_mask = img_attn_mask.unsqueeze(1).unsqueeze(2)
        img_attn_mask = img_attn_mask.to(dtype=next(self.parameters()).dtype)
        t2i_attn_mask = torch.matmul(txt_attn_mask.permute(0, 1, 3, 2), img_attn_mask)
        i2t_attn_mask = torch.matmul(img_attn_mask.permute(0, 1, 3, 2), txt_attn_mask)
        return t2i_attn_mask, i2t_attn_mask

    def forward(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                attention_mask, gather_index=None, img_masks=None,
                txt_attn_mask=None, img_attn_mask=None,
                output_all_encoded_layers=True,
                IAIS=False,
                txt_type_ids=None, img_type_ids=None, pairs_num=3):
        '''
        input_ids: [sample_num, max_tl], position_ids: [1, max_tl]
        img_feat: [sample_num, max_nbb, 2048], img_pos_feat: [sample_num, max_nbb, 7]
        attention_mask: [sample_num, max_attn_len(max_tl+max_nbb)]
        '''
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2)  # [sample_num, 1, 1, max_attn_len(max_tl+max_nbb)]
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, img_masks, txt_type_ids, img_type_ids)

        if IAIS:  # train & IAIS
            assert txt_attn_mask is not None and img_attn_mask is not None
            extended_txt_attn_mask = self.extend_self_attn_mask(
                txt_attn_mask)  # [sample_num, 1, max_attn_len, max_attn_len]
            extended_img_attn_mask = self.extend_self_attn_mask(img_attn_mask)
            extended_t2i_attn_mask, extended_i2t_attn_mask = self.extend_cross_attn_mask(txt_attn_mask, img_attn_mask)

            encoded_layers, self_attn_loss_per_layer = self.encoder(
                embedding_output, extended_attention_mask,
                extended_txt_attn_mask, extended_img_attn_mask,
                extended_t2i_attn_mask, extended_i2t_attn_mask,
                input_ids.size(1), img_feat.size(1),
                output_all_encoded_layers=output_all_encoded_layers,
                IAIS=IAIS,
                pairs_num=pairs_num)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, self_attn_loss_per_layer
        else:  # evaluation
            encoded_layers = self.encoder(
                embedding_output, extended_attention_mask,
                output_all_encoded_layers=output_all_encoded_layers)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers
