"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for ITM model
"""
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from .model import UniterPreTrainedModel, UniterModel
import pdb
from utils.heatmap import plot_attention_headmap
import numpy as np


class UniterForImageTextRetrieval(UniterPreTrainedModel):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, pairs_num, margin=0.2):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.margin = margin
        self.pairs_num = pairs_num
        self.apply(self.init_weights)

    def init_output(self):
        """ need to be called after from pretrained """
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def forward(self, batch, compute_loss=True, IAIS=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        if IAIS:
            gather_index = None
            txt_attn_masks = batch['txt_attn_masks']  # [sample_num, max_tl+max_nbb]
            img_attn_masks = batch['img_attn_masks']  # [sample_num, max_tl+max_nbb]
            sequence_output, self_attn_loss_per_layer = self.uniter(input_ids, position_ids,
                                                                    img_feat, img_pos_feat,
                                                                    attention_mask, gather_index, None,
                                                                    txt_attn_masks, img_attn_masks,
                                                                    output_all_encoded_layers=False,
                                                                    IAIS=IAIS,
                                                                    pairs_num=self.pairs_num)
        else:  # evaluation
            gather_index = batch['gather_index']
            sequence_output = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attention_mask, gather_index,
                                          output_all_encoded_layers=False)
        # sequence_output: [sample_num, max_tl+max_nbb, hidden_size(768)]
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)
        # rank_scores: [sample_num, 1]

        if compute_loss:
            # triplet loss
            rank_scores_sigmoid = torch.sigmoid(rank_scores)
            sample_size = batch['sample_size']
            scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
            pos = scores[:, :1]
            neg = scores[:, 1:]
            rank_loss = torch.clamp(self.margin + neg - pos, 0)
            # self-attn agree loss
            if IAIS:
                return rank_loss, self_attn_loss_per_layer
            else:
                return rank_loss
        else:
            return rank_scores


class UniterForImageTextRetrievalHardNeg(UniterForImageTextRetrieval):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2, hard_size=16):
        super().__init__(config, img_dim, hard_size + 1, margin)
        self.hard_size = hard_size

    def forward(self, batch, sample_from='t', compute_loss=True, IAIS=False):
        # expect same input_ids for all pairs
        batch_size = batch['attn_masks'].size(0)
        input_ids = batch['input_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        if sample_from == 't':
            if input_ids.size(0) == 1:
                batch['input_ids'] = input_ids.expand(batch_size, -1)
        elif sample_from == 'i':
            if img_feat.size(0) == 1:
                batch['img_feat'] = img_feat.expand(batch_size, -1, -1)  # copy img_feat for batch_size times
            if img_pos_feat.size(0) == 1:
                batch['img_pos_feat'] = img_pos_feat.expand(batch_size, -1, -1)
        else:
            raise ValueError()

        if self.training and compute_loss:
            with torch.no_grad():
                self.eval()
                scores = super().forward(batch, compute_loss=False)  # only evaluate
                hard_batch = self._get_hard_batch(batch, scores, sample_from, IAIS)
                self.train()
            return super().forward(hard_batch, compute_loss=True, IAIS=IAIS)
        else:
            return super().forward(batch, compute_loss)  # only evaluate

    def _get_hard_batch(self, batch, scores, sample_from='t', IAIS=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        hard_batch = {'sample_size': self.hard_size + 1}

        # NOTE first example is positive
        hard_indices = scores.squeeze(-1)[1:].topk(self.hard_size, sorted=False)[1] + 1
        indices = torch.cat([torch.zeros(1, dtype=torch.long,
                                         device=hard_indices.device),
                             hard_indices])  # [32]

        attention_mask = attention_mask.index_select(0, indices)

        if position_ids.size(0) != 1:
            position_ids = position_ids[:self.hard_size+1]

        if sample_from == 't':
            # cut to minimum padding
            max_len = attention_mask.sum(dim=1).max().item()
            max_i = max_len - input_ids.size(1)
            attention_mask = attention_mask[:, :max_len]
            img_feat = img_feat.index_select(0, indices)[:, :max_i, :]
            img_pos_feat = img_pos_feat.index_select(0, indices)[:, :max_i, :]
            # expect same input_ids for all pairs
            input_ids = input_ids[:self.hard_size+1]
        elif sample_from == 'i':
            input_ids = input_ids.index_select(0, indices)
            # expect same image features for all pairs
            img_feat = img_feat[:self.hard_size+1]
            img_pos_feat = img_pos_feat[:self.hard_size+1]
        else:
            raise ValueError()

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        if IAIS:
            txt_attn_masks = batch['txt_attn_masks']
            img_attn_masks = batch['img_attn_masks']
            if sample_from == 't':
                max_len = attention_mask.sum(dim=1).max().item()
                hard_batch['txt_attn_masks'] = txt_attn_masks[:, :max_len]
                hard_batch['img_attn_masks'] = img_attn_masks[:, :max_len]
            elif sample_from == 'i':
                hard_batch['txt_attn_masks'] = txt_attn_masks
                hard_batch['img_attn_masks'] = img_attn_masks
        else:
            if sample_from == 't':
                gather_index = batch['gather_index']
                gather_index = gather_index.index_select(0, indices)
                gather_index = gather_index[:, :max_len]
                hard_batch['gather_index'] = gather_index

        return hard_batch
