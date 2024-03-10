import pandas as pd
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import List

import numpy as np
import pygtrie
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
import torch.nn as nn
import typer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from transformers.optimization import Adafactor

from chrisbase.data import AppTyper
from models.modified_T5 import ModifiedT5ForConditionalGeneration
from models.modified_mT5 import ModifiedMT5ForConditionalGeneration
from nlpbook.arguments import TrainerArguments

logger = logging.getLogger(__name__)
main = AppTyper()


def get_num(dataset_path, dataset, mode='entity'):  # mode: {entity, relation}
    return int(open(os.path.join(dataset_path, dataset.replace("'", ""), mode + '2id.txt')).readline().strip())


def read(dataset_path, dataset, filename):
    file_name = os.path.join(dataset_path, dataset.replace("'", ""), filename)
    with open(file_name) as file:
        lines = file.read().strip().split('\n')
    n_triples = int(lines[0])
    triples = []
    for line in lines[1:]:
        split = line.split(' ')
        for i in range(3):
            split[i] = int(split[i])
        triples.append(split)
    assert n_triples == len(triples), 'number of triplets is not correct.'
    return triples


def read_file(dataset_path, dataset, filename, mode='descrip'):
    id2name = []
    file_name = os.path.join(dataset_path, dataset.replace("'", ""), filename)
    with open(file_name, encoding='utf-8') as file:
        lines = file.read().strip('\n').split('\n')
    for i in range(1, len(lines)):
        ids, name = lines[i].split('\t')
        if mode == 'descrip':
            name = name.split(' ')
            name = ' '.join(name)
        id2name.append(name)
    return id2name


def read_name(dataset_path, dataset):
    ent_name_file = 'entityid2name.txt'
    rel_name_file = 'relationid2name.txt'
    ent_name_list = read_file(dataset_path, dataset, ent_name_file, 'name')
    rel_name_list = read_file(dataset_path, dataset, rel_name_file, 'name')
    return ent_name_list, rel_name_list


def construct_prefix_trie(ent_token_ids_in_trie):
    trie = pygtrie.Trie()
    for input_ids in ent_token_ids_in_trie:
        trie[input_ids] = True
    return trie


def get_next_token_dict(args: TrainerArguments, ent_token_ids_in_trie, prefix_trie, extra_id_0_token_id):
    neg_candidate_mask = []
    next_token_dict = {(): [extra_id_0_token_id] * args.data.num_entity}
    for ent_id in tqdm(range(args.data.num_entity)):
        rows, cols = [0], [extra_id_0_token_id]
        input_ids = ent_token_ids_in_trie[ent_id]
        for pos_id in range(1, len(input_ids)):
            cur_input_ids = input_ids[:pos_id]
            if tuple(cur_input_ids) in next_token_dict:
                cur_tokens = next_token_dict[tuple(cur_input_ids)]
            else:
                seqs = prefix_trie.keys(prefix=cur_input_ids)
                cur_tokens = [seq[pos_id] for seq in seqs]
                next_token_dict[tuple(cur_input_ids)] = Counter(cur_tokens)
            cur_tokens = list(set(cur_tokens))
            rows.extend([pos_id] * len(cur_tokens))
            cols.extend(cur_tokens)
        sparse_mask = sp.coo_matrix(([1] * len(rows), (rows, cols)), shape=(len(input_ids), args.model.config.vocab_size), dtype=np.int64)  # np.long -> np.compat.long -> np.int64
        neg_candidate_mask.append(sparse_mask)
    return neg_candidate_mask, next_token_dict


def get_ground_truth(triples):
    tail_ground_truth, head_ground_truth = defaultdict(list), defaultdict(list)
    for triple in triples:
        head, tail, rel = triple
        tail_ground_truth[(head, rel)].append(tail)
        head_ground_truth[(tail, rel)].append(head)
    return tail_ground_truth, head_ground_truth


def batchify(output_dict, key, padding_value=None, return_list=False):
    tensor_out = [out[key] for out in output_dict]
    if return_list:
        return tensor_out
    if not isinstance(tensor_out[0], torch.LongTensor) and not isinstance(tensor_out[0], torch.FloatTensor):
        tensor_out = [torch.LongTensor(value) for value in tensor_out]
    if padding_value is None:
        tensor_out = torch.stack(tensor_out, dim=0)
    else:
        tensor_out = pad_sequence(tensor_out, batch_first=True, padding_value=padding_value)
    return tensor_out


def get_soft_prompt_pos(source_ids, target_ids, mode, src, tgt, vertical_bar_token_id, extra_id_0_token_id, extra_id_1_token_id):
    try:
        sep1, sep2 = [ids for ids in range(len(source_ids)) if source_ids[ids] == vertical_bar_token_id]
    except ValueError as e:
        print(f"e={e}")
        print(f"src={src}")
        print(f"tgt={tgt}")
        print(f"source_ids={source_ids}")
        print(f"target_ids={target_ids}")
        exit(1)
    if mode == 'tail':
        input_index = [0] + list(range(0, sep1)) + [0] + [sep1] + [0] + list(range(sep1 + 1, sep2)) + [0] + list(range(sep2, len(source_ids)))
        soft_prompt_index = torch.LongTensor([0, sep1 + 1, sep1 + 3, sep2 + 3])
    elif mode == 'head':
        input_index = list(range(0, sep1 + 1)) + [0] + list(range(sep1 + 1, sep2)) + [0, sep2, 0] + list(range(sep2 + 1, len(source_ids) - 1)) + [0] + [len(source_ids) - 1]
        soft_prompt_index = torch.LongTensor([sep2 + 3, len(source_ids) + 2, sep1 + 1, sep2 + 1])

    if target_ids is None:
        target_soft_prompt_index = None
    else:
        extra_token_01, extra_token_02 = target_ids.index(extra_id_0_token_id), target_ids.index(extra_id_1_token_id)
        target_soft_prompt_index = torch.LongTensor([extra_token_01, extra_token_02])
    return input_index, soft_prompt_index, target_soft_prompt_index


class TrainDataset(Dataset):
    def __init__(self, args: TrainerArguments, tokenizer, train_triples, name_list_dict, prefix_trie_dict, ground_truth_dict):
        self.args = args
        self.train_triples = train_triples
        self.tokenizer = tokenizer
        self.vertical_bar_tok_id = tokenizer('|').input_ids[0]
        self.extra_id_0_tok_id = tokenizer('<extra_id_0>').input_ids[0]
        self.extra_id_1_tok_id = tokenizer('<extra_id_1>').input_ids[0]
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.src_description_list = name_list_dict['src_description_list']
        self.tgt_description_list = name_list_dict['tgt_description_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.neg_candidate_mask = prefix_trie_dict['neg_candidate_mask']

    def __len__(self):
        return len(self.train_triples) * 2

    def __getitem__(self, index):
        train_triple = self.train_triples[index // 2]
        mode = 'tail' if index % 2 == 0 else 'head'
        head, tail, rel = train_triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]
        if self.args.model.tgt_descrip_max_length > 0:
            head_descrip, tail_descrip = '[' + self.src_description_list[head] + ']', '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''
        if self.args.model.tgt_descrip_max_length > 0:
            head_target_descrip, tail_target_descrip = '[' + self.tgt_description_list[head] + ']', '[' + self.tgt_description_list[tail] + ']'
        else:
            head_target_descrip, tail_target_descrip = '', ''

        if mode == 'tail':
            src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>'
            tgt = '<extra_id_0>' + tail_name + tail_target_descrip + '<extra_id_1>'
        else:
            src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip
            tgt = '<extra_id_0>' + head_name + head_target_descrip + '<extra_id_1>'

        tokenized_src = self.tokenizer(src, max_length=self.args.model.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(tgt, max_length=self.args.model.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask

        ent_rel = torch.LongTensor([head, rel]) if mode == 'tail' else torch.LongTensor([tail, rel])
        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'train_triple': train_triple,
            'ent_rel': ent_rel,
        }

        input_index, soft_prompt_index, target_soft_prompt_index = get_soft_prompt_pos(source_ids, target_ids, mode,
                                                                                       src=src, tgt=tgt,
                                                                                       vertical_bar_token_id=self.vertical_bar_tok_id,
                                                                                       extra_id_0_token_id=self.extra_id_0_tok_id,
                                                                                       extra_id_1_token_id=self.extra_id_1_tok_id)
        out['input_index'] = input_index
        out['soft_prompt_index'] = soft_prompt_index
        out['target_soft_prompt_index'] = target_soft_prompt_index
        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['train_triple'] = batchify(data, 'train_triple', return_list=True)
        agg_data['ent_rel'] = batchify(data, 'ent_rel')
        agg_data['input_index'] = batchify(data, 'input_index', padding_value=0)
        agg_data['soft_prompt_index'] = batchify(data, 'soft_prompt_index')
        agg_data['target_soft_prompt_index'] = batchify(data, 'target_soft_prompt_index')
        return agg_data


class TestDataset(Dataset):
    def __init__(self, args: TrainerArguments, tokenizer, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict, mode):  # mode: {tail, head}
        self.args = args
        self.test_triples = test_triples
        self.tokenizer = tokenizer
        self.vertical_bar_tok_id = tokenizer('|').input_ids[0]
        self.extra_id_0_tok_id = tokenizer('<extra_id_0>').input_ids[0]
        self.extra_id_1_tok_id = tokenizer('<extra_id_1>').input_ids[0]
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.src_description_list = name_list_dict['src_description_list']
        self.tgt_description_list = name_list_dict['tgt_description_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.mode = mode

    def __len__(self):
        return len(self.test_triples)

    def __getitem__(self, index):
        test_triple = self.test_triples[index]
        head, tail, rel = test_triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]
        if self.args.model.tgt_descrip_max_length > 0:
            head_descrip, tail_descrip = '[' + self.src_description_list[head] + ']', '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''

        if self.mode == 'tail':
            src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>'
            tgt_ids = tail
        else:
            src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip
            tgt_ids = head

        tokenized_src = self.tokenizer(src, max_length=self.args.model.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        source_names = src
        target_names = self.ent_name_list[tgt_ids]

        # ent_rel = test_triple[[0, 2]] if self.mode == 'tail' else test_triple[[1, 2]]
        ent_rel = torch.LongTensor([head, rel]) if self.mode == 'tail' else torch.LongTensor([tail, rel])
        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'source_names': source_names,
            'target_names': target_names,
            'test_triple': test_triple,
            'ent_rel': ent_rel
        }
        input_index, soft_prompt_index, _ = get_soft_prompt_pos(source_ids, None, self.mode,
                                                                src=src, tgt=None,
                                                                vertical_bar_token_id=self.vertical_bar_tok_id,
                                                                extra_id_0_token_id=self.extra_id_0_tok_id,
                                                                extra_id_1_token_id=self.extra_id_1_tok_id)
        out['input_index'] = input_index
        out['soft_prompt_index'] = soft_prompt_index
        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['source_names'] = [dt['source_names'] for dt in data]
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        agg_data['test_triple'] = batchify(data, 'test_triple', return_list=True)
        agg_data['ent_rel'] = batchify(data, 'ent_rel')
        agg_data['input_index'] = batchify(data, 'input_index', padding_value=0)
        agg_data['soft_prompt_index'] = batchify(data, 'soft_prompt_index')
        return agg_data


class DataModule(pl.LightningDataModule):
    def __init__(self, args: TrainerArguments, train_triples, valid_triples, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict):
        super().__init__()
        self.args = args
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        # ent_name_list, rel_name_list .type: list
        self.name_list_dict = name_list_dict
        self.prefix_trie_dict = prefix_trie_dict
        self.ground_truth_dict = ground_truth_dict

        tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained)
        if '<extra_id_0>' not in tokenizer.additional_special_tokens:
            tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, additional_special_tokens=['<extra_id_0>', '<extra_id_1>'])
        self.tokenizer = tokenizer
        self.train_both = None
        self.valid_tail, self.valid_head = None, None
        self.test_tail, self.test_head = None, None

    def prepare_data(self):
        self.train_both = TrainDataset(self.args, self.tokenizer, self.train_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict)
        self.valid_tail = TestDataset(self.args, self.tokenizer, self.valid_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'tail')
        self.valid_head = TestDataset(self.args, self.tokenizer, self.valid_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'head')
        self.test_tail = TestDataset(self.args, self.tokenizer, self.test_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'tail')
        self.test_head = TestDataset(self.args, self.tokenizer, self.test_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'head')

    def train_dataloader(self):
        train_loader = DataLoader(self.train_both,
                                  batch_size=self.args.hardware.train_batch,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.args.hardware.cpu_workers)
        return train_loader

    def val_dataloader(self):
        valid_tail_loader = DataLoader(self.valid_tail,
                                       batch_size=self.args.hardware.infer_batch,
                                       shuffle=False,
                                       collate_fn=self.valid_tail.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.args.hardware.cpu_workers)
        valid_head_loader = DataLoader(self.valid_head,
                                       batch_size=self.args.hardware.infer_batch,
                                       shuffle=False,
                                       collate_fn=self.valid_head.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.args.hardware.cpu_workers)
        return [valid_tail_loader, valid_head_loader]

    def test_dataloader(self):
        test_tail_loader = DataLoader(self.test_tail,
                                      batch_size=self.args.hardware.infer_batch,
                                      shuffle=False,
                                      collate_fn=self.test_tail.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.args.hardware.cpu_workers)
        test_head_loader = DataLoader(self.test_head,
                                      batch_size=self.args.hardware.infer_batch,
                                      shuffle=False,
                                      collate_fn=self.test_head.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.args.hardware.cpu_workers)
        return [test_tail_loader, test_head_loader]


class PrintingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.time()
        print(flush=True)
        print(flush=True)
        print('=' * 70, flush=True)
        print('Epoch: %4d, ' % trainer.current_epoch, flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if len(pl_module.history['loss']) == 0:
            return
        loss = pl_module.history['loss']
        avg_loss = sum(loss) / len(loss)
        pl_module.history['loss'] = []
        print(flush=True)
        print('Total time: %4ds, loss: %2.4f' % (int(time.time() - self.start), avg_loss), flush=True)

    def on_validation_start(self, trainer, pl_module):
        if hasattr(self, 'start'):
            print(flush=True)
            print('Training time: %4ds' % (int(time.time() - self.start)), flush=True)
        self.val_start = time.time()

    def on_validation_end(self, trainer, pl_module):
        print(flush=True)
        print(pl_module.history['perf'], flush=True)
        print('Validation time: %4ds' % (int(time.time() - self.val_start)), flush=True)

    def on_test_end(self, trainer, pl_module):
        print(flush=True)
        print('=' * 70, flush=True)
        print('Epoch: test', flush=True)
        print(pl_module.history['perf'], flush=True)
        print('=' * 70, flush=True)


class T5Finetuner(pl.LightningModule):
    def __init__(self, args: TrainerArguments, ground_truth_dict, name_list_dict, prefix_trie_dict=None):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.train_tail_ground_truth = ground_truth_dict['train_tail_ground_truth']
        self.train_head_ground_truth = ground_truth_dict['train_head_ground_truth']
        self.all_tail_ground_truth = ground_truth_dict['all_tail_ground_truth']
        self.all_head_ground_truth = ground_truth_dict['all_head_ground_truth']

        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']

        self.prefix_trie = prefix_trie_dict['prefix_trie']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.next_token_dict = prefix_trie_dict['next_token_dict']
        if self.args.model.tgt_descrip_max_length > 0:
            self.ent_token_ids_in_trie_with_descrip = prefix_trie_dict['ent_token_ids_in_trie_with_descrip']
        config = AutoConfig.from_pretrained(args.model.pretrained)
        if config.model_type == "t5":
            self.core_t5_model = ModifiedT5ForConditionalGeneration.from_pretrained(args.model.pretrained)
        elif config.model_type == "mt5":
            self.core_t5_model = ModifiedMT5ForConditionalGeneration.from_pretrained(args.model.pretrained)
        else:
            raise ValueError('Invalid model type: [%s] %s' % (config.model_type, args.model.pretrained))

        prompt_dim = self.core_t5_model.model_dim
        self.rel_embed1 = nn.Embedding(self.args.data.num_relation, prompt_dim)
        self.rel_embed2 = nn.Embedding(self.args.data.num_relation, prompt_dim)
        self.rel_embed3 = nn.Embedding(self.args.data.num_relation, prompt_dim)
        self.rel_embed4 = nn.Embedding(self.args.data.num_relation, prompt_dim)

        self.history = {'perf': ..., 'loss': []}

    def training_step(self, batched_data, batch_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # target_ids, target_mask, labels: .shape: (batch_size, padded_seq_len)
        target_ids = batched_data['target_ids']
        target_mask = batched_data['target_mask']
        labels = target_ids.clone()
        labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100
        # train_triples .shape: (batch_size, 3)
        train_triples = batched_data['train_triple']
        # ent_rel .shape: (batch_size, 2)
        ent_rel = batched_data['ent_rel']

        # input_index .shape: (batch_size, seq_len + 4)
        input_index = batched_data['input_index']
        # soft_prompt_index .shape: (batch_size, 4)
        soft_prompt_index = batched_data['soft_prompt_index']
        inputs_emb, input_mask = self.get_soft_prompt_input_embed(src_ids, src_mask, ent_rel, input_index, soft_prompt_index)

        if self.args.model.seq_dropout > 0.:
            batch_size, length = input_mask.shape
            rand = torch.rand_like(input_mask.float())
            dropout = torch.logical_not(rand < self.args.model.seq_dropout).long().type_as(input_mask)
            indicator_in_batch = torch.arange(batch_size).type_as(input_mask).unsqueeze(-1)

            trailing_mask = input_index[:, 1:] == 0
            input_index = (input_index + indicator_in_batch * src_ids.shape[1]).view(-1)
            input_src_ids = torch.index_select(src_ids.view(-1), 0, input_index).view(batch_size, length)
            input_src_ids[:, 1:][trailing_mask] = 0
            dropout[input_src_ids == 1820] = 1
            dropout[input_src_ids == 32099] = 1

            soft_prompt_index = (soft_prompt_index + indicator_in_batch * length).view(-1)
            dropout = dropout.view(-1)
            dropout[soft_prompt_index] = 1
            dropout = dropout.view(batch_size, length)
            input_mask = input_mask * dropout

        output = self.core_t5_model(inputs_embeds=inputs_emb, attention_mask=input_mask, labels=labels, output_hidden_states=True)
        loss = torch.mean(output.loss)

        self.history['loss'].append(loss.detach().item())
        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx, dataset_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len) .type: torch.tensor
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # src_names, target_names: .shape: (batch_size, ) .type:list(str)
        src_names = batched_data['source_names']
        target_names = batched_data['target_names']
        # test_triple: .shape: (batch_size, 3)
        self.test_triple = batched_data['test_triple']
        # ent_rel: .shape: (batch_size, 2)
        self.ent_rel = batched_data['ent_rel']

        self.dataset_idx = dataset_idx
        if dataset_idx == 0:
            self.all_ground_truth = self.all_tail_ground_truth
            self.train_ground_truth = self.train_tail_ground_truth
        else:
            self.all_ground_truth = self.all_head_ground_truth
            self.train_ground_truth = self.train_head_ground_truth

        # generated_text .type: list(str) .len: batch_size * num_beams
        generated_text = self.decode(src_ids, src_mask, batched_data)
        group_text = [generated_text[i:i + self.args.model.num_beams] for i in range(0, len(generated_text), self.args.model.num_beams)]
        if self.args.learning.log_text:
            self.log_generation(group_text, src_names, target_names, batch_idx, dataset_idx)

        ranks = []
        for i, texts in enumerate(group_text):
            hr_key = (self.test_triple[i][dataset_idx], self.test_triple[i][2])
            all_gt_ids = self.all_ground_truth[hr_key]
            all_gt_seqs = [self.ent_name_list[ids] for ids in all_gt_ids]

            ## get rank
            if target_names[i] in texts:
                top_entities = set()
                rank = 1
                for text in texts:
                    if text == target_names[i]:
                        ranks.append(rank)
                        break
                    if text in set(self.ent_name_list) and (text not in all_gt_seqs) and (text not in top_entities):
                        top_entities.add(text)
                        rank += 1
            else:
                ranks.append(random.randint(self.args.model.num_beams + 1, self.args.data.num_entity))

        out = {'ranks': ranks}
        return out

    def decode(self, src_ids, src_mask, batched_data):
        def _extract(generated_text):
            if self.args.model.tgt_descrip_max_length > 0:
                compiler = re.compile(r'<extra_id_0>(.*?)\[')
            else:
                compiler = re.compile(r'<extra_id_0>(.*)<extra_id_1>')
            extracted_text = []
            for text in generated_text:
                match = compiler.search(text)
                if match is None:
                    # text = text.strip().lstrip('<pad> <extra_id_0>')
                    extracted_text.append(text.strip())
                else:
                    extracted_text.append(match.group(1).strip())
            return extracted_text

        def _next_candidate(args: TrainerArguments, batch_idx, input_ids):
            hr_key = (self.test_triple[batch_idx][self.dataset_idx], self.test_triple[batch_idx][2])
            all_gt_ids = self.all_ground_truth[hr_key]
            ent_token_ids_in_trie = self.ent_token_ids_in_trie_with_descrip if args.model.tgt_descrip_max_length > 0 else self.ent_token_ids_in_trie
            all_gt_seq = [tuple(ent_token_ids_in_trie[ids]) for ids in all_gt_ids]

            pred_pos = 1 if self.dataset_idx == 0 else 0
            pred_ids = tuple(ent_token_ids_in_trie[self.test_triple[batch_idx][pred_pos]])

            input_ids = input_ids.tolist()
            if input_ids[0] == 0:
                input_ids = input_ids[1:]

            if tuple(input_ids) in self.next_token_dict:
                if len(input_ids) == 0:
                    return [32099]
                if input_ids[-1] == 32098:
                    return [1]
                next_tokens = self.next_token_dict[tuple(input_ids)]
                all_gt_seq = [seq for seq in all_gt_seq if tuple(seq[: len(input_ids)]) == tuple(input_ids)]
                gt_next_tokens = Counter([seq[len(input_ids)] for seq in all_gt_seq if len(input_ids) < len(seq)])
                if tuple(pred_ids[: len(input_ids)]) == tuple(input_ids) and len(input_ids) < len(pred_ids):
                    pred_id = Counter([pred_ids[len(input_ids)]])
                else:
                    pred_id = Counter([])
                next_tokens = list(set(next_tokens - gt_next_tokens + pred_id))
                return next_tokens
            else:
                return []

        if self.args.model.decoder in ['beam_search', 'diverse_beam_search']:
            num_beam_groups = self.args.model.num_beam_groups if self.args.model.decoder == 'diverse_beam_search' else 1
            diversity_penalty = self.args.model.diversity_penalty if self.args.model.decoder == 'diverse_beam_search' else 0.
            prefix_allowed_tokens_fn = lambda batch_idx, input_ids: _next_candidate(self.args, batch_idx, input_ids) if self.args.model.use_prefix_search else None
            # input_index .shape: (batch_size, seq_len + 4)
            input_index = batched_data['input_index']
            # soft_prompt_index .shape: (batch_size, 4)
            soft_prompt_index = batched_data['soft_prompt_index']
            inputs_emb, input_mask = self.get_soft_prompt_input_embed(src_ids, src_mask, self.ent_rel, input_index,
                                                                      soft_prompt_index)
            outputs = self.core_t5_model.generate(inputs_embeds=inputs_emb,
                                                  attention_mask=input_mask,
                                                  return_dict_in_generate=True,
                                                  num_return_sequences=self.args.model.num_beams,
                                                  max_length=self.args.model.eval_tgt_max_length,
                                                  diversity_penalty=diversity_penalty,
                                                  num_beam_groups=num_beam_groups,
                                                  prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                                  num_beams=self.args.model.num_beams,
                                                  bos_token_id=0, )
            raw_generated_text = self.trainer.datamodule.tokenizer.batch_decode(outputs.sequences)
            generated_text = _extract(raw_generated_text)
            assert len(generated_text) == self.args.model.num_beams * len(src_ids)
            return generated_text
        elif self.args.model.decoder == 'do_sample':
            # input_index .shape: (batch_size, seq_len + 4)
            input_index = batched_data['input_index']
            # soft_prompt_index .shape: (batch_size, 4)
            soft_prompt_index = batched_data['soft_prompt_index']
            inputs_emb, input_mask = self.get_soft_prompt_input_embed(src_ids, src_mask, self.ent_rel, input_index,
                                                                      soft_prompt_index)
            outputs = self.core_t5_model.generate(inputs_embeds=inputs_emb,
                                                  attention_mask=input_mask,
                                                  return_dict_in_generate=True,
                                                  num_return_sequences=self.args.model.num_beams,
                                                  max_length=self.args.model.eval_tgt_max_length,
                                                  output_scores=True,
                                                  do_sample=True)

            raw_generated_text = self.trainer.datamodule.tokenizer.batch_decode(outputs.sequences)
            generated_text = _extract(raw_generated_text)
            assert len(generated_text) == self.args.model.num_beams * len(src_ids)
            # sequences .shape: (batch_size * num_beams, max_seq_len - 1)
            sequences = outputs.sequences[:, 1:]
            # scores: .shape: (batch_size * num_beams, max_seq_len - 1, vocab_size)
            scores = torch.stack(outputs.scores).transpose(0, 1)
            # scores: .shape: (batch_size * num_beams, max_seq_len - 1, 1)
            scores = torch.stack([torch.gather(scores[i], 1, sequences[i].unsqueeze(1)) for i in range(len(sequences))])
            # scores .type: list(float) .len: batch_size * num_beams
            scores = torch.mean(scores.squeeze(-1), dim=-1).tolist()

            new_generated_text = []
            for i in range(0, len(generated_text), self.args.model.num_beams):
                gen_seqs = generated_text[i:i + self.args.model.num_beams]
                gen_scores = scores[i:i + self.args.model.num_beams]
                gen_seqs = [seq for _, seq in sorted(list(zip(gen_scores, gen_seqs)), key=lambda x: x[0], reverse=True)]
                new_generated_text.extend(gen_seqs)
            return new_generated_text
        else:
            raise ValueError('Invalid decoder')

    def get_soft_prompt_input_embed(self, src_ids, src_mask, ent_rel, input_index, soft_prompt_index):
        # ent_ids, rel_ids .shape: (batch_size, 1)
        ent_ids, rel_ids = ent_rel[:, [0]], ent_rel[:, [1]]
        # ent_emb1, ent_emb2, rel_emb1, rel_emb2 .shape: (batch_size, 1, model_dim)
        ent_emb1, ent_emb2 = self.rel_embed3(rel_ids), self.rel_embed4(rel_ids)
        rel_emb1, rel_emb2 = self.rel_embed1(rel_ids), self.rel_embed2(rel_ids)
        # ent_emb, rel_emb .shape: (batch_size, 2, model_dim)
        ent_emb, rel_emb = torch.cat([ent_emb1, ent_emb2], dim=1), torch.cat([rel_emb1, rel_emb2], dim=1)
        # soft_prompt_emb .shape: (batch_size, 4, model_dim)
        soft_prompt_emb = torch.cat([ent_emb, rel_emb], dim=1)
        # inputs_emb .shape: (batch_size, seq_len, model_dim)
        inputs_emb = self.core_t5_model.encoder.embed_tokens(src_ids)
        batch_size, seq_len, model_dim = inputs_emb.shape
        # indicator_in_batch .shape: (batch_size, 1) .examples: torch.LongTensor([[0], [1], [2], [3]])
        indicator_in_batch = torch.arange(batch_size).type_as(ent_ids).unsqueeze(-1)

        # inputs_emb .shape: (batch_size * seq_len, model_dim)
        inputs_emb = inputs_emb.view(-1, model_dim)
        input_index = (input_index + indicator_in_batch * seq_len).view(-1)
        # inputs_emb .shape: (batch_size * (seq_len + 4), model_dim)
        inputs_emb = torch.index_select(inputs_emb, 0, input_index)
        soft_prompt_index = (soft_prompt_index + indicator_in_batch * (seq_len + 4)).view(-1)
        inputs_emb[soft_prompt_index] = soft_prompt_emb.view(batch_size * 4, model_dim)
        inputs_emb = inputs_emb.view(batch_size, -1, model_dim)

        input_mask = torch.cat([torch.ones(batch_size, 4).type_as(src_mask), src_mask], dim=1)
        return inputs_emb, input_mask

    def log_generation(self, group_text, src_names, target_names, batch_idx, dataset_idx):
        log_file = os.path.join(self.args.env.output_home, 'Epoch-' + str(self.current_epoch) + '-generation.tmp')
        with open(log_file, 'a', encoding='utf-8') as file:
            for i, texts in enumerate(group_text):
                file.write(str(batch_idx * self.args.hardware.infer_batch + i) + ' -- ' + src_names[i] + ' => ' + target_names[i] + '\n')
                hr_key = (self.test_triple[i][dataset_idx], self.test_triple[i][2])
                all_gt_ids = self.all_ground_truth[hr_key]
                all_gt_seqs = [self.ent_name_list[ids] for ids in all_gt_ids]
                ii = 1
                for text_i, text in enumerate(texts):
                    if text == target_names[i]:
                        file.write('\t\t%2d %10s %s\n' % (ii, '(target):', text))
                        ii += 1
                    elif text in all_gt_seqs:
                        file.write('\t\t%2s %10s %s\n' % ('', '', text))
                    elif text in self.ent_name_list:
                        file.write('\t\t%2d %10s %s\n' % (ii, '(ent):', text))
                        ii += 1
                    else:
                        file.write('\t\t%2d %10s %s\n' % (ii, '(non-ent):', text))
                        ii += 1

    def validation_epoch_end(self, outs):
        pred_tail_out, pred_head_out = outs
        agg_tail_out, agg_head_out, agg_total_out = dict(), dict(), dict()
        for out in pred_tail_out:
            for key, value in out.items():
                if key in agg_tail_out:
                    agg_tail_out[key] += value
                else:
                    agg_tail_out[key] = value

        for out in pred_head_out:
            for key, value in out.items():
                if key in agg_head_out:
                    agg_head_out[key] += value
                else:
                    agg_head_out[key] = value

        tail_ranks, head_ranks = agg_tail_out['ranks'], agg_head_out['ranks']
        del agg_tail_out['ranks']
        del agg_head_out['ranks']

        perf = get_performance(self, tail_ranks, head_ranks)
        print(flush=True)
        print(flush=True)
        print(flush=True)
        print(perf, flush=True)

    def test_step(self, batched_data, batch_idx, dataset_idx):
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        if self.args.learning.optimizer_cls == 'Adafactor':
            optim = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.args.learning.learning_rate)
        elif self.args.learning.optimizer_cls == 'AdamW':
            optim = torch.optim.AdamW(self.parameters(), lr=self.args.learning.learning_rate)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=self.args.learning.learning_rate)
        return optim


def get_performance(model: T5Finetuner, tail_ranks, head_ranks):
    tail_out = _get_performance(tail_ranks, model.args.data.name)
    head_out = _get_performance(head_ranks, model.args.data.name)
    mrr = np.array([tail_out['mrr'], head_out['mrr']])
    hit1 = np.array([tail_out['hit1'], head_out['hit1']])
    hit3 = np.array([tail_out['hit3'], head_out['hit3']])
    hit10 = np.array([tail_out['hit10'], head_out['hit10']])

    val_mrr = mrr.mean().item()
    model.log('val_mrr', val_mrr)
    perf = {'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    perf = pd.DataFrame(perf, index=['tail', 'head'])
    perf.loc['mean'] = perf.mean(axis=0)
    for hit in ['hit@1', 'hit@3', 'hit@5', 'hit@10']:
        if hit in list(perf.columns):
            perf[hit] = perf[hit].apply(lambda x: '%.2f%%' % (x * 100))
    return perf


def _get_performance(ranks, dataset):
    ranks = np.array(ranks, dtype=np.float)
    out = dict()
    out['mrr'] = (1. / ranks).mean(axis=0)
    out['hit1'] = np.sum(ranks == 1, axis=0) / len(ranks)
    out['hit3'] = np.sum(ranks <= 3, axis=0) / len(ranks)
    out['hit10'] = np.sum(ranks <= 10, axis=0) / len(ranks)
    if dataset == 'NELL':
        out['hit5'] = np.sum(ranks <= 5, axis=0) / len(ranks)
    return out


@main.command()
def train(
        verbose: int = typer.Option(default=2),
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        # data
        data_name: str = typer.Option(default="FB15k-237N-ko-min"),
        train_file: str = typer.Option(default="train2id.txt"),
        valid_file: str = typer.Option(default="valid2id.txt"),
        test_file: str = typer.Option(default="test2id.txt"),
        num_check: int = typer.Option(default=2),
        # model
        pretrained: str = typer.Option(default="google/mt5-base"),
        model_name: str = typer.Option(default="{ep:3.1f}, {val_loss:06.4f}, {val_acc:06.4f}"),
        src_max_length: int = typer.Option(default=512),
        train_tgt_max_length: int = typer.Option(default=512),
        eval_tgt_max_length: int = typer.Option(default=90),
        src_descrip_max_length: int = typer.Option(default=200),
        tgt_descrip_max_length: int = typer.Option(default=200),
        seq_dropout: float = typer.Option(default=0.1),
        decoder: str = typer.Option(default="beam_search", help="[beam_search, diverse_beam_search, do_sample]"),
        num_beams: int = typer.Option(default=40),
        num_beam_groups: int = typer.Option(default=1),
        diversity_penalty: float = typer.Option(default=0.0),
        use_prefix_search: bool = typer.Option(default=False),
        # hardware
        train_batch: int = typer.Option(default=8),
        infer_batch: int = typer.Option(default=8),
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="16-mixed"),
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        # learning
        optimizer_cls: str = typer.Option(default="Adam"),
        learning_rate: float = typer.Option(default=0.001),
        saving_policy: str = typer.Option(default="max val_acc"),
        num_saving: int = typer.Option(default=3),
        num_epochs: int = typer.Option(default=3),
        log_text: bool = typer.Option(default=False),
        check_rate_on_training: float = typer.Option(default=1 / 10),
        print_rate_on_training: float = typer.Option(default=1 / 30),
        print_rate_on_validate: float = typer.Option(default=1 / 3),
        print_rate_on_evaluate: float = typer.Option(default=1 / 3),
        print_step_on_training: int = typer.Option(default=-1),
        print_step_on_validate: int = typer.Option(default=-1),
        print_step_on_evaluate: int = typer.Option(default=-1),
        tag_format_on_training: str = typer.Option(default="st={step:d}, ep={epoch:.1f}, loss={loss:06.4f}, acc={acc:06.4f}"),
        tag_format_on_validate: str = typer.Option(default="st={step:d}, ep={epoch:.1f}, val_loss={val_loss:06.4f}, val_acc={val_acc:06.4f}"),
        tag_format_on_evaluate: str = typer.Option(default="st={step:d}, ep={epoch:.1f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}"),
):
    args = TrainerArguments.from_args(
        # env
        project=project,
        job_name=job_name,
        debugging=debugging,
        # data
        data_name=data_name,
        train_file=train_file,
        valid_file=valid_file,
        test_file=test_file,
        num_check=num_check,
        # model
        pretrained=pretrained,
        model_name=model_name,
        src_max_length=src_max_length,
        train_tgt_max_length=train_tgt_max_length,
        eval_tgt_max_length=eval_tgt_max_length,
        src_descrip_max_length=src_descrip_max_length,
        tgt_descrip_max_length=tgt_descrip_max_length,
        seq_dropout=seq_dropout,
        decoder=decoder,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        use_prefix_search=use_prefix_search,
        # hardware
        train_batch=train_batch,
        infer_batch=infer_batch,
        accelerator=accelerator,
        precision=precision,
        strategy=strategy,
        device=device,
        # learning
        optimizer_cls=optimizer_cls,
        learning_rate=learning_rate,
        saving_policy=saving_policy,
        num_saving=num_saving,
        num_epochs=num_epochs,
        log_text=log_text,
        check_rate_on_training=check_rate_on_training,
        print_rate_on_training=print_rate_on_training,
        print_rate_on_validate=print_rate_on_validate,
        print_rate_on_evaluate=print_rate_on_evaluate,
        print_step_on_training=print_step_on_training,
        print_step_on_validate=print_step_on_validate,
        print_step_on_evaluate=print_step_on_evaluate,
        tag_format_on_training=tag_format_on_training,
        tag_format_on_validate=tag_format_on_validate,
        tag_format_on_evaluate=tag_format_on_evaluate,
    )
    args.data.num_entity = get_num(args.data.home, args.data.name, 'entity')
    args.data.num_relation = get_num(args.data.home, args.data.name, 'relation')
    args.model.config = AutoConfig.from_pretrained(args.model.pretrained)
    print(args.data)
    print(args.model.config)

    ## read triples
    train_triples = read(args.data.home, args.data.name, args.data.files.train)
    valid_triples = read(args.data.home, args.data.name, args.data.files.valid)
    test_triples = read(args.data.home, args.data.name, args.data.files.test)
    all_triples = train_triples + valid_triples + test_triples
    print(len(train_triples), train_triples)
    print(len(valid_triples), valid_triples)
    print(len(test_triples), test_triples)
    print(len(all_triples), all_triples)

    ## construct name list
    original_ent_name_list, rel_name_list = read_name(args.data.home, args.data.name)
    print(len(original_ent_name_list), original_ent_name_list)
    print(len(rel_name_list), rel_name_list)
    description_list = read_file(args.data.home, args.data.name, 'entityid2description.txt', 'descrip')
    print(len(description_list), description_list)
    tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained)
    if '<extra_id_0>' not in tokenizer.additional_special_tokens:
        tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, additional_special_tokens=['<extra_id_0>', '<extra_id_1>'])
    extra_id_0_tok_id = tokenizer('<extra_id_0>').input_ids[0]
    print()
    print("----------------------------------------------------------------------------------")
    print(f" - pretrained_model.model_type: {args.model.config.model_type}")
    print(f" - tokenizer.additional_special_tokens: {tokenizer.additional_special_tokens}")
    print(f" - extra_id_0_tok_id: {extra_id_0_tok_id}")
    print("----------------------------------------------------------------------------------")
    print()
    print('tokenizing entities...')
    src_description_list = tokenizer.batch_decode([descrip[:-1] for descrip in tokenizer(description_list, max_length=args.model.src_descrip_max_length, truncation=True).input_ids])
    tgt_description_list = tokenizer.batch_decode([descrip[:-1] for descrip in tokenizer(description_list, max_length=args.model.tgt_descrip_max_length, truncation=True).input_ids])
    print(len(src_description_list), src_description_list)
    print(len(tgt_description_list), tgt_description_list)

    ## construct prefix trie
    # ent_token_ids_in_trie .type: list(list(ids))
    ent_token_ids_in_trie = tokenizer(['<extra_id_0>' + ent_name + '<extra_id_1>' for ent_name in original_ent_name_list], max_length=args.model.train_tgt_max_length, truncation=True).input_ids
    print(len(ent_token_ids_in_trie), ent_token_ids_in_trie)

    print(f"args.model.tgt_descrip_max_length={args.model.tgt_descrip_max_length}")
    if args.model.tgt_descrip_max_length > 0:
        ent_token_ids_in_trie_with_descrip = tokenizer(['<extra_id_0>' + ent_name + '[' + tgt_description_list[i] + ']' + '<extra_id_1>' for i, ent_name in enumerate(original_ent_name_list)], max_length=args.model.train_tgt_max_length, truncation=True).input_ids
        print(len(ent_token_ids_in_trie_with_descrip), ent_token_ids_in_trie_with_descrip)
        prefix_trie = construct_prefix_trie(ent_token_ids_in_trie_with_descrip)
        print(f"prefix_trie1={prefix_trie}")
        neg_candidate_mask, next_token_dict = get_next_token_dict(args, ent_token_ids_in_trie_with_descrip, prefix_trie, extra_id_0_token_id=extra_id_0_tok_id)
        # print(f"neg_candidate_mask1={neg_candidate_mask}")
        print(f"next_token_dict1={next_token_dict}")
    else:
        prefix_trie = construct_prefix_trie(ent_token_ids_in_trie)
        print(f"prefix_trie2={prefix_trie}")
        neg_candidate_mask, next_token_dict = get_next_token_dict(args, ent_token_ids_in_trie, prefix_trie, extra_id_0_token_id=extra_id_0_tok_id)
        # print(f"neg_candidate_mask2={neg_candidate_mask}")
        print(f"next_token_dict2={next_token_dict}")
    ent_name_list = tokenizer.batch_decode([tokens[1:-2] for tokens in ent_token_ids_in_trie])
    print(len(ent_name_list), ent_name_list)
    name_list_dict = {
        'original_ent_name_list': original_ent_name_list,
        'ent_name_list': ent_name_list,
        'rel_name_list': rel_name_list,
        'src_description_list': src_description_list,
        'tgt_description_list': tgt_description_list
    }
    print(f"name_list_dict.keys()={name_list_dict.keys()}")

    prefix_trie_dict = {
        'prefix_trie': prefix_trie,
        'ent_token_ids_in_trie': ent_token_ids_in_trie,
        'neg_candidate_mask': neg_candidate_mask,
        'next_token_dict': next_token_dict
    }
    print(f"args.model.tgt_descrip_max_length={args.model.tgt_descrip_max_length}")
    if args.model.tgt_descrip_max_length > 0:
        prefix_trie_dict['ent_token_ids_in_trie_with_descrip'] = ent_token_ids_in_trie_with_descrip
    print(f"prefix_trie_dict.keys()={prefix_trie_dict.keys()}")

    ## construct ground truth dictionary
    # ground truth .shape: dict, example: {hr_str_key1: [t_id11, t_id12, ...], (hr_str_key2: [t_id21, t_id22, ...], ...}
    train_tail_ground_truth, train_head_ground_truth = get_ground_truth(train_triples)
    all_tail_ground_truth, all_head_ground_truth = get_ground_truth(all_triples)
    print(f"train_tail_ground_truth={train_tail_ground_truth}")
    print(f"train_head_ground_truth={train_head_ground_truth}")

    ground_truth_dict = {
        'train_tail_ground_truth': train_tail_ground_truth,
        'train_head_ground_truth': train_head_ground_truth,
        'all_tail_ground_truth': all_tail_ground_truth,
        'all_head_ground_truth': all_head_ground_truth,
    }
    print(f"ground_truth_dict.keys()={ground_truth_dict.keys()}")

    datamodule = DataModule(args, train_triples, valid_triples, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict)
    print('datamodule construction done.', flush=True)

    print(args.env.output_home)
    print()
    print("----------------------------------------------------------------------------------")
    print(f" * pl.Trainer(accelerator={args.hardware.accelerator}, precision={args.hardware.precision}, num_epochs={args.learning.num_epochs})")
    print("----------------------------------------------------------------------------------")
    print()
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.env.output_home,
        filename=args.data.name + '-{epoch:03d}-{' + 'val_mrr' + ':.4f}',
        every_n_epochs=int(args.learning.check_rate_on_training),
        save_top_k=5,
        monitor='val_mrr',
        mode='max',
    )
    trainer = pl.Trainer(
        devices=1,
        precision=args.hardware.precision,
        accelerator=args.hardware.accelerator,
        max_epochs=args.learning.num_epochs,
        check_val_every_n_epoch=int(args.learning.check_rate_on_training),
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        logger=False,
        callbacks=[
            checkpoint_callback,
            PrintingCallback(),
        ],
    )
    kw_args = {
        'ground_truth_dict': ground_truth_dict,
        'name_list_dict': name_list_dict,
        'prefix_trie_dict': prefix_trie_dict
    }

    model = T5Finetuner(args, **kw_args)
    print('model construction done.', flush=True)
    trainer.fit(model, datamodule)
    model_path = checkpoint_callback.best_model_path
    print(f'model_path: [{model_path}]', flush=True)
    model = T5Finetuner.load_from_checkpoint(model_path, strict=False, args=args, **kw_args)
    trainer.test(model, dataloaders=datamodule)


if __name__ == "__main__":
    main()
