import logging
import os
from collections import Counter
from collections import defaultdict as ddict
from typing import List

import numpy as np
import pygtrie
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
import typer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from chrisbase.data import AppTyper
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
    tail_ground_truth, head_ground_truth = ddict(list), ddict(list)
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
    def __init__(self, configs, tokenizer, train_triples, name_list_dict, prefix_trie_dict, ground_truth_dict):
        self.configs = configs
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
        if self.configs.temporal:
            head, tail, rel, time = train_triple
        else:
            head, tail, rel = train_triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]
        if self.configs.src_descrip_max_length > 0:
            head_descrip, tail_descrip = '[' + self.src_description_list[head] + ']', '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''
        if self.configs.tgt_descrip_max_length > 0:
            head_target_descrip, tail_target_descrip = '[' + self.tgt_description_list[head] + ']', '[' + self.tgt_description_list[tail] + ']'
        else:
            head_target_descrip, tail_target_descrip = '', ''

        if mode == 'tail':
            if self.configs.temporal:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | ' + time
            else:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>'
            tgt = '<extra_id_0>' + tail_name + tail_target_descrip + '<extra_id_1>'
        else:
            if self.configs.temporal:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip + ' | ' + time
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip
            tgt = '<extra_id_0>' + head_name + head_target_descrip + '<extra_id_1>'

        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(tgt, max_length=self.configs.train_tgt_max_length, truncation=True)
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

        input_index, soft_prompt_index, target_soft_prompt_index = get_soft_prompt_pos(self.configs, source_ids, target_ids, mode,
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
    def __init__(self, configs, tokenizer, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict, mode):  # mode: {tail, head}
        self.configs = configs
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
        if self.configs.temporal:
            head, tail, rel, time = test_triple
        else:
            head, tail, rel = test_triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]
        if self.configs.src_descrip_max_length > 0:
            head_descrip, tail_descrip = '[' + self.src_description_list[head] + ']', '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''

        if self.mode == 'tail':
            if self.configs.temporal:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | ' + time
            else:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>'
            tgt_ids = tail
        else:
            if self.configs.temporal:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip + ' | ' + time
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip
            tgt_ids = head

        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
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
        input_index, soft_prompt_index, _ = get_soft_prompt_pos(self.configs, source_ids, None, self.mode,
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
                                  batch_size=self.args.hardware.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.args.hardware.cpu_workers)
        return train_loader

    def val_dataloader(self):
        valid_tail_loader = DataLoader(self.valid_tail,
                                       batch_size=self.args.hardware.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_tail.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.args.hardware.cpu_workers)
        valid_head_loader = DataLoader(self.valid_head,
                                       batch_size=self.args.hardware.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_head.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.args.hardware.cpu_workers)
        return [valid_tail_loader, valid_head_loader]

    def test_dataloader(self):
        test_tail_loader = DataLoader(self.test_tail,
                                      batch_size=self.args.hardware.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_tail.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.args.hardware.cpu_workers)
        test_head_loader = DataLoader(self.test_head,
                                      batch_size=self.args.hardware.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_head.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.args.hardware.cpu_workers)
        return [test_tail_loader, test_head_loader]


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
        src_max_length: int = typer.Option(default=512),
        train_tgt_max_length: int = typer.Option(default=512),
        eval_tgt_max_length: int = typer.Option(default=90),
        src_descrip_max_length: int = typer.Option(default=200),
        tgt_descrip_max_length: int = typer.Option(default=200),
        # model
        pretrained: str = typer.Option(default="google/mt5-base"),
        model_name: str = typer.Option(default="{ep:3.1f}, {val_loss:06.4f}, {val_acc:06.4f}"),
        # hardware
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="16-mixed"),
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        batch_size: int = typer.Option(default=8),
        # learning
        learning_rate: float = typer.Option(default=0.001),
        saving_policy: str = typer.Option(default="max val_acc"),
        num_saving: int = typer.Option(default=3),
        num_epochs: int = typer.Option(default=3),
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
        src_max_length=src_max_length,
        train_tgt_max_length=train_tgt_max_length,
        eval_tgt_max_length=eval_tgt_max_length,
        src_descrip_max_length=src_descrip_max_length,
        tgt_descrip_max_length=tgt_descrip_max_length,
        # model
        pretrained=pretrained,
        model_name=model_name,
        # hardware
        accelerator=accelerator,
        precision=precision,
        strategy=strategy,
        device=device,
        batch_size=batch_size,
        # learning
        learning_rate=learning_rate,
        saving_policy=saving_policy,
        num_saving=num_saving,
        num_epochs=num_epochs,
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
    src_description_list = tokenizer.batch_decode([descrip[:-1] for descrip in tokenizer(description_list, max_length=args.data.src_descrip_max_length, truncation=True).input_ids])
    tgt_description_list = tokenizer.batch_decode([descrip[:-1] for descrip in tokenizer(description_list, max_length=args.data.tgt_descrip_max_length, truncation=True).input_ids])
    print(len(src_description_list), src_description_list)
    print(len(tgt_description_list), tgt_description_list)

    ## construct prefix trie
    # ent_token_ids_in_trie .type: list(list(ids))
    ent_token_ids_in_trie = tokenizer(['<extra_id_0>' + ent_name + '<extra_id_1>' for ent_name in original_ent_name_list], max_length=args.data.train_tgt_max_length, truncation=True).input_ids
    print(len(ent_token_ids_in_trie), ent_token_ids_in_trie)

    print(f"args.data.tgt_descrip_max_length={args.data.tgt_descrip_max_length}")
    if args.data.tgt_descrip_max_length > 0:
        ent_token_ids_in_trie_with_descrip = tokenizer(['<extra_id_0>' + ent_name + '[' + tgt_description_list[i] + ']' + '<extra_id_1>' for i, ent_name in enumerate(original_ent_name_list)], max_length=args.data.train_tgt_max_length, truncation=True).input_ids
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
    print(f"args.data.tgt_descrip_max_length={args.data.tgt_descrip_max_length}")
    if args.data.tgt_descrip_max_length > 0:
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


if __name__ == "__main__":
    main()
