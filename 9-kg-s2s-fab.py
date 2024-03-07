from collections import defaultdict as ddict
import logging
import os
from collections import Counter
from typing import List

import numpy as np
import pygtrie
import scipy.sparse as sp
import typer
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


if __name__ == "__main__":
    main()
