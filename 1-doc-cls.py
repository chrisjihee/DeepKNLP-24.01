import logging
import os
from typing import List
from pytorch_lightning import Trainer

import torch
import typer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import nlpbook
from chrisbase.data import AppTyper, JobTimer, RuntimeChecking
from chrisbase.io import hr
from nlpbook import TrainerArguments
from nlpbook.cls import NsmcCorpus, ClassificationDataset, ClassificationTask

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def hello(name: str):
    print(f"Hello {name}")


@app.command()
def train(
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        verbose: int = typer.Option(default=2),
        # data
        data_name: str = typer.Option(default="nsmc"),
        train_file: str = typer.Option(default="ratings_train.txt"),
        valid_file: str = typer.Option(default="ratings_test.txt"),
        test_file: str = typer.Option(default=None),
        num_check: int = typer.Option(default=2),
        # model
        pretrained: str = typer.Option(default="pretrained/KPF-BERT"),
        model_name: str = typer.Option(default="{ep:3.1f}, {val_loss:06.4f}, {val_acc:06.4f}"),
        seq_len: int = typer.Option(default=64),
        # hardware
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="16-mixed"),
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        train_batch: int = typer.Option(default=64),
        infer_batch: int = typer.Option(default=64),
        # learning
        validate_fmt: str = typer.Option(default="loss={val_loss:06.4f}, acc={val_acc:06.4f}"),
        validate_int: float = typer.Option(default=0.1),
        num_save: int = typer.Option(default=3),
        save_by: str = typer.Option(default="max val_acc"),
        epochs: int = typer.Option(default=1),
        lr: float = typer.Option(default=5e-5),
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('high')
    logging.getLogger("fsspec.local").setLevel(logging.WARNING)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
    args = TrainerArguments.from_args(
        project=project,
        job_name=job_name,
        debugging=debugging,
        data_name=data_name,
        train_file=train_file,
        valid_file=valid_file,
        test_file=test_file,
        num_check=num_check,
        pretrained=pretrained,
        model_name=model_name,
        seq_len=seq_len,
        train_batch=train_batch,
        infer_batch=infer_batch,
        accelerator=accelerator,
        precision=precision,
        strategy=strategy,
        device=device,
        tag_format_on_validate=validate_fmt,
        check_rate_on_training=validate_int,
        num_saving=num_save,
        saving_policy=save_by,
        num_epochs=epochs,
        learning_rate=lr,
    )
    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, mb=1, rc='=',
                  verbose=verbose > 0, args=args if debugging or verbose > 1 else None):
        args.set_seed()
        corpus = NsmcCorpus(args)
        tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
        logger.info(hr('-'))

        train_dataset = ClassificationDataset("train", data=corpus, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.train_batch,
                                      collate_fn=nlpbook.data_collator,
                                      drop_last=False)
        logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
        logger.info(f"Created train_dataloader providing {len(train_dataloader)} batches")
        logger.info(hr('-'))

        valid_dataset = ClassificationDataset("valid", data=corpus, tokenizer=tokenizer)
        valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                      num_workers=args.hardware.cpu_workers,
                                      batch_size=args.hardware.infer_batch,
                                      collate_fn=nlpbook.data_collator,
                                      drop_last=False)
        logger.info(f"Created valid_dataset providing {len(valid_dataset)} examples")
        logger.info(f"Created valid_dataloader providing {len(valid_dataloader)} batches")
        logger.info(hr('-'))

        pretrained_model_config = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=corpus.num_labels
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model.pretrained,
            config=pretrained_model_config,
        )
        logger.info(hr('-'))

        with RuntimeChecking(args.configure_csv_logger()):
            trainer: Trainer = nlpbook.make_trainer(args)
            trainer.fit(ClassificationTask(args,
                                           model=model,
                                           trainer=trainer,
                                           epoch_steps=len(train_dataloader)),
                        train_dataloaders=train_dataloader,
                        val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    app()
