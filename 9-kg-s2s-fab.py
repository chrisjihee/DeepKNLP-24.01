import logging
import os
from typing import List

import torch
import typer
from lightning import LightningModule
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

import nlpbook
from chrisbase.data import AppTyper, JobTimer
from chrisbase.io import hr
from chrisbase.time import now
from chrisbase.util import mute_tqdm_cls
from nlpbook.arguments import TrainerArguments, TesterArguments
from nlpbook.cls import NsmcCorpus, ClassificationDataset
from nlpbook.metrics import accuracy

logger = logging.getLogger(__name__)
main = AppTyper()


@main.command()
def train(
        verbose: int = typer.Option(default=2),
        # env
        project: str = typer.Option(default="DeepKNLP"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
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
        device: List[int] = typer.Option(default=[0, 1]),
        batch_size: int = typer.Option(default=64),
        # learning
        learning_rate: float = typer.Option(default=5e-5),
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
        # model
        pretrained=pretrained,
        model_name=model_name,
        seq_len=seq_len,
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
    print(args)


if __name__ == "__main__":
    main()
