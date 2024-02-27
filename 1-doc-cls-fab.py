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
from chrisbase.data import AppTyper
from chrisbase.io import hr
from chrisbase.time import now
from chrisbase.util import mute_tqdm_cls
from nlpbook.arguments import TrainerArguments, TesterArguments
from nlpbook.cls import NsmcCorpus, ClassificationDataset
from nlpbook.metrics import accuracy

epsilon = 1e-7
logger = logging.getLogger(__name__)
main = AppTyper()


def fabric_barrier(fabric: Fabric, title: str, c='-'):
    fabric.barrier(title)
    fabric.print(hr(c=c, title=title))


class TCModel(LightningModule):
    def __init__(self,
                 args: TrainerArguments | TesterArguments,
                 model: PreTrainedModel,
                 corpus: NsmcCorpus,
                 ):
        super().__init__()
        self.args: TesterArguments | TrainerArguments = args
        self.model: PreTrainedModel = model
        self.corpus: NsmcCorpus = corpus
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.args.learning.rate)

    def train_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        train_dataset = ClassificationDataset("train", corpus=self.corpus, tokenizer=self.tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=self.args.hardware.cpu_workers,
                                      batch_size=self.args.hardware.batch_size,
                                      collate_fn=nlpbook.data_collator,
                                      drop_last=False)
        self.fabric.print(f"Created train_dataset providing {len(train_dataset)} examples")
        self.fabric.print(f"Created train_dataloader providing {len(train_dataloader)} batches")
        return train_dataloader

    def val_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        val_dataset = ClassificationDataset("valid", corpus=self.corpus, tokenizer=self.tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=self.args.hardware.cpu_workers,
                                    batch_size=self.args.hardware.batch_size,
                                    collate_fn=nlpbook.data_collator,
                                    drop_last=False)
        self.fabric.print(f"Created val_dataset providing {len(val_dataset)} examples")
        self.fabric.print(f"Created val_dataloader providing {len(val_dataloader)} batches")
        return val_dataloader

    def test_dataloader(self):
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug
        val_dataset = ClassificationDataset("test", corpus=self.corpus, tokenizer=self.tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=self.args.hardware.cpu_workers,
                                    batch_size=self.args.hardware.batch_size,
                                    collate_fn=nlpbook.data_collator,
                                    drop_last=False)
        self.fabric.print(f"Created test_dataset providing {len(val_dataset)} examples")
        self.fabric.print(f"Created test_dataloader providing {len(val_dataloader)} batches")
        return val_dataloader

    def training_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        labels: torch.Tensor = inputs["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        acc: torch.Tensor = accuracy(preds, labels)
        return {
            "loss": outputs.loss,
            "acc": acc,
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        labels: List[int] = inputs["labels"].tolist()
        preds: List[int] = outputs.logits.argmax(dim=-1).tolist()
        return {
            "loss": outputs.loss,
            "preds": preds,
            "labels": labels
        }

    @torch.no_grad()
    def test_step(self, inputs, batch_idx):
        return self.validation_step(inputs, batch_idx)


def train_loop(
        fabric: Fabric,
        model: TCModel,
        optimizer: OptimizerLRScheduler,
        dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
):
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    model.args.prog.global_step = 0
    model.args.prog.global_epoch = 0.0
    num_batch = len(dataloader)
    check_interval = model.args.learning.checking_epochs * num_batch - epsilon
    print_interval = model.args.learning.training_printing * num_batch - epsilon
    for epoch in range(num_epochs):
        progress = mute_tqdm_cls(bar_size=30, desc_size=8)(dataloader, desc="training", total=num_batch, unit=f"x{dataloader.batch_size}b")
        progress.update()
        for i, batch in enumerate(progress, start=1):
            model.train()
            model.args.prog.global_step += 1
            model.args.prog.global_epoch = model.args.prog.global_step / num_batch
            optimizer.zero_grad()
            outputs = model.training_step(batch, i)
            metrics = {
                "step": model.args.prog.global_step,
                "epoch": round(model.args.prog.global_epoch, 4),
                "loss": outputs["loss"].item(),
                "acc": outputs["acc"].item(),
            }
            fabric.log_dict(metrics=metrics, step=model.args.prog.global_step)
            fabric.backward(outputs["loss"])
            optimizer.step()
            if i % print_interval < 1:
                fabric.print(f"(Ep {model.args.prog.global_epoch:3.1f}) {progress}"
                             f" | {model.args.learning.training_format.format(**metrics)}")
            if model.args.prog.global_step % check_interval < 1:
                val_loop(fabric, model, val_dataloader)
                fabric_barrier(fabric, "[after-check]")
        fabric_barrier(fabric, "[after-epoch]", c='=')


@torch.no_grad()
def val_loop(
        fabric: Fabric,
        model: TCModel,
        dataloader: DataLoader,
):
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = model.args.learning.checking_printing * num_batch - epsilon
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    progress = mute_tqdm_cls(bar_size=20, desc_size=8)(dataloader, desc="checking", total=num_batch, unit=f"x{dataloader.batch_size}b")
    progress.update()
    for i, batch in enumerate(progress, start=1):
        model.eval()
        outputs = model.validation_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
        if i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:3.1f}) {progress}")
        if i >= len(progress):
            metrics = {
                "step": model.args.prog.global_step,
                "epoch": round(model.args.prog.global_epoch, 4),
                "val_loss": torch.stack(losses).mean().item(),
                "val_acc": accuracy(torch.tensor(preds), torch.tensor(labels)).item(),
            }
            fabric.log_dict(metrics=metrics, step=model.args.prog.global_step)
            fabric.print(f"(Ep {model.args.prog.global_epoch:3.1f}) {progress}"
                         f" | {model.args.learning.checking_format.format(**metrics)}")


@torch.no_grad()
def test_loop(
        fabric: Fabric,
        model: TCModel,
        dataloader: DataLoader,
):
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = model.args.learning.checking_printing * num_batch - epsilon
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    progress = mute_tqdm_cls(bar_size=20, desc_size=8)(dataloader, desc="testing", total=num_batch, unit=f"x{dataloader.batch_size}b")
    progress.update()
    for i, batch in enumerate(progress, start=1):
        model.eval()
        outputs = model.test_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
        if i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:3.1f}) {progress}")
        if i >= len(progress):
            metrics = {
                "step": model.args.prog.global_step,
                "epoch": round(model.args.prog.global_epoch, 4),
                "test_loss": torch.stack(losses).mean().item(),
                "test_acc": accuracy(torch.tensor(preds), torch.tensor(labels)).item(),
            }
            fabric.log_dict(metrics=metrics, step=model.args.prog.global_step)
            fabric.print(f"(Ep {model.args.prog.global_epoch:3.1f}) {progress}"
                         f" | {model.args.learning.testing_format.format(**metrics)}")


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
        training_printing: float = typer.Option(default=0.0333),
        checking_printing: float = typer.Option(default=0.34),
        checking_epochs: float = typer.Option(default=0.1),
        training_format: str = typer.Option(default="st={step:d}, ep={epoch:.1f}, loss={loss:06.4f}, acc={acc:06.4f}"),
        checking_format: str = typer.Option(default="st={step:d}, ep={epoch:.1f}, val_loss={val_loss:06.4f}, val_acc={val_acc:06.4f}"),
        testing_format: str = typer.Option(default="st={step:d}, ep={epoch:.1f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}"),
        num_save: int = typer.Option(default=3),
        save_by: str = typer.Option(default="max val_acc"),
        num_epochs: int = typer.Option(default=2),
        lr: float = typer.Option(default=5e-5),
):
    torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)

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
        training_printing=training_printing,
        checking_printing=checking_printing,
        checking_epochs=checking_epochs,
        training_format=training_format,
        checking_format=checking_format,
        testing_format=testing_format,
        num_epochs=num_epochs,
        num_save=num_save,
        save_by=save_by,
        rate=lr,
    )
    fabric = Fabric(
        loggers=[
            CSVLogger(root_dir=".", version=now('%m%d.%H%M%S'), flush_logs_every_n_steps=1),
            TensorBoardLogger(root_dir=".", version=now('%m%d.%H%M%S')),  # tensorboard --logdir lightning_logs
        ],
        devices=args.hardware.devices,
        strategy=args.hardware.strategy,
        precision=args.hardware.precision,
        accelerator=args.hardware.accelerator,
    )
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    fabric.seed_everything(args.learning.seed)
    fabric.launch()
    args.prog.world_size = fabric.world_size
    args.prog.node_rank = fabric.node_rank
    args.prog.local_rank = fabric.local_rank
    args.prog.global_rank = fabric.global_rank

    corpus = NsmcCorpus(args)
    fabric_barrier(fabric, "[after-corpus]", c='=')

    pretrained_model_config = AutoConfig.from_pretrained(
        args.model.pretrained,
        num_labels=corpus.num_labels
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model.pretrained,
        config=pretrained_model_config,
    )
    model = TCModel(args=args, model=model, corpus=corpus)
    optimizer = model.configure_optimizers()
    model, optimizer = fabric.setup(model, optimizer)
    fabric_barrier(fabric, "[after-model]", c='=')

    train_dataloader = model.train_dataloader()
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    fabric_barrier(fabric, "[after-train_dataloader]", c='=')

    val_dataloader = model.val_dataloader()
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    fabric_barrier(fabric, "[after-val_dataloader]", c='=')

    train_loop(fabric, model, optimizer, train_dataloader, val_dataloader, num_epochs=args.learning.num_epochs)
    fabric_barrier(fabric, "[after-train]", c='=')

    test_loop(fabric, model, val_dataloader)
    fabric_barrier(fabric, "[after-test]", c='=')


if __name__ == "__main__":
    main()
