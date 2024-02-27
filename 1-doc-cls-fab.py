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

logger = logging.getLogger(__name__)
main = AppTyper()


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
        return AdamW(self.model.parameters(), lr=self.args.learning.lr)

    def train_dataloader(self):
        train_dataset = ClassificationDataset("train", corpus=self.corpus, tokenizer=self.tokenizer)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                      num_workers=self.args.hardware.cpu_workers,
                                      batch_size=self.args.hardware.batch_size,
                                      collate_fn=nlpbook.data_collator,
                                      drop_last=False)
        logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
        logger.info(f"Created train_dataloader providing {len(train_dataloader)} batches")
        return train_dataloader

    def val_dataloader(self):
        val_dataset = ClassificationDataset("valid", corpus=self.corpus, tokenizer=self.tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=self.args.hardware.cpu_workers,
                                    batch_size=self.args.hardware.batch_size,
                                    collate_fn=nlpbook.data_collator,
                                    drop_last=False)
        logger.info(f"Created val_dataset providing {len(val_dataset)} examples")
        logger.info(f"Created val_dataloader providing {len(val_dataloader)} batches")
        return val_dataloader

    def test_dataloader(self):
        val_dataset = ClassificationDataset("test", corpus=self.corpus, tokenizer=self.tokenizer)
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                    num_workers=self.args.hardware.cpu_workers,
                                    batch_size=self.args.hardware.batch_size,
                                    collate_fn=nlpbook.data_collator,
                                    drop_last=False)
        logger.info(f"Created test_dataset providing {len(val_dataset)} examples")
        logger.info(f"Created test_dataloader providing {len(val_dataloader)} batches")
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

    def validation_step(self, inputs, batch_idx):
        outputs: SequenceClassifierOutput = self.model(**inputs)
        labels: torch.Tensor = inputs["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)
        return {
            "loss": outputs.loss,
            "preds": preds,
            "labels": labels
        }

    def test_step(self, inputs, batch_idx):
        return self.validation_step(inputs, batch_idx)


def train_loop(
        fabric: Fabric,
        model: TCModel,
        optimizer: OptimizerLRScheduler,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
):
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    model.args.prog.global_step = 0
    model.args.prog.global_epoch = 0.0
    num_train_batch = len(train_dataloader)
    log_interval = model.args.learning.training_int * num_train_batch
    val_interval = model.args.learning.validate_int * num_train_batch
    for epoch in range(num_epochs):
        epoch_info = f"(Epoch {epoch + 1:02d})"
        epoch_tqdm = mute_tqdm_cls(bar_size=40)
        progress = epoch_tqdm(train_dataloader, total=num_train_batch,
                              unit=f"x{train_dataloader.batch_size}",
                              pre=epoch_info, desc="training")
        progress.update()
        for i, batch in enumerate(progress, start=1):
            model.train()
            model.args.prog.global_step += 1
            model.args.prog.global_epoch = model.args.prog.global_step / num_train_batch
            optimizer.zero_grad()
            outputs = model.training_step(batch, i)
            metrics = {
                "epoch": round(model.args.prog.global_epoch, 4),
                "loss": outputs["loss"].item(),
                "acc": outputs["acc"].item(),
            }
            fabric.log_dict(metrics=metrics, step=model.args.prog.global_step)
            fabric.backward(outputs["loss"])
            optimizer.step()
            if i % log_interval < 1 or i >= len(progress):
                fabric.print(f"{progress} | {model.args.learning.training_fmt.format(**metrics)}")
            if model.args.prog.global_step % val_interval < 1:
                fabric.barrier()
                val_loop(fabric, model, val_dataloader, str(progress).replace('training', 'validate'))
        fabric.barrier()
        fabric.print(hr('-'))


def val_loop(
        fabric: Fabric,
        model: TCModel,
        dataloader: DataLoader,
        progress: str,
):
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.test_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
    metrics = {
        "epoch": round(model.args.prog.global_epoch, 4),
        "val_loss": torch.stack(losses).mean().item(),
        "val_acc": accuracy(torch.tensor(preds), torch.tensor(labels)).item(),
    }
    fabric.log_dict(metrics=metrics, step=model.args.prog.global_step)
    fabric.print(f"{progress} | {model.args.learning.validate_fmt.format(**metrics)}")


def test_loop(
        fabric: Fabric,
        model: TCModel,
        dataloader: DataLoader,
):
    fabric.print = logger.info
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.test_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
    metrics = {
        "epoch": round(model.args.prog.global_epoch, 4),
        "test_loss": torch.stack(losses).mean().item(),
        "test_acc": accuracy(torch.tensor(preds), torch.tensor(labels)).item(),
    }
    fabric.log_dict(metrics=metrics, step=model.args.prog.global_step)
    fabric.print(f"Test Result[{fabric.local_rank}]: {metrics}")


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
        training_fmt: str = typer.Option(default="ep={epoch:.1f}, loss={loss:06.4f}, acc={acc:06.4f}"),
        validate_fmt: str = typer.Option(default="ep={epoch:.1f}, loss={val_loss:06.4f}, acc={val_acc:06.4f}"),
        training_int: float = typer.Option(default=0.1),
        validate_int: float = typer.Option(default=0.2),
        num_save: int = typer.Option(default=3),
        save_by: str = typer.Option(default="max val_acc"),
        epochs: int = typer.Option(default=3),
        lr: float = typer.Option(default=5e-5),
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
        training_fmt=training_fmt,
        training_int=training_int,
        validate_fmt=validate_fmt,
        validate_int=validate_int,
        num_save=num_save,
        save_by=save_by,
        epochs=epochs,
        lr=lr,
    )
    fabric = Fabric(
        devices=device,
        loggers=[
            CSVLogger(root_dir=".", version=now('%m%d.%H%M%S'), flush_logs_every_n_steps=1),
            TensorBoardLogger(root_dir=".", version=now('%m%d.%H%M%S')),  # tensorboard --logdir lightning_logs
        ],
    )
    fabric.seed_everything(args.learning.seed)
    fabric.launch()

    corpus = NsmcCorpus(args)
    logger.info(hr('-'))

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
    logger.info(hr('-'))

    train_dataloader = model.train_dataloader()
    logger.info(hr('-'))
    val_dataloader = model.val_dataloader()
    logger.info(hr('-'))

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    train_loop(fabric, model, optimizer, train_dataloader, val_dataloader, num_epochs=args.learning.epochs)
    test_loop(fabric, model, val_dataloader)


if __name__ == "__main__":
    main()
