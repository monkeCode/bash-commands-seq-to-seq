import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import mlflow.transformers
from torchmetrics.text import BLEUScore
from torchmetrics.text import Perplexity
from torchmetrics import Metric
import re

MLFLOW_ADDR = "http://mlflow.k3s.home"
# MLFLOW_ADDR = "mlruns"
MODEL_NAME = "google/flan-t5-small"


class ExactMatch(Metric):
    def __init__(self, normalize_fn=None):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.normalize_fn = normalize_fn or (lambda x: x.strip().lower())

    def update(self, preds, targets):
        if isinstance(preds, list):
            preds = [self.normalize_fn(p) for p in preds]
        else:
            preds = [self.normalize_fn(preds)]

        if isinstance(targets[0], list):
            # Handle multiple reference texts
            targets = [
                [self.normalize_fn(t) for t in target_list] for target_list in targets
            ]
        else:
            # Single reference text
            targets = [[self.normalize_fn(t)] for t in targets]

        for pred, target_list in zip(preds, targets):
            if pred in target_list:
                self.correct += 1
            self.total += 1

    def compute(self):
        return (
            self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)
        )


class CommandDataset(Dataset):
    def __init__(
        self,
        csv_file,
        tokenizer,
        max_source_len=128,
        max_target_len=64,
        description_column="description",
        command_column="command",
    ):
        self.data = csv_file
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.description_column = description_column
        self.command_column = command_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        description = str(self.data.iloc[idx][self.description_column])
        command = str(self.data.iloc[idx][self.command_column])

        source = self.tokenizer(
            description,
            max_length=self.max_source_len,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer(
            command,
            max_length=self.max_target_len,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source["input_ids"].flatten(),
            "attention_mask": source["attention_mask"].flatten(),
            "labels": target["input_ids"].flatten(),
        }


def t5_collate_fn(batch, tokenizer, max_source_len=128, max_target_len=64):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    if padded_input_ids.size(1) > max_source_len:
        padded_input_ids = padded_input_ids[:, :max_source_len]
        padded_attention_mask = padded_attention_mask[:, :max_source_len]

    if padded_labels.size(1) > max_target_len:
        padded_labels = padded_labels[:, :max_target_len]

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }


class T5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_file,
        test_csv_file,
        tokenizer,
        batch_size=16,
        max_source_len=128,
        max_target_len=64,
        description_column="description",
        command_column="command",
    ):
        super().__init__()
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.description_column = description_column
        self.command_column = command_column

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate" or stage is None:
            full_dataset = CommandDataset(
                self.train_csv_file,
                self.tokenizer,
                self.max_source_len,
                self.max_target_len,
                self.description_column,
                self.command_column,
            )

            val_size = 5000
            train_size = len(full_dataset) - val_size
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator
            )
            print(f"Train size: {train_size}\nValidate size: {val_size}")

        if stage == "test" or stage is None:
            self.test_dataset = CommandDataset(
                self.test_csv_file,
                self.tokenizer,
                self.max_source_len,
                self.max_target_len,
                self.description_column,
                self.command_column,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda batch: t5_collate_fn(
                batch, self.tokenizer, self.max_source_len, self.max_target_len
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda batch: t5_collate_fn(
                batch, self.tokenizer, self.max_source_len, self.max_target_len
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda batch: t5_collate_fn(
                batch, self.tokenizer, self.max_source_len, self.max_target_len
            ),
        )


class T5Model(pl.LightningModule):
    def __init__(
        self, tokenizer, model_name="google-t5/t5-small", lr=1e-4, max_target_len=64
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).train()

        self.tokenizer = tokenizer
        self.lr = lr
        self.max_target_len = max_target_len
        self.save_hyperparameters()

        # Normalization function
        self.normalize_fn = lambda x: re.sub(
            r"\s+", " ", x.strip().lower().replace("'", "").replace('"', "")
        )

        # Validation metrics
        self.val_bleu_1 = BLEUScore(n_gram=1)
        self.val_bleu_2 = BLEUScore(n_gram=2)
        self.val_bleu_4 = BLEUScore(n_gram=4)
        self.val_perplexity = Perplexity()
        self.val_exact_match = ExactMatch(self.normalize_fn)

        # Test metrics
        self.test_bleu_1 = BLEUScore(n_gram=1)
        self.test_bleu_2 = BLEUScore(n_gram=2)
        self.test_bleu_4 = BLEUScore(n_gram=4)
        self.test_perplexity = Perplexity()
        self.test_exact_match = ExactMatch(self.normalize_fn)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def _prepare_texts_for_metrics(self, batch, preds):
        """Helper method to prepare texts for metric computation"""
        pred_texts = []
        for p in preds:
            text = self.tokenizer.decode(p, skip_special_tokens=True)
            # print(text, p)
            if len(text.strip()) == 0:
                print("prediction has zero len")
            pred_texts.append(text)

        # Handle multiple references for BLEU
        target_texts = []
        for i in range(len(batch["labels"])):
            target_list = []
            main_target = self.tokenizer.decode(
                batch["labels"][i], skip_special_tokens=True
            )
            if main_target.strip():
                target_list.append(main_target)

            if "labels2" in batch:
                second_target = self.tokenizer.decode(
                    batch["labels2"][i], skip_special_tokens=True
                )
                if second_target.strip():
                    target_list.append(second_target)

            if not target_list:
                target_list = [main_target] 

            target_texts.append(target_list)

        # For exact match - create normalized versions
        exact_match_targets = []
        for target_list in target_texts:
            normalized_targets = []
            for target in target_list:
                normalized_targets.extend([target, self.normalize_fn(target)])
            exact_match_targets.append(list(set(normalized_targets)))

        return pred_texts, target_texts, exact_match_targets

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)


        preds = torch.argmax(outputs.logits, dim=-1)
        pred_texts, target_texts, exact_match_targets = self._prepare_texts_for_metrics(
            batch, preds
        )

        # Update metrics
        self.val_bleu_1(pred_texts, target_texts)
        self.val_bleu_2(pred_texts, target_texts)
        self.val_bleu_4(pred_texts, target_texts)
        self.val_perplexity(outputs.logits, batch["labels"])
        self.val_exact_match(pred_texts, exact_match_targets)

        # Log a few examples occasionally
        if batch_idx == 0 and self.current_epoch % 5 == 0:  # Log every 5 epochs
            t = ""
            for i in range(min(10, len(pred_texts))):
                t += f"""Epoch {self.current_epoch} - Example {i}:
                Input: {self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)}
                Pred: {pred_texts[i]}
                Target: {target_texts[i][0]}
                ---\n"""
            self.logger.experiment.log_text(
                self.logger.run_id,
                t,
                f"validation_examples_epoch_{self.current_epoch}.txt",
            )

        return loss

    def on_validation_epoch_end(self):
        self.log("val_bleu_1", self.val_bleu_1.compute(), prog_bar=True, logger=True)
        self.log("val_bleu_2", self.val_bleu_2.compute(), prog_bar=True, logger=True)
        self.log("val_bleu_4", self.val_bleu_4.compute(), prog_bar=True, logger=True)
        self.log(
            "val_perplexity", self.val_perplexity.compute(), prog_bar=True, logger=True
        )
        self.log(
            "val_exact_match",
            self.val_exact_match.compute(),
            prog_bar=True,
            logger=True,
        )

        # Reset metrics
        self.val_bleu_1.reset()
        self.val_bleu_2.reset()
        self.val_bleu_4.reset()
        self.val_perplexity.reset()
        self.val_exact_match.reset()

    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)


        preds = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.max_target_len,
        )

        pred_texts, target_texts, exact_match_targets = self._prepare_texts_for_metrics(
            batch, preds
        )

        # Update metrics
        self.test_perplexity(outputs.logits, batch["labels"])
        self.test_bleu_1(pred_texts, target_texts)
        self.test_bleu_2(pred_texts, target_texts)
        self.test_bleu_4(pred_texts, target_texts)
        self.test_exact_match(pred_texts, exact_match_targets)

        return loss

    def on_test_epoch_end(self):
        self.log("test_bleu_1", self.test_bleu_1.compute(), prog_bar=True, logger=True)
        self.log("test_bleu_2", self.test_bleu_2.compute(), prog_bar=True, logger=True)
        self.log("test_bleu_4", self.test_bleu_4.compute(), prog_bar=True, logger=True)
        self.log(
            "test_perplexity",
            self.test_perplexity.compute(),
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_exact_match",
            self.test_exact_match.compute(),
            prog_bar=True,
            logger=True,
        )

        self.test_bleu_1.reset()
        self.test_bleu_2.reset()
        self.test_bleu_4.reset()
        self.test_perplexity.reset()
        self.test_exact_match.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    parser = argparse.ArgumentParser(
        description="Finetuning T5 with PyTorch Lightning and MLflow"
    )
    parser.add_argument("--input", type=str, required=True, help="train.csv path")
    parser.add_argument(
        "--description_column",
        type=str,
        default="description",
        help="Description column",
    )
    parser.add_argument(
        "--command_column", type=str, default="command", help="Command column"
    )
    parser.add_argument("--max_epochs", type=int, default=10, help="Epoches count")
    parser.add_argument(
        "--test", type=str, default="data/test.csv", help="test csv file"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--experiment_name", type=str, default="T5-Training-New", help="MLflow exp name"
    )
    parser.add_argument(
        "--run_id", type=str, default=None, help="Mlflow run id for resume training"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="path to checkpoing.ckpt file"
    )

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)
    model = T5Model(lr=args.lr, model_name=MODEL_NAME, tokenizer=tokenizer)

    train_dataset = pd.read_csv(args.input)
    mlflow_train_dataset = mlflow.data.from_pandas(
        train_dataset, source=args.input, name="train-data"
    )
    test_dataset = pd.read_csv(args.test)
    mlflow_test_dataset = mlflow.data.from_pandas(
        test_dataset, source=args.test, name="test-data"
    )

    data_module = T5DataModule(
        train_csv_file=train_dataset,
        test_csv_file=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        description_column=args.description_column,
        command_column=args.command_column,
    )

    mlflow.set_tracking_uri(MLFLOW_ADDR)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_id=args.run_id) as run:
        mlflow.log_inputs(
            datasets=[mlflow_train_dataset, mlflow_test_dataset],
            contexts=["train", "test"],
            tags_list=[None, None],
        )

        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "description_column": args.description_column,
                "command_column": args.command_column,
                "architecture": "T5ForConditionalGeneration",
                "task": "text2text-generation",
            }
        )

        mlflow.set_tags(
            {
                "architecture": "T5ForConditionalGeneration",
                "task": "text2text-generation",
                "framework": "pytorch",
                "library": "transformers",
            }
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_bleu_2",
            dirpath="./checkpoints",
            filename=MODEL_NAME + "-{epoch:02d}-{val_bleu_2:.2f}",
            save_top_k=3,
            mode="max",
        )
        early_stopping_checkpoint = EarlyStopping(
            "val_bleu_2", min_delta=0.01, mode="max"
        )

        mlflow_logger = MLFlowLogger(
            experiment_name="T5 Training",
            tracking_uri=MLFLOW_ADDR,
            run_id=run.info.run_id,
        )

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=mlflow_logger,
            log_every_n_steps=500,
            accelerator="auto",
            devices="auto",
            callbacks=[checkpoint_callback, early_stopping_checkpoint],
        )

        # trainer.validate(model, data_module, ckpt_path=args.checkpoint_path)

        trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)

        model = T5Model.load_from_checkpoint(checkpoint_callback.best_model_path)
        trainer.test(model, data_module)

        # Create signature for the model
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec

        input_schema = Schema(
            [
                ColSpec("string"),
            ]
        )
        output_schema = Schema(
            [
                ColSpec("string"),
            ]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        components = {
            "model": model.model,
            "tokenizer": tokenizer,
        }

        # Log model with signature and additional metadata
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="t5-model",
            signature=signature,
            input_example="list all files in current directory",
            task="text2text-generation",
            architecture="T5ForConditionalGeneration",
        )


if __name__ == "__main__":
    main()
