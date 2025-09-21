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
from mlflow.data.meta_dataset import MetaDataset


class CommandDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_source_len=128, max_target_len=64, 
                 description_column='description', command_column='command'):
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
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            command,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source['input_ids'].flatten(),
            'attention_mask': source['attention_mask'].flatten(),
            'labels': target['input_ids'].flatten()
        }

class T5DataModule(pl.LightningDataModule):
    def __init__(self, train_csv_file, test_csv_file, tokenizer, batch_size=16, max_source_len=128,
                 max_target_len=64, description_column='description', command_column='command'):
        super().__init__()
        self.csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.description_column = description_column
        self.command_column = command_column

        dataset = CommandDataset(
            self.csv_file,
            self.tokenizer,
            self.max_source_len,
            self.max_target_len,
            self.description_column,
            self.command_column
        )
        self.test_dataset = CommandDataset(
            self.test_csv_file, 
            self.tokenizer,
            self.max_source_len,
            self.max_target_len,
            self.description_column,
            self.command_column)
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

class T5Model(pl.LightningModule):
    def __init__(self, tokenizer, model_name='t5-small', lr=1e-4, max_target_len=64):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.lr = lr
        self.max_target_len = max_target_len
        self.save_hyperparameters()

        self.val_bleu = BLEUScore(2)
        self.test_bleu = BLEUScore(2)
        self.val_perplexity = Perplexity()
        self.test_perplexity = Perplexity()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        preds = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.max_target_len
        )
        target_texts = [[self.tokenizer.decode(t, skip_special_tokens=True)] for t in batch['labels']]
        pred_texts = [self.tokenizer.decode(p, skip_special_tokens=True) for p in preds]

        self.val_bleu(pred_texts, target_texts)
        self.val_perplexity(outputs.logits, batch['labels'])

        return loss

    def on_validation_epoch_end(self):
        self.log('val_bleu', self.val_bleu.compute(), prog_bar=True)
        self.log('val_perplexity', self.val_perplexity.compute(), prog_bar=True)
        self.val_bleu.reset()
        self.val_perplexity.reset()

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)

        preds = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.max_target_len
        )
        target_texts = [[self.tokenizer.decode(t, skip_special_tokens=True)] for t in batch['labels']]
        pred_texts = [self.tokenizer.decode(p, skip_special_tokens=True) for p in preds]

        self.test_bleu(pred_texts, target_texts)
        self.test_perplexity(outputs.logits, batch['labels'])

        return loss

    def on_test_epoch_end(self):
        self.log('test_bleu', self.test_bleu.compute(), prog_bar=True)
        self.log('test_perplexity', self.test_perplexity.compute(), prog_bar=True)
        self.test_bleu.reset()
        self.test_perplexity.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    parser = argparse.ArgumentParser(description='Обучение T5 с использованием PyTorch Lightning и MLflow')
    parser.add_argument('--input', type=str, required=True, help='Путь к CSV файлу')
    parser.add_argument('--description_column', type=str, default='description', help='Колонка с описаниями')
    parser.add_argument('--command_column', type=str, default='command', help='Колонка с командами')
    parser.add_argument('--max_epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default='T5-Training', help='Название эксперимента MLflow')
    
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Model(lr=args.lr, tokenizer=tokenizer)

    train_dataset = pd.read_csv(args.input)
    mlflow_train_dataset = mlflow.data.from_pandas(train_dataset, source=args.input, name="train-data")
    test_dataset = pd.read_csv("data/test.csv")
    mlflow_test_dataset = mlflow.data.from_pandas(test_dataset, source="data/test.csv", name="test-data")


    data_module = T5DataModule(
        train_csv_file=train_dataset,
        test_csv_file=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        description_column=args.description_column,
        command_column=args.command_column
    )

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
    
        mlflow.log_inputs(datasets=[mlflow_train_dataset, mlflow_test_dataset], contexts=["train", "test"], tags_list=[None, None])

        mlflow.log_params({
            'model_name': 't5-small',
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'description_column': args.description_column,
            'command_column': args.command_column,
        })
        

        checkpoint_callback = ModelCheckpoint(
            monitor='val_bleu',
            dirpath='./checkpoints',
            filename='t5-best-{epoch:02d}-{val_bleu:.2f}',
            save_top_k=1,
            mode='max'
        )
        early_stopping_checkpoint = EarlyStopping("val_bleu", min_delta=0.01, mode="max" )

        mlflow_logger = MLFlowLogger(
            experiment_name="T5 Training",
            tracking_uri="./mlruns",
            run_id=run.info.run_id
        )

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=mlflow_logger,
            log_every_n_steps=10,
            accelerator='auto',
            devices='auto',
            callbacks=[checkpoint_callback, early_stopping_checkpoint],
        )

        trainer.fit(model, data_module)

        model = T5Model.load_from_checkpoint(checkpoint_callback.best_model_path)
        trainer.test(model, data_module)

        components = {
                    "model": model.model,
                    "tokenizer": tokenizer,
                }
        mlflow.transformers.log_model(components, "t5-model")

if __name__ == "__main__":
    main()