import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow
import mlflow.transformers
import git

class CommandDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_source_len=128, max_target_len=64, 
                 description_column='description', command_column='command'):
        self.data = pd.read_csv(csv_file)
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

    def setup(self, stage=None):
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
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

class T5Model(pl.LightningModule):
    def __init__(self, model_name='t5-small', lr=1e-4):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

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
    model = T5Model(lr=args.lr)
    
    data_module = T5DataModule(
        train_csv_file=args.input,
        test_csv_file="data/test.csv",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        description_column=args.description_column,
        command_column=args.command_column
    )

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:

        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha

        mlflow.log_params({
            'model_name': 't5-small',
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'description_column': args.description_column,
            'command_column': args.command_column,
        })
        

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='./checkpoints',
            filename='t5-best-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min'
        )

        mlflow.log_metric('train_samples', len(data_module.train_dataset))
        mlflow.log_metric('val_samples', len(data_module.val_dataset))
        mlflow.log_metric('test_samples', len(data_module.test_dataset))


        mlflow_logger = MLFlowLogger(
            experiment_name="T5 Training",
            tracking_uri="./mlruns",
            run_id=run.info.run_id
        )

        mlflow_logger.run_tags = {"git_commit": commit_hash}
        # Тренер
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=mlflow_logger,
            log_every_n_steps=10,
            accelerator='auto',
            devices='auto',
            callbacks=[checkpoint_callback],
        )

        # Запуск обучения
        trainer.fit(model, data_module)
        trainer.test(model, data_module)

        

    # Сохранение модели
    trainer.save_checkpoint("t5_model_final.ckpt")
    tokenizer.save_pretrained("./t5_final_model")

if __name__ == "__main__":
    main()