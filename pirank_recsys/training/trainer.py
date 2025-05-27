"""Training module using PyTorch Lightning."""

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from typing import Dict, Any

from ..models.pirank import PiRankModel
from ..utils.logging import setup_mlflow_logging


class PiRankLightningModule(pl.LightningModule):
    """PyTorch Lightning module for PiRank."""

    def __init__(self, model_config: Dict[str, Any], learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = PiRankModel(**model_config)
        self.learning_rate = learning_rate

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = self.model.compute_loss(batch)
        metrics = self.model.compute_metrics(batch)
        
        self.log("val_loss", loss)
        self.log("val_ndcg", metrics["ndcg"], prog_bar=True)
        self.log("val_map", metrics["map"])
        self.log("val_pfound", metrics["pfound"])

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        metrics = self.model.compute_metrics(batch)
        
        self.log("test_ndcg", metrics["ndcg"])
        self.log("test_map", metrics["map"])
        self.log("test_pfound", metrics["pfound"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_model(cfg: Dict[str, Any]) -> None:
    """Train the PiRank model."""
    from ..data.dataset import IstellaDataset, create_data_splits
    from pathlib import Path

    # Setup MLflow logging
    setup_mlflow_logging(cfg.experiment_name, cfg.mlflow_tracking_uri)

    # Create data splits
    data_path = Path(cfg.data.data_path)
    create_data_splits(
        data_path, cfg.data.train_split, cfg.data.val_split, cfg.random_seed
    )

    # Create datasets
    train_dataset = IstellaDataset(data_path, "train")
    val_dataset = IstellaDataset(data_path, "val")
    test_dataset = IstellaDataset(data_path, "test")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )

    # Create model
    model = PiRankLightningModule(cfg.model, cfg.model.learning_rate)

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor=cfg.training.monitor,
        patience=cfg.training.patience,
        mode=cfg.training.mode,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.monitor,
        mode=cfg.training.mode,
        save_top_k=1,
        filename="best-{epoch}-{val_ndcg:.3f}",
    )

    # Setup logger
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        logger=mlflow_logger,
        callbacks=[early_stopping, checkpoint_callback],
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    trainer.test(model, test_loader, ckpt_path="best")
