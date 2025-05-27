"""Main entry point for all commands."""

import fire
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pirank_recsys.training.trainer import train_model
from pirank_recsys.inference.predictor import run_inference


def train(config_name: str = "config") -> None:
    """Train the PiRank model."""
    config_dir = Path(__file__).parent / "configs"
    
    with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        cfg = compose(config_name=config_name)
        train_model(OmegaConf.to_container(cfg, resolve=True))


def infer(
    model_path: str,
    data_path: str,
    output_path: str,
    config_name: str = "config"
) -> None:
    """Run inference on new data."""
    config_dir = Path(__file__).parent / "configs"
    
    with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        cfg = compose(config_name=config_name)
        run_inference(
            Path(model_path),
            Path(data_path),
            Path(output_path),
            OmegaConf.to_container(cfg, resolve=True)
        )


def convert_to_production(model_path: str, output_dir: str) -> None:
    """Convert models to production formats."""
    from convert_to_production import convert_model
    convert_model(Path(model_path), Path(output_dir))


if __name__ == "__main__":
    fire.Fire()
