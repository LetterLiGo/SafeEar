import importlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import argparse
import pytorch_lightning as pl
import torch
import hydra

torch.set_float32_matmul_precision("high")

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def print_only(message: str):
    """Prints a message only on rank 0."""
    print(message)
    
def train(cfg: DictConfig, args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    # instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    # instantiate decouple model
    print_only(f"Instantiating decouple model <{cfg.decouple_model._target_}>")
    decouple_model: torch.nn.Module = hydra.utils.instantiate(cfg.decouple_model)
    decouple_model.load_state_dict(torch.load(cfg.speechtokenizer_path))
    # import pdb; pdb.set_trace()
    
    # instantiate detect model
    print(f"Instantiating detect model <{cfg.detect_model._target_}>")
    detect_model: torch.nn.Module = hydra.utils.instantiate(cfg.detect_model)
    # import pdb; pdb.set_trace()
    
    # instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        decouple_model=decouple_model,
        detect_model=detect_model,
    )
    
    # instantiate trainer
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    trainer.test(system, datamodule=datamodule, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="local/conf.yml",
        help="Full path to save best validation model",
    )
    parser.add_argument(
        "--ckpt_path",
        help="Full path to save best validation model",
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    # 保存配置到新的文件
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    train(cfg, args)
    