"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import datetime
import json
import os
import random
from io import BytesIO
from os.path import basename
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config
from hilayout_acc.lightning_module import LayoutTransformerModelPLModule, LayoutTransformerDataPLModule
from hilayout_acc.util import LayoutTransformerDataset
from pytorch_lightning.strategies import DeepSpeedStrategy, SingleDeviceStrategy
import torch.multiprocessing


class NotLoadOptimizerStateStrategy(SingleDeviceStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def lightning_restore_optimizer():
        return False

class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path)
        checkpoint.pop("optimizer_states", None)
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision("medium")
    
    pl.seed_everything(config.get("seed", 42), workers=True)

    # model_module = DonutModelPLModule.load_from_checkpoint(Path(config.result_path) / config.exp_name / config.exp_version/ "artifacts-v8.ckpt")
    # if config.load_from_checkpoint:
    #     model_module = LayoutTransformerModelPLModule.load_from_checkpoint(Path(config.pretrained_model_name_or_path) / "artifacts.ckpt/checkpoint/mp_rank_00_model_states.pt", config=config)
    # else:
    model_module = LayoutTransformerModelPLModule(config)
    data_module = LayoutTransformerDataPLModule(config)

    # add datasets to data_module
    datasets = {"train": [], "val": []}
    for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
        task_name = os.path.basename(dataset_name_or_path)  # e.g., cord-v2, docvqa, rvlcdip, ...
        
        # add categorical special tokens (optional)
        if task_name == "rvlcdip":
            model_module.model.decoder.add_special_tokens([
                "<advertisement/>", "<budget/>", "<email/>", "<file_folder/>", 
                "<form/>", "<handwritten/>", "<invoice/>", "<letter/>", 
                "<memo/>", "<news_article/>", "<presentation/>", "<questionnaire/>", 
                "<resume/>", "<scientific_publication/>", "<scientific_report/>", "<specification/>"
            ])
        if task_name == "docvqa":
            model_module.model.decoder.add_special_tokens(["<yes/>", "<no/>"])
            
        for split in ["train", "val"]:
            datasets[split].append(
                LayoutTransformerDataset(
                    dataset_name_or_path=dataset_name_or_path,
                    model=model_module.model,
                    max_length=config.max_length,
                    split=split,
                    task_start_token=config.task_start_tokens[i]
                    if config.get("task_start_tokens", None)
                    else f"<s_{task_name}>",
                    prompt_end_token="<s_answer>" if "docvqa" in dataset_name_or_path else f"<s_{task_name}>",
                    sort_json_key=config.sort_json_key,
                    page_tokens=config.page_tokens,
                    max_pages=config.max_pages
                )
            )
            # prompt_end_token is used for ignoring a given prompt in a loss function
            # for docvqa task, i.e., {"question": {used as a prompt}, "answer": {prediction target}},
            # set prompt_end_token to "<s_answer>"
    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["val"]

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    custom_ckpt = CustomCheckpointIO()
    profiler = AdvancedProfiler(dirpath=".", filename="profiler.log")
    trainer = pl.Trainer(
        enable_checkpointing=config.get("enable_checkpointing", None),
        num_nodes=config.get("num_nodes", 1),
        # gpus=torch.cuda.device_count(),
        # strategy="ddp",
        strategy=NotLoadOptimizerStateStrategy(device=torch.device("cuda:0")),
        profiler=profiler,
        overfit_batches=config.get("overfit_batches", 0.0),
        accelerator="gpu",
        # strategy=DeepSpeedStrategy(stage=2, offload_optimizer=True, load_full_weights=True),
        plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.get("val_check_interval", 1.0),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        gradient_clip_val=config.gradient_clip_val,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )
    
    trainer.strategy.lightning_restore_optimizer = False

    trainer.fit(model_module, data_module,
                ckpt_path=config.pretrained_model_name_or_path
                if config.load_from_checkpoint else None)


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()
    
    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version

    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
    train(config)
