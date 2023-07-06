"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import random
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
# from nltk import edit_distance
from rapidfuzz.distance.Levenshtein import distance as edit_distance
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
# from torchsummary import summary
from donut import DonutConfig, DonutModel
from hilayout_acc.HiLT5 import Proxy_HiLT5
from hilayout_acc.util import save_yaml
from deepspeed.ops.adam import DeepSpeedCPUAdam
from nonechucks import SafeDataset


class LayoutTransformerModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.accelerator = Accelerator()

        if not self.config.get("pretrained_model_name_or_path", False):
            self.config["pretrained_model_name_or_path"] = "t5-large"
        self.model = Proxy_HiLT5(self.config)
        
    def training_step(self, batch, batch_idx):
        outputs, _ = self.model.forward(batch, return_pred_parse=False)
        loss = outputs.loss + outputs.ret_loss if hasattr(outputs, 'ret_loss') else outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        gt_parses = batch["processed_parse"]
        _, pred_parses = self.model.forward(batch, return_pred_parse=True, return_json=False)

        scores = list()
        for pred, gt in zip(pred_parses, gt_parses):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            gt = re.sub(r"<.*?>", "", gt, count=1)
            gt = gt.replace(self.model.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, gt) / max(len(pred), len(gt)))

            if self.config.get("verbose", False) and len(scores) == 1 and len(self.validation_step_outputs) == 0:
                self.print(f"  Prediction: {pred}")
                self.print(f"Ground truth: {gt}")
                self.print(f"   Normed ED: {scores[0]}")

        self.validation_step_outputs.append(scores)
        return scores
    
    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = list()
        self.model.model.eval()

    def on_validation_epoch_end(self):
        self.model.model.train()
        
        validation_step_outputs = self.validation_step_outputs
        num_of_loaders = len(self.config.dataset_name_or_paths)
        if num_of_loaders == 1:
            validation_step_outputs = [validation_step_outputs]
        assert len(validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True, prog_bar=True, logger=True)
        log_dict = {"val_metric": np.sum(total_metric) / np.sum(cnt)}
        
        if self.config.get("verbose", False):
            print("Average Normed ED:", log_dict)
        
        self.log_dict(log_dict, sync_dist=True, prog_bar=True, logger=True)

    def configure_optimizers(self):

        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        # optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.config.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = Path(self.config.result_path) / self.config.exp_name / self.config.exp_version
        self.save_pretrained(save_path)
        
    @rank_zero_only
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = self.model.state_dict()
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        
    def save_pretrained(self, save_path):
        model = self.model
        ckpt_path = save_path / "model.ckpt"
        model.model.save_pretrained(ckpt_path)

        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.processor if hasattr(model, 'processor') else None
        if tokenizer is not None:
            tokenizer.save_pretrained(ckpt_path)

        if hasattr(model.model, 'visual_embeddings'):
            model.model.visual_embeddings.feature_extractor.save_pretrained(ckpt_path)

        save_yaml(ckpt_path / "experiment_config.yaml", self.config)

        # if update_best:
        #     model.model.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        #     tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        #     save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), kwargs)


    def load_model(base_model, ckpt_name, **kwargs):
        load_dir = kwargs['save_dir']
        base_model.model.from_pretrained(os.path.join(load_dir, ckpt_name))


def singlepage_docvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    return batch

class LayoutTransformerDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        loaders = list()
        for train_dataset, batch_size in zip(self.train_datasets, self.train_batch_sizes):
            loaders.append(
                DataLoader(
                    SafeDataset(train_dataset),
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    prefetch_factor=self.config.prefetch_factor,
                    pin_memory=True,
                    worker_init_fn=self.seed_worker,
                    generator=self.g,
                    shuffle=True,
                    collate_fn=singlepage_docvqa_collate_fn
                )
            )
        return loaders[0]

    def val_dataloader(self):
        loaders = list()
        for val_dataset, batch_size in zip(self.val_datasets, self.val_batch_sizes):
            loaders.append(
                DataLoader(
                    SafeDataset(val_dataset),
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    prefetch_factor=self.config.prefetch_factor,
                    pin_memory=True,
                    shuffle=False,
                    collate_fn=singlepage_docvqa_collate_fn
                )
            )
        return loaders[0]

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
