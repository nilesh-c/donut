"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import jsonlines
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
import numpy as np

import torch
from tqdm import tqdm
import yaml
import zss
from datasets import load_dataset
# from nltk import edit_distance
from rapidfuzz.distance.Levenshtein import distance as lv_distance
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from zss import Node
from PIL import Image
import contextlib
import joblib
from joblib import Parallel, delayed
from hilayout_acc.HiLT5 import Proxy_HiLT5


def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w") as f:
        json.dump(save_obj, f)

def save_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f)

def load_json(json_path: Union[str, bytes, os.PathLike]):
    with open(json_path, "r") as f:
        return json.load(f)

def normalize_bbox(bbox, image_size):
    width, height = image_size
    x0, y0, x1, y1 = bbox
    return [
        np.clip(int(1000 * x0 / width), 0, 1000),
        np.clip(int(1000 * y0 / height), 0, 1000),
        np.clip(int(1000 * x1 / width), 0, 1000),
        np.clip(int(1000 * y1 / height),  0, 1000)
    ]

def unnormalize_bbox(bbox, image_size):
    width, height = image_size
    x0, y0, x1, y1 = bbox
    return [
        (x0 * width) / 1000,
        (y0 * height) / 1000,
        (x1 * width) / 1000,
        (y1 * height) / 1000
    ]
    
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class LayoutTransformerDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    num_pages = 10

    def __init__(
        self,
        dataset_name_or_path: str,
        model: Proxy_HiLT5,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
        page_tokens: int = 10,
        max_pages: int = 10,
        use_images: bool = False
    ):
        super().__init__()

        self.model = model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token else task_start_token
        )
        self.sort_json_key = sort_json_key
        self.dataset_name_or_path = dataset_name_or_path
        self.page_tokens = page_tokens
        self.max_pages = max_pages
        self.use_images = use_images

        print(dataset_name_or_path, self.split)
        self.dataset = self.load_dataset_multipage(
            dataset_name_or_path, split=self.split
        )
        self.dataset_length = len(self.dataset)
        print("Size of dataset split {}: {}".format(self.split, self.dataset_length))

        self.gt_token_sequences = []
        gt_jsons = []
        for sample in self.dataset:                
            ground_truth = sample["ground_truth"]

            if (
                "gt_parses" in ground_truth
            ):  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(
                    ground_truth["gt_parse"], dict
                )
                gt_jsons.append(ground_truth["gt_parse"])

        all_special_tokens = set()
        
        with tqdm_joblib(tqdm(desc="Parallel processing for split {}".format(self.split), total=len(gt_jsons))) as progress_bar:
            parsed = Parallel(n_jobs=2, verbose=2, backend='multiprocessing', prefer='processes')(
                delayed(self.json2token)(gt_json, sort_json_key)
                for gt_json in tqdm(gt_jsons)
            )
            
        for gt_tokens, special_tokens in parsed:
            self.gt_token_sequences.append(gt_tokens)
            all_special_tokens.update(special_tokens)
        
        if self.split == "train":
            self.model.add_special_tokens(all_special_tokens)

        self.model.add_special_tokens(
            [self.task_start_token, self.prompt_end_token]
        )
        self.prompt_end_token_id = (
            self.model.tokenizer.convert_tokens_to_ids(
                self.prompt_end_token
            )
        )

    def json2token(
        self,
        obj: Any,
        sort_json_key: bool = True
    ):
        def _json2token(
            self,
            obj: Any,
            sort_json_key: bool = True,
            all_special_tokens: Set[str] = None
        ):
            if type(obj) == dict:
                if len(obj) == 1 and "text_sequence" in obj:
                    return obj["text_sequence"]
                else:
                    output = ""
                    if sort_json_key:
                        keys = sorted(obj.keys(), reverse=True)
                    else:
                        keys = obj.keys()
                    for k in keys:
                        all_special_tokens.update([rf"<s_{k}>", rf"</s_{k}>"])
                            
                        output += (
                            rf"<s_{k}>"
                            + _json2token(
                                obj[k],
                                sort_json_key,
                                all_special_tokens=all_special_tokens,
                            )
                            + rf"</s_{k}>"
                        )
                    return output
            elif type(obj) == list:
                return r"<sep/>".join(
                    [
                        _json2token(
                            item, sort_json_key
                        )
                        for item in obj
                    ]
                )
            else:
                obj = str(obj)
                if f"<{obj}/>" in all_special_tokens:
                    obj = f"<{obj}/>"  # for categorical special tokens
                return obj
            
        all_special_tokens = set()
        obj = self.task_start_token + _json2token(self, obj, sort_json_key, all_special_tokens) + self.model.tokenizer.eos_token
        return [obj], all_special_tokens

    def __len__(self) -> int:
        return self.dataset_length
    
    def _get_boxes(self, tokens, page_size):
        boxes = [(token["x0"], token["y0"], token["x1"], token["y1"]) for token in tokens]
        return [normalize_bbox(box, page_size) for box in boxes]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]
        num_pages = len(sample["page_names"])
        doc_name = sample["doc_name"]
        page_path = os.path.join(self.dataset_name_or_path, "images")
        page_size = sample["page_size"]
        page_size = page_size["width"], page_size["height"]

        words = []
        boxes = []
        context = []
        image_paths = []

        for page_ix in range(min(num_pages, self.max_pages)):
            words.append([token["token_text"] for token in sample['tokens'][page_ix]])
            boxes.append(np.array(self._get_boxes(sample['tokens'][page_ix], page_size), dtype=np.float32))
            context.append(' '.join([token["token_text"] for token in sample['tokens'][page_ix]]))
            image_paths.append(os.path.join(page_path, f"{doc_name}_page_{page_ix+1}.png"))

        input_text = ["{:s}: context: {:s}".format("[PAGE]" * self.page_tokens, c) for c in context]
        tokens = self.model.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
    
        context_page_corresp = None

        if num_pages < self.max_pages:
            for _ in range(self.max_pages - num_pages):
                words.append([''])
                boxes.append(np.zeros([1, 4], dtype=np.float32))

        if self.use_images:
            images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
            images += [Image.new('RGB', (2, 2)) for i in range(self.max_pages - len(image_paths))]  # Pad with 2x2 images.

        # input_ids
        try:
            processed_parse = random.choice(
                self.gt_token_sequences[idx]
            )  # can be more than one, e.g., DocVQA Task 1
        except Exception as a:
            print(self.gt_token_sequences[idx])
            raise a
        labels = self.model.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        
        prompt_end_index = None

        # if self.split == "train":
        #     labels[
        #         labels == self.model.tokenizer.pad_token_id
        #     ] = self.ignore_id  # model doesn't need to predict pad token
        #     labels[
        #         : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
        #     ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        # else:
        #     prompt_end_index = torch.nonzero(
        #         labels == self.prompt_end_token_id
        #     ).sum()  # return prompt end index instead of target output labels
            
        sample_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words": words,
            "boxes": boxes,
            "context": context,
            "num_pages": num_pages,
            "processed_parse": processed_parse,
            "labels": labels
        }
        
        return sample_dict
        
        
    @classmethod
    def load_dataset_multipage(cls, dataset_name_or_path, split):
        import itertools
        json_path = os.path.join(dataset_name_or_path, split, "metadata.jsonl")
        
        with jsonlines.open(json_path) as reader:
            return [data for data in itertools.islice(reader, 8)]
            # return [data for data in reader]
                


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(
                        child_value, f"{key}.{child_key}" if key else child_key
                    )
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return lv_distance(
                label1.replace("<leaf>", ""), label2.replace("<leaf>", "")
            )
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [
                    str(item).strip()
                    for item in data
                    if type(item) in {str, int, float} and str(item).strip()
                ]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(
                self.normalize_dict(pred)
            ), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(
        self, data: Union[Dict, List], node_name: str = None
    ):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )