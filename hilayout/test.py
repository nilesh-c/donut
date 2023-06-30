"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
import re
from pathlib import Path
import jsonlines

import numpy as np
import torch
from datasets import load_dataset
from donut.util import DonutDataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json

entity_types = [# candidate columns
                "candidates_ent_Title__c",
                "candidates_cxsrec__Address_line_1__c",
                "candidates_cxsrec__Country__c",
                "candidates_cxsrec__Current_department__c",
                "candidates_cxsrec__Current_position__c",
                "candidates_cxsrec__First_name__c",
                "candidates_cxsrec__Last_name__c",
                "candidates_cxsrec__Date_of_birth__c",
                "candidates_cxsrec__Language_rel__c",
                "candidates_cxsrec__E_mail_address__c",
                "candidates_cxsrec__Work_e_mail__c",
                "candidates_cxsrec__mobilePhone__c",
                "candidates_ent_Working_Mobile__c",
                "candidates_cxsrec__privatePhone__c",
                "candidates_ent_Working_Phone__c",
                "candidates_ent_Skype_ID__c",
                "candidates_ent_Private_address__Street__s",
                "candidates_ent_Private_address__PostalCode__s",
                "candidates_ent_Private_address__City__s",
                "candidates_ent_Private_address__StateCode__s",
                "candidates_ent_Private_address__CountryCode__s",
                "candidates_cxsrec__Linked_in_URL__c",
                
                
                # work experience columns
                # "cxsrec__cxsWork_Experience__c",
                "workexperience_Name",
                "workexperience_cxsrec__Start_date__c",
                "workexperience_cxsrec__End_date__c",
                "workexperience_cxsrec__Employer_Name_and_Place__c",
                "workexperience_ent_Employer_description__c",
                "workexperience_cxsrec__Description__c",
                "workexperience_ent_Responsible_for__c",
                "workexperience_ent_Highlights__c",
                # "cxsrec__Candidate__c_workexperience",
                
                
                # education columns
                # "cxsrec__cxsEducation__c",
                "education_Name",
                "education_cxsrec__Start_date__c",
                "education_cxsrec__End_date__c",
                "education_cxsrec__Institute_Name_and_Place__c",
                "education_cxsrec__Level__c",
                "education_cxsrec__Period__c",
                "education_ent_Time_period__c",
                # "cxsrec__Candidate__c_education"
]

def sort_json_dict(json_dict):
    index_map = {v: i for i, v in enumerate(entity_types)}
    for key in json_dict:
        json_dict[key] = json_dict[key] if isinstance(json_dict[key], dict) else json_dict[key][0]
        if isinstance(json_dict[key], dict):
            json_dict[key] = dict(sorted(json_dict[key].items(),
                                        key=lambda pair: index_map[pair[0]]))
    return json_dict


def test(args):
    args.task_name = os.path.basename(args.dataset_name_or_path)
    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

    # if torch.cuda.is_available():
    #     pretrained_model.half()
    #     pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    predictions = []
    ground_truths = []
    accs = []

    evaluator = JSONParseEvaluator()
    dataset = DonutDataset.load_dataset_multipage(args.dataset_name_or_path, split=args.split)
    with jsonlines.open(args.save_path + ".jsonl", mode="w", flush=True) as writer:
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            # print(sample)
            ground_truth = sample["ground_truth"]

            num_pages = len(sample["page_names"])
            doc_name = sample["doc_name"]
            page_path = os.path.join(args.dataset_name_or_path, "images")
            max_page_num = args.max_page_num
            C, H, W, P = 3, 2560, 1920, max_page_num
            # C, H, W, P = 3, 1280, 960, max_page_num  ##########
            input_tensor = torch.zeros(size=(1, C, H, W, P))

            for i in range(min(num_pages, max_page_num)):
                image_path = os.path.join(page_path, f"{doc_name}_page_{i+1}.png")
                if not os.path.exists(image_path):
                        continue
                image_tensor = Image.open(image_path)
                image_tensor = pretrained_model.encoder.prepare_input(
                    image_tensor, random_padding=False
                )
                input_tensor[0, :, :, :, i] = image_tensor

            if args.task_name == "docvqa":
                output = pretrained_model.inference(
                    image=sample["image"],
                    prompt=f"<s_{args.task_name}><s_question>{ground_truth['gt_parses'][0]['question'].lower()}</s_question><s_answer>",
                )["predictions"][0]
            else:
                output = pretrained_model.inference(image_tensors=input_tensor,
                                                    prompt=f"<s_{args.task_name}>")["predictions"][0]

            if args.task_name == "rvlcdip":
                gt = ground_truth["gt_parse"]
                score = float(output["class"] == gt["class"])
            elif args.task_name == "docvqa":
                # Note: we evaluated the model on the official website.
                # In this script, an exact-match based score will be returned instead
                gt = ground_truth["gt_parses"]
                answers = set([qa_parse["answer"] for qa_parse in gt])
                score = float(output["answer"] in answers)
            else:
                gt = ground_truth["gt_parse"]
                score = evaluator.cal_acc(output, gt)

            accs.append(score)
            
            output = sort_json_dict(output)
            gt = sort_json_dict(gt)

            predictions.append(output)
            ground_truths.append(gt)
            
            writer.write({"doc_name": sample["doc_name"], "score": score, "prediction": output, "ground_truth": gt})

    scores = {
        "ted_accuracies": accs,
        "ted_accuracy": np.mean(accs),
        "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
    }
    print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, F1 accuracy score: {scores['f1_accuracy']}"
    )

    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        save_json(args.save_path, scores)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--max_page_num", type=int, default=4   )
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predictions = test(args)
