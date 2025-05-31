# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import torch
import json
import base64
import requests
from io import BytesIO
from openai import OpenAI
from openai import AzureOpenAI
import pandas as pd
import random
from PIL import Image
import pdb
from collections import Counter
from filelock import FileLock
import csv
import torch.distributed as dist
import time


from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration, DonutProcessor, VisionEncoderDecoderModel, Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoProcessor, VisionEncoderDecoderConfig
from qwen_vl_utils import process_vision_info
from PIL import Image

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config




collected_data = []


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (list[str]):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    selector_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model that selects between table conversion models"},
    )


def extract_model_choice2(content):
    """
    Extract flat tool IDs from unified format. If "assistant" is found in the content,
    only consider <answer> tags that appear after "assistant". Otherwise, process the entire content.
    
    Returns:
        list[int] or None if format is invalid
    """
    # Find where "assistant" appears in the content
    assistant_pos = content.find("assistant")
    
    # If "assistant" is not found, process the entire content
    # Otherwise, only consider content after "assistant"
    content_to_process = content[assistant_pos:] if assistant_pos != -1 else content
    
    # Search for answer tag in the content
    match = re.search(r"<answer>(.*?)</answer>", content_to_process)
    if not match:
        return None
    
    try:
        tool_ids = [int(x.strip()) for x in match.group(1).split(",")]
    except ValueError:
        return None
    
    # Valid flat tool ID range: 0 to 7
    if any(t < 0 or t > 8 for t in tool_ids):
        return None
    
    return tool_ids




def extract_model_choice(content):
    # First try to extract content following "assistant"
    assistant_pattern = re.search(r"assistant\s*\n?(.*?)($|\n\n)", content, re.DOTALL)
    if assistant_pattern:
        extracted_content = assistant_pattern.group(1).strip()
    else:
        # If no "assistant" pattern, use the entire content
        extracted_content = content.strip()
    
    # Check for <answer> tags
    answer_pattern = re.search(r"<answer>(.*?)</answer>", extracted_content)
    if answer_pattern:
        extracted_content = answer_pattern.group(1).strip()
    else:
        # Check for array notation [0, 1, 2]
        array_pattern = re.search(r"\[(.*?)\]", extracted_content)
        if array_pattern:
            extracted_content = array_pattern.group(1).strip()
        # If nothing matched, we'll use the entire extracted content
        # This handles cases like "0, 1, 2" directly
    
    # Now extract the numbers
    try:
        # Split by comma and convert to integers
        tool_ids = []
        for item in extracted_content.split(','):
            item = item.strip()
            # Skip empty items
            if not item:
                continue
            num = int(item)
            # Only add if in valid range
            if 0 <= num <= 8:
                tool_ids.append(num)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tool_ids = [x for x in tool_ids if not (x in seen or seen.add(x))]
        return unique_tool_ids if unique_tool_ids else None
    except ValueError:
        return None
    




def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


QUESTION_TEMPLATE = "You are an expert agent specialized in selecting tools to solve **chart reasoning tasks**. You are provided with access to **9 tools**, indexed from **0 to 8**. Each tool belongs to one of three function types:- **type1**- **type2**- **type3**Each tool is implemented differently, even if they share the same type. Treat all tools as **independent**, but be aware that tools of the same type often offer similar capabilities.\n\n\
**Function:** \n\
0: type1 (A)\n\
1: type1 (B)\n\
2: type1 (C)\n\
3: type2 (D)\n\
4: type2 (E)\n\
5: type2 (F)\n\
6: type3 (G)\n\
7: type3 (H)\n\
8: type3 (I)\n\
**Query:** {Question}\n\n\
Your job: \n\
1. Carefully examine the **Chart** and the **Query**. \n\
2. Select the **index number(s)** of the tools that are **most necessary and helpful** for solving the task.\n\
3. Most tasks can be solved using **1–3 tools**. Within each type of tools, DO NOT select more than one.\n\
4. Output **only the selected tool indices** as a comma-separated list, enclosed in `<answer>` tags.\n\n"




def format_reward(completions, **kwargs):
    """
    Check if the output contains a valid <answer>...</answer> tag
    and that all selected tool IDs are within the valid range [0–19].
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content in completion_contents:
        tool_ids = extract_model_choice2(content)
        if tool_ids is None:
            rewards.append(0.0)
            continue
        rewards.append(0.0)

    return rewards






def save_json_data():
    """Save the collected data to a JSON file"""
    global collected_data
    output_path = "./tool_choices_chartqa.json"
    
    # Create a lock to prevent multiple processes from writing simultaneously
    lock_path = output_path + ".lock"
    with FileLock(lock_path):
        try:
            # If the file exists, read existing data to check for duplicates
            existing_data = []
            existing_pairs = set()
            
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                
                # Create a set of existing question-label pairs for quick lookup
                for item in existing_data:
                    if "query" in item and "label" in item:
                        pair_key = f"{item['query']}|{item['label']}"
                        existing_pairs.add(pair_key)
            
            # Filter out any items that already exist in the file
            new_items = []
            for item in collected_data:
                pair_key = f"{item['query']}|{item['label']}"
                if pair_key not in existing_pairs:
                    new_items.append(item)
                    existing_pairs.add(pair_key)  # Add to set to prevent duplicates within new data
            
            # Merge with existing data and write back to file
            merged_data = existing_data + new_items
            with open(output_path, 'w') as f:
                json.dump(merged_data, f, indent=2)
                
            # Clear the collected_data list since we've saved it
            
            collected_data = []
                
        except Exception as e:
            print(f"Error saving JSON data: {e}")




def accuracy_reward(completions, solution, image, problem, frozen_vl_model=None, **kwargs):

    rank = get_rank()




    contents = [completion[0]["content"] for completion in completions]
    rewards = []


    for content, sol, img, pro in zip(contents, solution, image, problem):
        global collected_data


        reward = 0.0

        tool_ids = extract_model_choice(content)
      
        record = {
            "query": pro,
            "label": sol,
            "tool_choice": tool_ids if tool_ids is not None else [],
        }
        
        # Add the record to our collection
        collected_data.append(record)
        
        # Save the collected data periodically (e.g., every 10 examples)
        if len(collected_data) % 10 == 0:
            save_json_data()


        rewards.append(0.0)
        continue


    return rewards 





class GPTAPI:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.client = model

    

def get_gpt_api(api_key=None, model="gpt-4o-mini"):

    _gpt_api = GPTAPI(api_key=api_key, model=model)
    return _gpt_api





with open("./chartqa_train_mapping.json") as f:
    mapping = json.load(f)


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


def make_conversation_image(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                ],
            },
        ],
    }



def modify_dataset_format(example):
    return {
        "image": example["image"],
        "problem": example["query"],
        "solution": example["label"][0] if isinstance(example["label"], list) else example["label"]
    }


def main(script_args, training_args, model_args):
    # Load the dataset


    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Initialize the frozen VL model
    gpt_api = get_gpt_api() 
    
    # Initialize reward functions with the frozen model
    reward_funcs = []
    for func_name in script_args.reward_funcs:
        base_func = reward_funcs_registry[func_name]
        
        if func_name == "accuracy":
            def create_wrapper(current_base_func):
                def wrapped_func(*args, **kwargs):
                    # Add the frozen model as an additional parameter
                    kwargs["frozen_vl_model"] = gpt_api
                    return current_base_func(*args, **kwargs)
                return wrapped_func
            
            reward_funcs.append(create_wrapper(base_func))
        else:
            reward_funcs.append(base_func)




    # # Process the dataset
    if "image" in dataset[script_args.dataset_test_split].features:
        print("has image in dataset")

        dataset = dataset.map(modify_dataset_format)
        dataset = dataset.map(make_conversation_image)
        dataset = dataset.remove_columns(["human_or_machine", "query", "label"])
    



    # Initialize the trainer
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_test_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)