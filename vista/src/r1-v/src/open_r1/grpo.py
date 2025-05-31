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



reward_stat_counter = {"total": 0, "reward_00": 0, "reward_01": 0, "reward_10": 0, "reward_11": 0}

tools_counter = {"total": 0, "tool 0": 0, "tool 1": 0, "tool 2": 0, "tool 3": 0, "tool 4": 0, "tool 5": 0, "tool 6": 0, "tool 7": 0, "tool 8": 0, "no tool": 0}




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

QUESTION_TEMPLATE = "You are an expert agent specialized in selecting tools to solve **chart reasoning tasks**. You are provided with access to **9 tools**, indexed from **0 to 8**. Each tool belongs to one of three function types:- **type1**- **type2**- **type3**Each tool is implemented differently, even if they share the same type. Treat all tools as **independent**.\n\n\
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
1. Carefully examine the **Chart** and the **Query**. Select the **index number(s)** of the tools that are **most helpful** for solving the task\n\
2. Output **only the selected tool indices** as a comma-separated list, enclosed in `<answer>` tags.\n\n"



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




def read_tool_output(path, index):
    """
    Try reading tool output as JSON or SVG.
    Return either dict (parsed JSON) or str (raw SVG content).
    """
    json_path = os.path.join(path, f"{index}.json")
    svg_path = os.path.join(path, f"{index}.svg")

    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                return json.load(f)
        except Exception:
            print("no json")
            return None

    return None



def get_imgname_and_table(query, label, root):
    meta_files = [
        "./train_augmented.json",
        "./train_human.json"
    ]
    table_dir = root

    matches = []

    for meta_path in meta_files:
        with open(meta_path, "r", encoding='utf-8') as f:
            entries = json.load(f)
            for entry in entries:
                if entry["query"] == query and entry["label"] == label:
                    matches.append(entry)

 
    if len(matches) != 1:
        return None


    imgname = matches[0]["imgname"]
    csv_name = imgname.replace(".png", ".csv")
    csv_path = os.path.join(table_dir, csv_name)

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Failed to read table: {csv_path}, error: {e}")
            return None
    else:
        return None


def svg_gt_get_imgname_and_table(query, label, root):
    meta_files = [
        "./train_augmented.json",
        "./train_human.json"
    ]
    svg_dir = root

    matches = []

    for meta_path in meta_files:
        with open(meta_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
            for entry in entries:
                if entry["query"] == query and entry["label"] == label:
                    matches.append(entry)


    if len(matches) != 1:
        return None


    imgname = matches[0]["imgname"]
    svg_name = imgname.replace(".png", ".json")
    svg_path = os.path.join(svg_dir, svg_name)

    if os.path.exists(svg_path):
        try:
            with open(svg_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to read table: {svg_path}, error: {e}")
            return None
    else:
        return None
    


def svg_gt_get_imgname_and_table_remove_value(query, label, root):
    meta_files = [
        "./train_augmented.json",
        "./train_human.json"
    ]
    svg_dir = root

    matches = []

    for meta_path in meta_files:
        with open(meta_path, "r") as f:
            entries = json.load(f)
            for entry in entries:
                if entry["query"] == query and entry["label"] == label:
                    matches.append(entry)


    if len(matches) != 1:
        return None

 
    imgname = matches[0]["imgname"]
    svg_name = imgname.replace(".png", ".json")
    svg_path = os.path.join(svg_dir, svg_name)

    if os.path.exists(svg_path):
        try:
            with open(svg_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to read table: {svg_path}, error: {e}")
            return None
    else:
        return None
    


def shrink_and_pad_random(img: Image.Image) -> Image.Image:
    """
    Args:
        img (Image.Image): Input image in RGBA mode.

    Returns:
        Image.Image: Transformed image with padding.
    """
    assert img.mode == 'RGBA', "Image must be in RGBA mode"

    original_width, original_height = img.size
    white = (255, 255, 255, 255)

    direction = random.choice(['horizontal', 'vertical'])

    if direction == 'horizontal':
        # Resize width to half
        resized_img = img.resize((original_width // 2, original_height), Image.Resampling.LANCZOS)
        new_img = Image.new('RGBA', (original_width, original_height), white)
        position = random.choice(['left', 'right'])
        if position == 'left':
            new_img.paste(resized_img, (0, 0))
        else:  # right
            new_img.paste(resized_img, (original_width // 2, 0))
    else:  # vertical
        # Resize height to half
        resized_img = img.resize((original_width, original_height // 2), Image.Resampling.LANCZOS)
        new_img = Image.new('RGBA', (original_width, original_height), white)
        position = random.choice(['top', 'bottom'])
        if position == 'top':
            new_img.paste(resized_img, (0, 0))
        else:  # bottom
            new_img.paste(resized_img, (0, original_height // 2))

    return new_img





def accuracy_reward(completions, solution, image, problem, frozen_vl_model=None, **kwargs):

    rank = get_rank()

    tool_paths = {
        0: f"./tools/table1",  # run https://github.com/google-research/google-research/tree/master/deplot and save results on ./tools/table1
        1: f"./tools/table2",  # run https://github.com/vis-nlp/UniChart and save results on ./tools/table2
        2: f"./tools/table3",  # run https://github.com/IDEA-FinAI/ChartMoE and save results on ./tools/table3
        3: f"./tools/svg1",    # already saved on ./tools/svg1
        4: f"./tools/svg2",    # run https://github.com/pengyu965/ChartDete and save results on ./tools/svg2
        5: f"./tools/svg3",    # run https://github.com/soap117/DeepRule and save results on ./tools/svg3
        6: f"./tools/caption1", # run https://github.com/OpenGVLab/ChartAst and save results on ./tools/caption1
        7: f"./tools/caption2", # run https://github.com/Alpha-Innovator/ChartVLM and save results on ./tools/caption2
        8: f"./tools/caption3"  # run https://github.com/QwenLM/Qwen2.5-VL and save results on ./tools/caption3
    }

    contents = [completion[0]["content"] for completion in completions]
    rewards = []


    for content, sol, img, pro in zip(contents, solution, image, problem):
        global reward_stat_counter
        global tools_counter


        reward_stat_counter["total"] += 1
        tools_counter["total"] += 1
        reward = 0.0


        img = img.convert('RGBA')

        img = shrink_and_pad_random(img)

        
        img = img.convert("RGB")
        tool_ids = extract_model_choice(content)
      

        if tool_ids is None or len(tool_ids) == 0:
            tools_counter["no tool"] += 1
            tool_ids = []

        # print(tool_ids)
        
        for tid in tool_ids:
            if 0 <= tid <= 8:
                tool_name = f"tool {tid}"
                tools_counter[tool_name] += 1
        


        index_candidates = [k for k, v in mapping.items() if v["query"] == pro and v["label"] in sol]



        if len(index_candidates) != 1:
            reward_stat_counter["reward_0"] += 1
            rewards.append(0.0)
            continue
        index = index_candidates[0]

        chart_table_data = []



        for tid in tool_ids:
            if tid not in tool_paths:
                continue

            if tid == 2:
                path = tool_paths[tid]
                table_df = get_imgname_and_table(pro, sol, path) 
                if table_df is not None:
                    chart_table_data.append(table_df.to_dict(orient="records"))

                continue  

            if tid == 4:
                path = tool_paths[tid]
                svg_df_r = svg_gt_get_imgname_and_table_remove_value(pro, sol, path) 
                if svg_df_r is not None:
                    chart_table_data.append(svg_df_r)

                continue  

            if tid == 5:
                path = tool_paths[tid]
                svg_df = svg_gt_get_imgname_and_table(pro, sol, path) 
                if svg_df is not None:
                    chart_table_data.append(svg_df)

                continue  



            path = tool_paths[tid]

            data = read_tool_output(path, index)
            if data is None:
                continue
                    
            if 0 <= tid <= 8:
                chart_table_data.append(data)


        chart_table_info = "\n\n".join([json.dumps(d) for d in chart_table_data]) if chart_table_data else ""

        reward_case = ""
        try:

            response1 = frozen_vl_model.generate_answer2(
                img,
                pro,
                chart_table_info=chart_table_info,
            )


            response2 = frozen_vl_model.generate_answer(
                img,
                pro,
                chart_table_info=chart_table_info,
            )


            ans1 = extract_answer(response1)
            ans2 = extract_answer(response2)
            

            correct1 = get_prediction(ans1, sol)
            correct2 = get_prediction(ans2, sol)
            

            
            if correct1 and correct2:
                reward = 1.0
                reward_case = "11"
            elif correct1 and not correct2:
                reward = -0.5
                reward_case = "10"
            elif not correct1 and correct2:
                reward = 1.0
                reward_case = "01"
            else:  # both incorrect
                reward = 0.0
                reward_case = "00"





        except Exception:
            print("API error")
            reward = 0.0

        if reward_case == "01":
            reward_stat_counter["reward_01"] += 1
        elif reward_case == "10":
            reward_stat_counter["reward_10"] += 1
        elif reward_case == "11":
            reward_stat_counter["reward_11"] += 1
        elif reward_case == "00":
            reward_stat_counter["reward_00"] += 1           

        stats_line = f"[Stats] Processed: {reward_stat_counter['total']} | reward=01: {reward_stat_counter['reward_01']} | reward=11: {reward_stat_counter['reward_11']} | reward=10: {reward_stat_counter['reward_10']} | reward=00: {reward_stat_counter['reward_00']}"
        tool_line = "[Tool Usage] " + " | ".join(f"{tool_name}: {tools_counter[tool_name]}" for tool_name in sorted(tools_counter.keys()))

        with open(f"./results/output_table_Qwen2-VL-7B_chartqa/qwen2_reward_stats_filter_data_01_rank{rank}.txt", "a") as f:
            f.write(stats_line + "\n")

        with open(f"./results/output_table_Qwen2-VL-7B_chartqa/qwen2_tool_id_count_filter_data_01_rank{rank}.txt", "a") as f:
            f.write(tool_line + "\n")



        rewards.append(reward)


    return rewards 





class GPTAPI:
    def __init__(self, api_key=None, model="gpt-4o-mini"):



        self.client = OpenAI(
                   base_url="http://10.0.16.10:8000/v1",
                   api_key="token-abc123",
                   )



    def get_client(self):
        
        base_url = random.choice(self.base_urls)
        return OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )
    

    def encode_image(self, image):
       
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    
    
    def generate_answer(self, image, question, chart_table_info="", max_new_tokens=512):
 
        

        base64_image = self.encode_image(image)
 
        prompt = f"Question: {question}\n\nExtra information from the chart:\n{chart_table_info}\n\nAnswer the question based on the chart and extra information. Output your final answer in <answer></answer> tags."
        
        # print(prompt)

        response = self.client.chat.completions.create(
          model="/qwen2.5-vl-7b-instruct",
          messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt,
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                  },
                },
              ],
            }
          ],
        )
        
        return response.choices[0].message.content


    def generate_answer2(self, image, question, chart_table_info="", max_new_tokens=512):



        base64_image = self.encode_image(image)

        response2 = self.client.chat.completions.create(
          model="/qwen2.5-vl-7b-instruct",
          messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": f"Question: {question}\n\nAnswer the question based on the chart. Output your final answer in <answer></answer> tags.",
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                  },
                },
              ],
            }
          ],
        )

        
        return response2.choices[0].message.content
    



def get_gpt_api(api_key=None, model="gpt-4o-mini"):

    _gpt_api = GPTAPI(api_key=api_key, model=model)
    return _gpt_api



def clean_answer_for_number_comparison(label, answer):
    """
    Clean answer for number comparison when label is purely numeric.
    Only keeps valid decimal points (must be followed by digits).
    """
    if answer is None:
        return answer

    if str(label).replace(".", "").isdigit():
        temp_answer = ''.join(char for char in str(answer) if char.isdigit() or char == '.')
        result = []
        i = 0
        while i < len(temp_answer):
            if temp_answer[i] == '.':
                if i == len(temp_answer) - 1 or not temp_answer[i + 1].isdigit():
                    i += 1
                    continue
            result.append(temp_answer[i])
            i += 1
        return ''.join(result).strip()
    return answer


def get_lrstrip(text):
    """Strip whitespace and convert to lowercase."""
    if text is None:
        return None
    return str(text).strip().lower()


def _to_float(text):
    """Convert text to float, handling percentage format."""
    if text is None:
        return None
    text = str(text)
    if text.endswith("%"):
        return float(text.rstrip('%'))
    return float(text)


def calculate_ratio(ratio_string):
    """Calculate numeric ratio from string format (e.g., '3:4')."""
    if ratio_string is None:
        return None
    numerator, denominator = ratio_string.split(':')
    return float(numerator) / float(denominator)


def extract_numbers_and_dots(input_string):
    """Extract all numbers and decimal points from string."""
    if input_string is None:
        return None
    numbers_and_dots = re.findall(r'[\d.]+', str(input_string))
    return ''.join(numbers_and_dots)


def compare_pred_label(pred, label):
    """Compare prediction and label for list-type answers."""
    if pred is None:
        return False
    label_list = re.sub(r'[\[\]]', '', label).split(', ')
    pred_list = [item.strip() for item in re.split(r' and |, ', str(pred))]
    label_list.sort()
    pred_list.sort()
    return label_list == pred_list


def get_prediction(pred, label):
    """
    Compare prediction and label with relaxed matching rules.
    Handles various formats and allows for small numerical differences.
    """
    pred = get_lrstrip(pred)
    label = get_lrstrip(label)

    if pred is None:
        return False

    pred = clean_answer_for_number_comparison(label, pred)

    if label.endswith("]"):
        return compare_pred_label(pred, label)

    try:
        label_float = _to_float(label)
        try:
            pred_float = _to_float(pred)
            if pred_float is None:
                return False
            relative_change = abs(pred_float - label_float) / abs(label_float)
            return relative_change <= 0.05
        except:
            try:
                pred_float = calculate_ratio(pred)
                if pred_float is None:
                    return False
                relative_change = abs(pred_float - label_float) / abs(label_float)
                return relative_change <= 0.05
            except:
                pred_float = extract_numbers_and_dots(pred)
                if pred_float == '':
                    return False
                pred_float = _to_float(pred_float)
                relative_change = abs(pred_float - label_float) / abs(label_float)
                return relative_change <= 0.05
    except:
        return pred == label



def extract_answer(content):
    """Extract answer from content, handling various formats."""
    if 'assistant' in content:
        idx = content.index('assistant') + len('assistant')
        subcontent = content[idx:]
        match = re.search(r'<answer>(.*?)</answer>', subcontent)
    else:
        match = re.search(r'<answer>(.*?)</answer>', content)
    return match.group(1).strip() if match else content.strip()



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
    if "image" in dataset[script_args.dataset_train_split].features:
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
        train_dataset=dataset[script_args.dataset_train_split],
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