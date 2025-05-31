import os
import json
import re
from tqdm import tqdm
from PIL import Image
import base64
from openai import OpenAI
import pandas as pd
import requests
import random
import torch



OPENAI_DEPLOYMENTS = [
    (
        'endpoints',
        'key')
]

IMAGE_FOLDER = '/path/ChartQA/ChartQA_Dataset/test/png'



TOOL_CHOICES_FILE = './tool_choices_chartqa.json'


    tool_paths = {
        0: f"./tools_test/table_unichart",
        1: f"./tools_test/table_deplot",
        2: f"./tools_test/table_gt",
        3: f"./tools_test/svg1", 
        4: f"./tools_test/svg2",
        5: f"./tools_test/svg3",      
        6: f"./tools_test/caption_qwen3b",
        7: f"./tools_test/caption_qwen7b",
        8: f"./tools_test/caption_qwen72b"
    }


TOOL_TYPES = {
    0: "Table Version of Chart",
    1: "Table Version of Chart",
    2: "Table Version of Chart",
    3: "SVG Representation",
    4: "SVG Representation",
    5: "SVG Representation",
    6: "Chart Caption",
    7: "Chart Caption",
    8: "Chart Caption"
}

RESULT_FILE = 'result_gpt4o_chartqa_human.json'

class QwenChartReasoner:
    def __init__(self):

        self.tool_choices_map = self.load_tool_choices()

    def load_tool_choices(self):
        """Load tool choices from JSON file and create a mapping of (query, label) -> tool_choice"""
        tool_choices_map = {}
        try:
            with open(TOOL_CHOICES_FILE, 'r') as f:
                tool_choices_data = json.load(f)
                for item in tool_choices_data:
                    key = (item['query'], item['label'])
                    tool_choices_map[key] = item['tool_choice']
            print(f"Loaded {len(tool_choices_map)} tool choices")
            return tool_choices_map
        except Exception as e:
            print(f"Error loading tool choices file: {e}")
            return {}

    def get_tool_choice(self, query, label):
        """Get the tool choice for a given query and label pair"""
        return self.tool_choices_map.get((query, label), [])

    def local_image_to_data_url(self, image_path):
        """Convert a local image to base64 data URL"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
    
    def read_json_file(self, path):
        """Read a JSON file and return its contents"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to read JSON at {path}: {e}")
            return None
            
    def read_csv_file(self, path):
        """Read a CSV file and return its contents as dict"""
        try:
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Failed to read CSV at {path}: {e}")
            return None

    def get_chart_table_data(self, imgname, query, label):
        """Get chart table data using file paths based on imgname and tool choices"""
        tool_ids = self.get_tool_choice(query, label)

        # tool_ids = []
        if not tool_ids:
            print(f"No tool choices found for query: {query}, label: {label}")
            return {
                "Table Version of Chart": [],
                "SVG Representation": [],
                "Chart Caption": []
            }

        chart_data_by_type = {
            "Table Version of Chart": [],
            "SVG Representation": [],
            "Chart Caption": []
        }

        chart_table_data = []
        
        # For each selected tool, get the corresponding data
        for tid in tool_ids:
            if tid not in TOOL_PATHS:
                print(f"Tool ID {tid} not found in TOOL_PATHS")
                continue

            tool_type = TOOL_TYPES[tid]


            if tid == 2:
                csv_name = imgname.replace(".png", ".csv")
                csv_path = os.path.join(TOOL_PATHS[tid], csv_name)

                if os.path.exists(csv_path):
                    data = self.read_csv_file(csv_path)
                    if data is not None:
                        chart_table_data.append(data)
                        chart_data_by_type[tool_type].append(data)
                else:
                    print(f"CSV file not found: {csv_path}")
                continue

    
            json_name = imgname.replace(".png", ".json")
            json_path = os.path.join(TOOL_PATHS[tid], json_name)
            
            if os.path.exists(json_path):
                data = self.read_json_file(json_path)
                if data is not None:
                    chart_table_data.append(data)
                    chart_data_by_type[tool_type].append(data)
            else:
                print(f"JSON file not found: {json_path}")

        return chart_data_by_type

    def analyze_chart(self, image_path, imgname, query, label):
        """Analyze the chart using Qwen model to answer the question"""


        endpoints = random.choice(OPENAI_DEPLOYMENTS)
        url = f'https://{endpoints[0]}.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview'
        api_key = endpoints[1]

        chart_data_by_type = self.get_chart_table_data(imgname, query, label)
            

        formatted_data = ""
        
        for data_type, data_list in chart_data_by_type.items():
            if data_list:
                formatted_data += f"\n\n{data_type}:\n"
                formatted_data += "\n\n".join([json.dumps(d, indent=2) for d in data_list])
        

        data_url = self.local_image_to_data_url(image_path)





        prompt = f'''
        The first image is one of a bar chart, line chart, or pie chart.
        It comes with instructions, predicted information, a question, and an answer. 
        Based on the image and its predicted information, the question must be solved.
        
        Your response should contain four fields and four fields only: 
        "instruction explanation": an explanation the process of following instructions according to the chart type
        "explanation": an explanation of how you arrived at the answer
        "answer": the answer based on the chart and question
        Output your final answer in <answer> </answer> tags
    
        Instruction for whole chart: 
        1. If the question is about values corresponding to specific positions (ex: lower, last, top), then you must match the information with the chart image's positions for reasoning. 
        2. If the question demands interpretation based on the magnitude of values, reasoning should be based on the information's values. 
        3. Originally, this task requires solving based solely on the image, meaning all positions should be interpreted based on the image itself.
        4. In most cases, the presence of x-axis values or y-axis values enables the determination of the chart's values.
        5. Note that, you can utilize the predicted information. The predicted columns and rows are very likely to correspond to the actual columns and rows of the chart, and this can help you determine where the rows and columns exist in the chart image.
        
        Instruction for bar chart:
        1. Firstly, bars of the same color represent the same column. Therefore, distinguishing colors and identifying corresponding columns is crucial (usually displayed around the main chart in the form of a legend).
        2. Next, determine the location of rows. For vertical bar charts, rows are typically annotated at the bottom of the main chart, while for horizontal bar charts, they are annotated on the left or right side of the main chart.
        3. Then, combine the colors of the nearest bars with annotated rows to determine which row and column the bars correspond to in the information.
        4. Afterwards, locate the values corresponding to each row and column. If values are annotated on the bars, refer to them. Otherwise, compare the sizes of the bars to find the values.
        5. For vertical bar charts, the y-axis value at the end of the bar corresponds to its value. Similarly, for horizontal bar charts, the x-axis value at the end of the bar corresponds to its value.
        
        Instruction for line chart: 
        1. In the case of a line chart, the bottom x-axis will primarily represent the rows, and each colored line will represent a column.
        2. The legend, which indicates which column corresponds to the color of the line, is usually located within the main chart. If the legend is absent or located separately, the text annotated with the color corresponding to the line will likely indicate the column (if colors are not present, the text annotated at the left or right end of the line is likely to correspond to the column).
        3. The point of the line passing through the same x coordinates as each x-axis is the value itself (meaning the x-axis corresponds to the row, the color of the line corresponds to the column, and that point is the value).
        4. If there is an annotation near a line point, it is highly likely that this value represents the value of the point.
        5. If there is no annotation near a line point, you can determine the value of the point by referring to the y-axis value corresponding to the y coordinate of the point.
        6. In a line chart, it is crucial to understand the flow of the line. Lines can show trends of decreasing, increasing, or remaining constant, and when multiple lines intersect, it is important to identify which line corresponds to which column based on their colors.
        
        Instruction for pie chart:
        1. In a pie chart, it is very important to determine which color corresponds to which row.
        2. Each section has one color, and the row it corresponds to is likely indicated by text either inside the section or close to it (if not nearby, it can be identified through the legend or connected to the corresponding text by lines or markers).
        3. In the case of a pie chart, the values are usually annotated on each section of the pie chart.
        
        Now you have to answer the question based on the first image, predicted information and question.
        
        Please follow the below rules:
        1. If question ask a yes/no, then answer should be "Yes" or "No". You must not answer with "true" or "false".
        2. At 'answer' field, you must only answer the question. You must not include any other information.
    
        Chart image is first image.
        
        Here is predicted information:
        {formatted_data}
    
        Here is a question:
        {query}
        '''



        
        response = requests.post(
            url,
            headers={'Content-Type': 'application/json', 'api-key': api_key},
            json={
                "messages": [
                    {"role": "user", "content":
                        [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                "max_tokens": 400,
                "temperature": 0,
                "seed": 1024
            }
        )
        response_json = response.json()
        if 'error' in response_json:
            print(f"Error: {response_json['error']}")
            return None
        else:
            return response_json['choices'][0]['message']['content']

def save_results(results):
    """Save results to a JSON file"""
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    
    existing_data.extend(results)
    
    with open(RESULT_FILE, 'w') as f:
        json.dump(existing_data, f, indent=4)

def is_question_already_processed(imgname, query, existing_results):
    """Check if a question has already been processed"""
    for result in existing_results:
        if result['imgname'] == imgname and result['query'] == query:
            return True
    return False

def main():

    with open('/path/ChartQA/ChartQA_Dataset/test/test_human.json', 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)

    qwen_reasoner = QwenChartReasoner()
    

    results = []
    

    existing_results = []
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r') as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
    

    for entry in tqdm(data, desc="Processing", unit="sample"):
        imgname = entry['imgname']
        query = entry['query']
        label = entry['label']
        

        if is_question_already_processed(imgname, query, existing_results):
            continue
        

        img_path = os.path.join(IMAGE_FOLDER, imgname)
        if not os.path.exists(img_path):
            print(f"Image file {imgname} does not exist in the folder {IMAGE_FOLDER}. Skipping...")
            continue
        

        generated_answer = qwen_reasoner.analyze_chart(img_path, imgname, query, label)
        if generated_answer is None:
            continue
        

        tool_choices = qwen_reasoner.get_tool_choice(query, label)
        

        results.append({
            'imgname': imgname,
            'query': query,
            'label': label,
            'generated_answer': generated_answer,
            'tool_choices': tool_choices
        })
        
        if len(results) >= 10:
            save_results(results)
            results = []

    if results:
        save_results(results)

if __name__ == "__main__":
    main()