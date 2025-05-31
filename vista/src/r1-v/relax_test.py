import os
import json
import shutil
import re
import pandas as pd


def clean_answer_for_number_comparison(label, answer):

    if answer is None:
        return answer

    
    if str(label).replace(".", "").isdigit():

        if ":" in str(answer):
            return answer
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

        cleaned_answer = ''.join(result).strip()
        return cleaned_answer
    return answer


def compare_pred_label(pred, label):
    if pred is None:
        return False
    label_list = re.sub(r'[\[\]]', '', label).split(', ')
    pred_list = [item.strip() for item in re.split(r' and |, ', str(pred))]

    label_list.sort()
    pred_list.sort()

    return label_list == pred_list


def _to_float(text):
    if text is None:
        return None
    text = str(text)
    if text.endswith("%"):
        return float(text.rstrip('%'))
    else:
        return float(text)


def _to_float_percent_divided(text):
    """
    Convert text to float, if text ends with %, remove % and divide by 100
    """
    if text is None:
        return None
    text = str(text)
    if text.endswith("%"):
        return float(text.rstrip('%')) / 100
    else:
        return float(text)


def calculate_ratio(ratio_string):
    if ratio_string is None:
        return None
    numerator, denominator = ratio_string.split(':')
    numerator = float(numerator)
    denominator = float(denominator)
    result = numerator / denominator
    return result

def remove_commas_from_numbers(answer):
    if answer is None:
        return None

    return re.sub(r'(?<=\d),(?=\d)', '', answer)

def extract_numbers_and_dots(input_string):
    if input_string is None:
        return None
    numbers_and_dots = re.findall(r'[\d.]+', str(input_string))
    return ''.join(numbers_and_dots)


def get_lrstrip(text):
    if text is None:
        return None
    text = str(text)
    text = text.rstrip()
    text = text.lstrip()
    return text.lower()


def get_prediction(pred, label):

    pred = get_lrstrip(pred)
    label = get_lrstrip(label)
    
    if pred is None:
        return False
    
    pred = clean_answer_for_number_comparison(label, pred)
    pred = remove_commas_from_numbers(pred)

    if pred == label:
        return True
    

    if ":" in pred:
        parts = pred.split(":")


        try:
            if len(parts) == 2:

                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator != 0:
                    ratio_value = numerator / denominator
                    label_value = float(label)
                    if abs(ratio_value - label_value) <= 0.05 * max(1, abs(label_value)):
                        return True
        except (ValueError, ZeroDivisionError):
            pass
    
    if label.endswith("]"):
        return compare_pred_label(pred, label)
    
   
    try:
        
        label_float = float(label) if not label.endswith("%") else float(label.rstrip("%")) / 100
        
      
        if label_float < 1 and "%" in pred:
            try:
                pred_decimal = float(pred.rstrip("%")) / 100
                if abs(pred_decimal - label_float) <= 0.05 * max(1, abs(label_float)):
                    return True
            except:
                pass
        
        
        if label.endswith("%") and "%" not in pred:
            try:
                label_decimal = float(label.rstrip("%")) / 100
                pred_float = float(pred)
                if abs(pred_float - label_decimal) <= 0.05 * max(1, abs(label_decimal)):
                    return True
            except:
                pass
        
      
        try:
            pred_float = float(pred.rstrip("%")) if pred.endswith("%") else float(pred)
            if abs(pred_float - label_float) <= 0.05 * max(1, abs(label_float)):
                return True
        except:
            pass
        
      
        try:
            if pred.endswith("%"):
                pred_float = float(pred.rstrip("%")) / 100
            else:
                pred_float = float(pred) / 100
                
            if abs(pred_float - label_float) <= 0.05 * max(1, abs(label_float)):
                return True
        except:
            pass
        
    
        try:
            pred_float = extract_numbers_and_dots(pred)
            if pred_float:
                pred_float = float(pred_float)
                return (abs(pred_float - label_float) <= 0.05 * max(1, abs(label_float)) or 
                        abs(pred_float/100 - label_float) <= 0.05 * max(1, abs(label_float)))
        except:
            pass
        
        return False
        
    except Exception as e:
 
        return pred == label



def extract_answer_from_generated_answer(generated_answer):


    
    if isinstance(generated_answer, str):
    
        match = re.search(r'<answer>(.*?)</answer>', generated_answer, re.DOTALL)
        if match:
            return match.group(1).strip()

      
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", generated_answer, re.DOTALL)
        if match:
            try:
                parsed_json = json.loads(match.group(1))
                if "answer" in parsed_json:
                    answer = parsed_json["answer"]
                    if isinstance(answer, str) and "<answer>" in answer:
                        inner_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
                        return inner_match.group(1).strip() if inner_match else answer.strip()
                    return str(answer).strip()
            except:
                pass

        
        try:
            parsed_json = json.loads(generated_answer)
            if "answer" in parsed_json:
                answer = parsed_json["answer"]
                if isinstance(answer, str) and "<answer>" in answer:
                    inner_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
                    return inner_match.group(1).strip() if inner_match else answer.strip()
                return str(answer).strip()
        except:
            pass

        # Fallback to direct string
        return generated_answer.strip()


    return None



def calculate_accuracy_and_collect_cases(data, correct_folder=None, error_folder=None, image_folder=None,
                                         correct_json_path=None, error_json_path=None):
    correct_count = 0
    total_count = 0
    error_images = []
    correct_images = []
    error_entries = []
    correct_entries = []

    for entry in data:
        label = entry['label']
        generated_answer = entry['generated_answer']
        image_filename = entry.get('imgname', None)

        extracted_answer = extract_answer_from_generated_answer(generated_answer)
        # extracted_answer = remove_commas_from_numbers(extracted_answer)

        # Using the modified comparison logic
        is_correct = get_prediction(extracted_answer, label)
        entry['extracted_answer'] = extracted_answer
        entry['is_correct'] = is_correct
        entry.pop('generated_answer', None)

        if is_correct:
            correct_count += 1
            if image_filename:
                correct_images.append(image_filename)
            correct_entries.append(entry)
        else:
            if image_filename:
                error_images.append(image_filename)
            error_entries.append(entry)

        total_count += 1

    if image_folder:
        if correct_folder:
            os.makedirs(correct_folder, exist_ok=True)
            # for image in correct_images:
            #     source_path = os.path.join(image_folder, image)
            #     destination_path = os.path.join(correct_folder, image)
            #     if os.path.exists(source_path):
            #         shutil.copy(source_path, destination_path)

        if error_folder:
            os.makedirs(error_folder, exist_ok=True)
            # for image in error_images:
            #     source_path = os.path.join(image_folder, image)
            #     destination_path = os.path.join(error_folder, image)
            #     if os.path.exists(source_path):
            #         shutil.copy(source_path, destination_path)

    if correct_json_path:
        with open(correct_json_path, 'w') as correct_file:
            json.dump(correct_entries, correct_file, indent=4)

    if error_json_path:
        with open(error_json_path, 'w') as error_file:
            json.dump(error_entries, error_file, indent=4)

    accuracy = (correct_count / total_count) * 100
    return accuracy, correct_images, error_images


def main():
    folder = './result_gpt4o_chartqa_human'
    json_file = './result_gpt4o_chartqa_human.json'
    print(json_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

    image_folder = '/path/ChartQA/ChartQA_Dataset/test/png'
    correct_folder = f'{folder}/correct_cases'
    error_folder = f'{folder}/error_cases'
    correct_json_path = f'{folder}/correct_cases.json'
    error_json_path = f'{folder}/error_cases.json'

    accuracy, correct_images, error_images = calculate_accuracy_and_collect_cases(
        data,
        correct_folder,
        error_folder,
        image_folder,
        correct_json_path,
        error_json_path
    )

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Number of correct images: {len(correct_images)}')
    print(f'Number of incorrect images: {len(error_images)}')


if __name__ == "__main__":
    main()