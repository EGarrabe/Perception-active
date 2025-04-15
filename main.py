import os
import sys
import csv
import time
from collections import defaultdict
import ollama



# check if certain words are in the response (ie : check if the model detected the correct object)
def evaluate_model_response(response, correct_names):
    response_lower = response.lower()
    return any(name.lower() in response_lower for name in correct_names)

def load_existing_results(results_path):
    existing = set()
    if os.path.exists(results_path):
        with open(results_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['prompt'], row['model'], row['image_folder'], row['image_name'])
                existing.add(key)
    return existing

def load_accuracy_matrix(matrix_path):
    accuracy = defaultdict(lambda: defaultdict(int))
    if os.path.exists(matrix_path):
        with open(matrix_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)[1:]  # skip first column (prompts)
            for row in reader:
                prompt = row[0]
                for i, val in enumerate(row[1:]):
                    accuracy[prompt][headers[i]] = int(val)
    return accuracy

def save_accuracy_matrix(matrix_path, accuracy, prompts, models):
    existing_data = {}
    if os.path.exists(matrix_path):
        with open(matrix_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)[1:]
            for row in reader:
                existing_data[row[0]] = dict(zip(headers, map(int, row[1:])))

    # Merge in new values
    for prompt in accuracy:
        if prompt not in existing_data:
            existing_data[prompt] = {}
        for model in accuracy[prompt]:
            existing_data[prompt][model] = accuracy[prompt][model]

    # Save updated matrix
    all_prompts = sorted(existing_data.keys())
    all_models = sorted(set(model for row in existing_data.values() for model in row))

    with open(matrix_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt'] + all_models)
        for prompt in all_prompts:
            row = [prompt] + [existing_data[prompt].get(model, 0) for model in all_models]
            writer.writerow(row)

def append_to_image_matrix(matrix_path, image_id, model_names, prompts, data_dict):
    existing = {}

    # Load existing if it exists
    if os.path.exists(matrix_path):
        with open(matrix_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)[1:]
            for row in reader:
                prompt = row[0]
                existing[prompt] = dict(zip(headers, row[1:]))

    # Update or insert new data
    for prompt in data_dict:
        if prompt not in existing:
            existing[prompt] = {}
        for model in data_dict[prompt]:
            existing[prompt][model] = data_dict[prompt][model]

    # Write it back
    all_prompts = sorted(existing.keys())
    with open(matrix_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([f"{image_id}"] + model_names)
        for prompt in all_prompts:
            row = [prompt] + [existing[prompt].get(model, "") for model in model_names]
            writer.writerow(row)


def format_time(seconds):
    return f"{int(seconds//60)} min {int(seconds%60)} sec"

def run_evaluation(model_names, prompts, image_paths, images_object,
                   results_path="output/responses.csv", accuracy_folder="output", matrix_folder="output/image_matrices"):

    os.makedirs(accuracy_folder, exist_ok=True)
    os.makedirs(matrix_folder, exist_ok=True)

    existing = load_existing_results(results_path)

    results_fields = ["prompt", "model", "image_folder", "image_name", "model_response",
                      "correct_object_names", "success", "response_time"]
    if not os.path.exists(results_path):
        with open(results_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results_fields)
            writer.writeheader()

    total_prompts = len(prompts)
    global_start_time = time.time()

    for image_folder in image_paths:
        folder_name = os.path.basename(image_folder)
        accuracy_path = os.path.join(accuracy_folder, f"accuracy.csv")
        accuracy_matrix = load_accuracy_matrix(accuracy_path)
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
        total_images = len(image_files)

        # Initialize per-image matrix storage
        per_image_data = {img: defaultdict(dict) for img in image_files}

        for prompt_idx, prompt in enumerate(prompts, 1):
            prompt_start_time = time.time()
            model_results = {}
            model_progress_lines = {}

            for model in model_names:
                correct = 0
                incorrect = 0
                status_line = ["_" for _ in image_files]

                for i, image_name in enumerate(image_files):
                    key = (prompt, model, image_folder, image_name)
                    if key in existing:
                        continue

                    image_path = os.path.join(image_folder, image_name)
                    correct_names = images_object.get(image_name, [])

                    try:
                        start = time.time()
                        response, response_time = single_call(model, prompt, image_path, 0)
                        success = evaluate_model_response(response, correct_names)
                    except Exception as e:
                        response = f"ERROR: {str(e)}"
                        response_time = 0
                        success = False

                    with open(results_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=results_fields)
                        writer.writerow({
                            "prompt": prompt,
                            "model": model,
                            "image_folder": folder_name,
                            "image_name": image_name,
                            "model_response": response,
                            "correct_object_names": "; ".join(correct_names),
                            "success": success,
                            "response_time": response_time
                        })

                    per_image_data[image_name][prompt][model] = success  # Save to per-image matrix

                    if success:
                        accuracy_matrix[prompt][model] += 1
                        correct += 1
                        status_line[i] = "\033[0;32m1\033[0m"
                    else:
                        incorrect += 1
                        status_line[i] = "\033[0;31m0\033[0m"

                    total = correct + incorrect
                    line = f"\033[0;36m{model}\033[0m | {''.join(status_line)} total : \033[0;32m{correct}\033[0m/\033[0;31m{incorrect}\033[0m/{total_images} | time : {format_time(time.time() - prompt_start_time)}"
                    sys.stdout.write("\r" + " " * 160 + "\r")
                    sys.stdout.write(line)
                    sys.stdout.flush()

                model_results[model] = accuracy_matrix[prompt][model]
                save_accuracy_matrix(accuracy_path, accuracy_matrix, prompts, model_names)

            sys.stdout.write("\r" + " " * 160 + "\r")
            prompt_summary = f"prompt {prompt_idx}/{total_prompts} | " + \
                             " ; ".join(f"\033[0;36m{m}\033[0m \033[0;32m{accuracy_matrix[prompt].get(m, 0)}\033[0m/{total_images}" for m in model_names) + \
                             f" | time : {format_time(time.time() - prompt_start_time)}"
            print(prompt_summary)

        # Save (append to) per-image matrix CSVs
        for image_name, prompt_data in per_image_data.items():
            first_object = images_object.get(image_name, ["unknown"])[0].replace(" ", "_")
            image_name_no_ext = os.path.splitext(image_name)[0]
            matrix_path = os.path.join(matrix_folder, f"{image_name_no_ext}_{first_object}.csv")
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True)

            append_to_image_matrix(matrix_path, f"{image_name_no_ext}_{first_object}", model_names, prompts, prompt_data)




# Dummy model function for testing
def query_model(model_name, prompt, image_path):
    # Simulate a dummy response with a fake delay
    time.sleep(0.1)
    return "This contains " + os.path.splitext(os.path.basename(image_path))[0], 0.1


def single_call(model_name, input_prompt, image_path, input_temperature):
    """
    Call a model with the prompt, image and temperature specified
    Returns the response and the generation time
    """
    start_time = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                'role': 'user',
                'content': input_prompt,
                'images': [image_path]
            },
        ],
        stream=False,
        options={'temperature': input_temperature}
    )
    time_taken = time.time() - start_time
    model_response = response['message']['content']
    return model_response, time_taken


def get_prompts(txt_path):
    """
    Get the prompts from a text file
    """
    with open(txt_path, 'r') as f:
        prompts = f.readlines()
    return [prompt.strip() for prompt in prompts]

model_names = ["llava", "llava:13b", "llava:34b", "llava-llama3", "llama3.2-vision", "gemma3:4b", "gemma3:12b", "gemma3:27b"]
# model_names = ["llava", "llava-llama3", "gemma3:4b"]
prompts = get_prompts("ressources/all_prompts.txt")
image_paths = ["ressources/images_base"]
images_object = {
    "01.jpg": ["mug", "cup"],
    "02.jpg": ["mug", "cup"],
    "03.jpg": ["mug", "cup"],
    "04.jpg": ["mug", "cup"],
    "05.jpg": ["sponge"],
    "06.jpg": ["sponge"],
    "07.jpg": ["sponge"],
    "08.jpg": ["sponge"],
    "09.jpg": ["gloves", "glove"],
    "10.jpg": ["gloves", "glove"],
    "11.jpg": ["gloves", "glove"],
    "12.jpg": ["gloves", "glove"],
    "13.jpg": ["plate", "dish", "tray"],
    "14.jpg": ["plate", "dish", "tray"],
    "15.jpg": ["plate", "dish", "tray"],
    "16.jpg": ["plate", "dish", "tray"],
    "17.jpg": ["fan"],
    "18.jpg": ["fan"],
    "19.jpg": ["fan"],
    "20.jpg": ["fan"],
    "21.jpg": ["soap bottle", "bottle of soap", "cleaning solution", "hand sanitizer"],
    "22.jpg": ["soap bottle", "bottle of soap", "cleaning solution", "hand sanitizer"],
    "23.jpg": ["soap bottle", "bottle of soap", "cleaning solution", "hand sanitizer"],
    "24.jpg": ["soap bottle", "bottle of soap", "cleaning solution", "hand sanitizer"],
    "25.jpg": ["controller"],
    "26.jpg": ["controller"],
    "27.jpg": ["controller"],
    "28.jpg": ["controller"],
    "29.jpg": ["cloth", "fabric", "clothing"],
    "30.jpg": ["cloth", "fabric", "clothing"],
    "31.jpg": ["cloth", "fabric", "clothing"],
    "32.jpg": ["cloth", "fabric", "clothing"],
    "33.jpg": ["mouse"],
    "34.jpg": ["mouse"],
    "35.jpg": ["mouse"],
    "36.jpg": ["mouse"],
    "37.jpg": ["screwdriver"],
    "38.jpg": ["screwdriver"],
    "39.jpg": ["screwdriver"],
    "40.jpg": ["screwdriver"],
    "41.jpg": ["pcb", "circuit board", "pcie"],
    "42.jpg": ["pcb", "circuit board", "pcie"],
    "43.jpg": ["pcb", "circuit board", "pcie"],
    "44.jpg": ["pcb", "circuit board", "pcie"],
    "45.jpg": ["cleaning foam", "cleaning solution"],
    "46.jpg": ["cleaning foam", "cleaning solution"],
    "47.jpg": ["cleaning foam", "cleaning solution"],
    "48.jpg": ["cleaning foam", "cleaning solution"],
    "49.jpg": ["mug", "cup"],
    "50.jpg": ["sponge"],
    "51.jpg": ["gloves", "glove"],
    "52.jpg": ["plate", "dish", "tray"],
    "53.jpg": ["fan"],
    "54.jpg": ["soap bottle", "bottle of soap", "cleaning solution", "hand sanitizer"],
    "55.jpg": ["controller"],
    "56.jpg": ["cloth", "fabric", "clothing"],
    "57.jpg": ["mouse"],
    "58.jpg": ["screwdriver"],
    "59.jpg": ["pcb", "circuit board", "pcie"],
    "60.jpg": ["cleaning foam", "cleaning solution"]
}

run_evaluation(model_names, prompts, image_paths, images_object)
