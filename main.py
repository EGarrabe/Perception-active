import os
import sys
import csv
import time
import ollama


def evaluate_model_response(response, correct_names):
    """
    Evaluate whether any of the correct object names appear in the model's response.
    """
    response_lower = response.lower()
    return any(name.lower() in response_lower for name in correct_names)


def load_existing_results(results_path):
    """
    Load the set of completed evaluations to avoid duplicate work.
    """
    existing = set()
    if os.path.exists(results_path):
        with open(results_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['prompt'], row['model'], row['image_folder'], row['image_name'])
                existing.add(key)
    return existing


def single_call(model_name, input_prompt, image_path, input_temperature):
    """
    Make a single call to the model and return the response and the time taken.
    """
    start_time = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[{
            'role': 'user',
            'content': input_prompt,
            'images': [image_path]
        }],
        stream=False,
        options={'temperature': input_temperature}
    )
    time_taken = time.time() - start_time
    return response['message']['content'], time_taken


def format_time(seconds):
    """
    Convert seconds to minutes:seconds format.
    """
    return f"{int(seconds//60)} min {int(seconds%60)} sec"


def write_response_row(path, row, fieldnames):
    """
    Append a single row to the results CSV.
    """
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def get_prompts(txt_path):
    """
    Read all prompts from a text file, one per line.
    """
    with open(txt_path, 'r') as f:
        prompts = f.readlines()
    return [prompt.strip() for prompt in prompts]


def run_evaluation(model_names, prompts, image_paths, images_object,
                   results_path="output/responses.csv"):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    existing = load_existing_results(results_path)

    results_fields = ["prompt", "model", "image_folder", "image_name", "model_response",
                      "correct_object_names", "success", "response_time"]

    if not os.path.exists(results_path):
        with open(results_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results_fields)
            writer.writeheader()

    total_prompts = len(prompts)

    for image_folder in image_paths:
        folder_name = os.path.basename(image_folder)
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
        total_images = len(image_files)
        

        for prompt_idx, prompt in enumerate(prompts, 1):
            prompt_start_time = time.time()
            model_accuracy = {model: 0 for model in model_names}

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
                        response, response_time = single_call(model, prompt, image_path, 0)
                        success = evaluate_model_response(response, correct_names)
                    except Exception as e:
                        response = f"ERROR: {str(e)}"
                        response_time = 0
                        success = False

                    write_response_row(results_path, {
                        "prompt": prompt,
                        "model": model,
                        "image_folder": folder_name,
                        "image_name": image_name,
                        "model_response": response,
                        "correct_object_names": "; ".join(correct_names),
                        "success": success,
                        "response_time": response_time
                    }, results_fields)

                    if success:
                        correct += 1
                        model_accuracy[model] += 1
                        status_line[i] = "\033[0;32m1\033[0m"
                    else:
                        incorrect += 1
                        status_line[i] = "\033[0;31m0\033[0m"

                    total = correct + incorrect
                    line = f"\033[0;36m{model}\033[0m | {''.join(status_line)} total : \033[0;32m{correct}\033[0m/\033[0;31m{incorrect}\033[0m/{total_images} | time : {format_time(time.time() - prompt_start_time)}"
                    sys.stdout.write("\r" + " " * 160 + "\r")
                    sys.stdout.write(line)
                    sys.stdout.flush()

            sys.stdout.write("\r" + " " * 160 + "\r")
            prompt_summary = f"prompt {prompt_idx}/{total_prompts} | " + \
                             " ; ".join(f"\033[0;36m{m}\033[0m \033[0;32m{v}\033[0m" for m, v in model_accuracy.items()) + \
							 f" | time : {format_time(time.time() - prompt_start_time)}"
            print(prompt_summary)


# === CONFIGURATION ===

model_names = [
    "llava", "llava:13b", "llava:34b", "llava-llama3",
    "llama3.2-vision", "gemma3:4b", "gemma3:12b", "gemma3:27b"
]
prompts = get_prompts("ressources/all_prompts.txt")
image_paths = ["ressources/images_base"]

# mapping of image names to the correct object labels
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
    "45.jpg": ["cleaning foam", "cleaning solution"],   # TODO: add "cleaning product"; it would modify 16 evaluations done in the past at the time of writing
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
# === RUN EVALUATION ===
run_evaluation(model_names, prompts, image_paths, images_object)
