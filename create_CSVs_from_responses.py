import os
import csv
from collections import defaultdict

def create_CSVs_from_responses(
    responses_path="output/responses.csv",
    matrix_folder="output/image_matrices",
    prompt_folder="output/prompt_matrices",
    model_folder="output/model_matrices",
    accuracy_csv_path="output/accuracy_matrix.csv"
):
    os.makedirs(matrix_folder, exist_ok=True)
    os.makedirs(prompt_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(os.path.dirname(accuracy_csv_path), exist_ok=True)

    # Containers
    image_data = defaultdict(lambda: defaultdict(dict))       # image -> prompt -> model -> success
    prompt_data = defaultdict(lambda: defaultdict(dict))      # prompt -> image -> model -> success
    model_data = defaultdict(lambda: defaultdict(dict))       # model -> image -> prompt -> success
    accuracy_data = defaultdict(lambda: defaultdict(int))     # prompt -> model -> success_count

    image_labels = {}  # image_key -> image_object string
    all_prompts = set()
    all_models = set()
    all_images = set()

    # Read and process responses.csv
    with open(responses_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            model = row['model']
            image_name = row['image_name']
            correct_objects = row['correct_object_names'].split(";")[0].strip()
            object_label = correct_objects.replace(" ", "_")
            success = row['success'].strip().lower() == "true"

            image_key = os.path.splitext(image_name)[0]
            image_csv_name = f"{image_key}_{object_label}"

            # Track all items
            all_prompts.add(prompt)
            all_models.add(model)
            all_images.add(image_csv_name)
            image_labels[image_csv_name] = image_csv_name

            # Fill all 4 data structures
            image_data[image_csv_name][prompt][model] = success
            prompt_data[prompt][image_csv_name][model] = success
            model_data[model][image_csv_name][prompt] = success

            # Count success for accuracy matrix
            if success:
                accuracy_data[prompt][model] += 1

    all_prompts = sorted(all_prompts)
    all_models = sorted(all_models)
    all_images = sorted(all_images)

    # Save image-based CSVs
    for image_csv_name, prompt_map in image_data.items():
        path = os.path.join(matrix_folder, f"{image_csv_name}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([image_csv_name] + all_models)
            for prompt in all_prompts:
                row = [prompt] + [prompt_map.get(prompt, {}).get(model, "") for model in all_models]
                writer.writerow(row)

    # Save prompt-based CSVs
    for i, prompt in enumerate(all_prompts, 1):
        path = os.path.join(prompt_folder, f"prompt_{i}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([prompt] + all_images)
            for model in all_models:
                row = [model] + [prompt_data[prompt].get(img, {}).get(model, "") for img in all_images]
                writer.writerow(row)

    # Save model-based CSVs
    for model in all_models:
        path = os.path.join(model_folder, f"{model.replace(':', '_')}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([model] + all_images)
            for prompt in all_prompts:
                row = [prompt] + [model_data[model].get(img, {}).get(prompt, "") for img in all_images]
                writer.writerow(row)

    # Save accuracy CSV
    with open(accuracy_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([""] + all_models)
        for prompt in all_prompts:
            row = [prompt] + [accuracy_data[prompt].get(model, 0) for model in all_models]
            writer.writerow(row)

    print(f"✅ Created:")
    print(f"  - {len(image_data)} image matrix CSVs in '{matrix_folder}'")
    print(f"  - {len(all_prompts)} prompt matrix CSVs in '{prompt_folder}'")
    print(f"  - {len(all_models)} model matrix CSVs in '{model_folder}'")
    print(f"  - 1 accuracy matrix CSV in '{accuracy_csv_path}'")

create_CSVs_from_responses(
    responses_path="output/responses.csv",
    matrix_folder="output/image_matrices",
    prompt_folder="output/prompt_matrices",
    model_folder="output/model_matrices",
    accuracy_csv_path="output/accuracy_matrix.csv"
)