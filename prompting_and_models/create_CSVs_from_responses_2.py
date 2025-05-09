import os
import csv
from collections import defaultdict

def create_transformation_CSVs_from_responses(
    responses_path="output/responses_2.csv",
    matrix_folder="output/image_matrices_2",
    transformation_folder="output/transformation_matrices_2",
    model_folder="output/model_matrices_2",
    accuracy_csv_path="output/accuracy_matrix_2.csv"
):
    os.makedirs(matrix_folder, exist_ok=True)
    os.makedirs(transformation_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(os.path.dirname(accuracy_csv_path), exist_ok=True)

    # Containers
    image_data = defaultdict(lambda: defaultdict(dict))          # image -> transformation -> model -> success
    transformation_data = defaultdict(lambda: defaultdict(dict)) # transformation -> image -> model -> success
    model_data = defaultdict(lambda: defaultdict(dict))          # model -> image -> transformation -> success
    accuracy_data = defaultdict(lambda: defaultdict(int))        # transformation -> model -> success_count

    image_labels = {}  # image_key -> label string
    all_transformations = set()
    all_models = set()
    all_images = set()

    # Read and process responses.csv
    with open(responses_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transformation = row['image_folder']
            model = row['model']
            image_name = row['image_name']
            correct_objects = row['correct_object_names'].split(";")[0].strip()
            object_label = correct_objects.replace(" ", "_")
            success = row['success'].strip().lower() == "true"

            image_key = os.path.splitext(image_name)[0]
            image_csv_name = f"{image_key}_{object_label}"

            # Track all items
            all_transformations.add(transformation)
            all_models.add(model)
            all_images.add(image_csv_name)
            image_labels[image_csv_name] = image_csv_name

            # Fill data
            image_data[image_csv_name][transformation][model] = success
            transformation_data[transformation][image_csv_name][model] = success
            model_data[model][image_csv_name][transformation] = success

            if success:
                accuracy_data[transformation][model] += 1

    # Sort keys
    all_transformations = sorted(all_transformations)
    all_models = sorted(all_models)
    all_images = sorted(all_images)

    # # 1. Accuracy matrix
    # with open(accuracy_csv_path, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([""] + all_transformations)
    #     for model in all_models:
    #         row = [model] + [accuracy_data[transformation].get(model, 0) for transformation in all_transformations]
    #         writer.writerow(row)

    # 1. Accuracy matrix
    with open(accuracy_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([""] + all_models)
        for transformation in all_transformations:
            row = [transformation] + [accuracy_data[transformation].get(model, 0) for model in all_models]
            writer.writerow(row)

    # 2. Per-transformation matrix
    for transformation in all_transformations:
        path = os.path.join(transformation_folder, f"{transformation}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([transformation] + all_images)
            for model in all_models:
                row = [model] + [transformation_data[transformation].get(img, {}).get(model, "") for img in all_images]
                writer.writerow(row)

    # 3. Per-model matrix
    for model in all_models:
        path = os.path.join(model_folder, f"{model.replace(':', '_')}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([model] + all_images)
            for transformation in all_transformations:
                row = [transformation] + [model_data[model].get(img, {}).get(transformation, "") for img in all_images]
                writer.writerow(row)

    # 4. Per-image matrix
    for image_csv_name, transformation_map in image_data.items():
        path = os.path.join(matrix_folder, f"{image_csv_name}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([image_csv_name] + all_models)
            for transformation in all_transformations:
                row = [transformation] + [transformation_map.get(transformation, {}).get(model, "") for model in all_models]
                writer.writerow(row)

    print(f"✅ Created:")
    print(f"  - {len(image_data)} image matrix CSVs in '{matrix_folder}'")
    print(f"  - {len(all_transformations)} transformation matrix CSVs in '{transformation_folder}'")
    print(f"  - {len(all_models)} model matrix CSVs in '{model_folder}'")
    print(f"  - 1 accuracy matrix CSV in '{accuracy_csv_path}'")

# Run the function
create_transformation_CSVs_from_responses()
