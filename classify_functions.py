import ollama
import time
import os
import re
import string
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def read_user_choices(user_choice_file):
    """
    Read user choices from a CSV file
    """
    user_choice_data = {}
    with open(user_choice_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 3:
                image_name, description, user_choice = map(str.strip, row)
                user_choice_data[image_name] = {"description": description, "user_choice": user_choice}
    return user_choice_data


def load_images(image_folder):
    """
    Load images from a folder
    """
    image_list = [img for img in sorted(os.listdir(image_folder)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return image_list


def init_csv(image_list, user_choice_data):
    """
    Initialize the CSV data with image info
    """
    csv_rows = {}
    for image_name in image_list:
        uc_info = user_choice_data.get(image_name, {"description": "N/A", "user_choice": "4"}) # default '4' if missing
        description = uc_info["description"]
        user_choice = uc_info["user_choice"]

        csv_rows[image_name] = [image_name, description, user_choice]
    return csv_rows, description, user_choice


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


def classify_response(response):
    """
    Classify the response into one of 4 classes by counting the occurences of specific words
    """
    classif_counts = {"1": 0, "2": 0, "3": 0, "4": 0}
    for word in response.split():
        word = word.strip(string.punctuation).lower()
        if word in ["1", "personal"]:
            classif_counts["1"] += 1
        if word in ["2", "tools", "electronics"]:
            classif_counts["2"] += 1
        if word in ["3", "trash", "rubbish", "dispos", "throw", "thrown"] and "not trash" not in response:
            classif_counts["3"] += 1

    max_classes = [k for k, v in classif_counts.items() if v == max(classif_counts.values())]
    classif = max_classes[0] if len(max_classes) == 1 else "4"
    return classif


def print_response(model_name, image_name, classif, model_response, time_taken, user_choice, i, description, image_count):
    """
    Prints the log of the response with color coding
    """
    classif_color = f"\033[92m{classif}\033[37m" if classif == user_choice else f"\033[91m{classif}\033[37m"
    colored_response = model_response
    for word in ["1", "personal", "2", "tools", "electronics", "3", "trash", "rubbish", "dispos", "throw", "thrown"]:
        if word == "trash":
            colored_response = re.sub(rf'\b(?<!not\s){word}\b', f"\033[1;33m{word}\033[0m", colored_response)
        else:
            colored_response = re.sub(rf'\b{word}\b', f"\033[1;33m{word}\033[0m", colored_response)

    print(f"\n\033[1;34m[{model_name}] \033[37m[{i+1}/{image_count}] {image_name} : Class {classif_color} ({user_choice}) ({time_taken:.2f}s) ({description})\033[0m")
    print(colored_response)


def process_images(image_list, models, user_choice_data, input_prompt, image_folder, csv_rows, user_choice, description):
    """
    Main loop to process images with models
    Returns classification data and error data
    """
    classification_data = {model['name']: [] for model in models}
    error_data = {model['name']: 0 for model in models} # error tracking
    for model in models:
        model_name = model['name']

        print(f"\n=== Processing with Model: {model_name} ===")

        for i, image_name in enumerate(image_list):
            model_response, time_taken = single_call(model_name, input_prompt, os.path.join(image_folder, image_name), model['temperature'])

            classif = classify_response(model_response.lower())
            classification_data[model_name].append((image_name, classif)) # for plot

            # errors
            user_choice = user_choice_data.get(image_name, {}).get("user_choice", "4")
            description = user_choice_data.get(image_name, {}).get("description", "N/A")
            error = 1 if classif != user_choice else 0
            error_data[model_name] += error

            print_response(model_name, image_name, classif, model_response, time_taken, user_choice, i, description, len(image_list))

            if len(csv_rows[image_name]) == 3: # (image_name, description, user_choice)
                for _ in range(len(models) * 3): # space for model data
                    csv_rows[image_name].append("")
            
            model_idx = models.index(model)
            base_idx = 3 + (model_idx * 3)
            csv_rows[image_name][base_idx] = classif
            csv_rows[image_name][base_idx + 1] = model_response
            csv_rows[image_name][base_idx + 2] = round(time_taken, 3)
            
    return csv_rows


def save_csv(csv_file, csv_rows, image_list, models):
    """
    Save the classification data to a CSV file
    """
    header = ["Image", "Description", "User Choice"]
    for model in models:
        header.extend([model['name'], f"{model['name']} Response", f"{model['name']} Time (s)"])

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for image_name in image_list:
            writer.writerow(csv_rows[image_name])


def plot_results(categories, image_folder, csv_file):
    """
    Plot the classification results with user choice comparison
    Data is read directly from the CSV file
    """
    # Read data from CSV file
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)
    
    model_names = []
    model_indices = {}
    model_time_indices = {}
    
    for i in range(3, len(header), 3):
        if i < len(header):
            model_name = header[i]
            if not model_name.endswith("Response") and not model_name.endswith("Time (s)"):
                model_names.append(model_name)
                model_indices[model_name] = i
                model_time_indices[model_name] = i + 2
    
    image_list = []
    csv_rows = {}
    user_choice_list = []
    classification_data = {model: [] for model in model_names}
    error_data = {model: 0 for model in model_names}
    time_data = {model: [] for model in model_names}

    # data: image name, description, user choice, model1 classification, model1 response, model1 time, model2 classification, model2 response, model2 time, ...
    
    for row in rows:
        if len(row) < 3:
            continue # incomplete: skip
            
        image_name = row[0]
        description = row[1]
        user_choice = row[2]
        
        image_list.append(image_name)
        csv_rows[image_name] = row
        user_choice_list.append((image_name, user_choice))
        
        # Extract model classifications
        for model_name in model_names:
            idx = model_indices[model_name]
            time_idx = model_time_indices[model_name]

            if idx < len(row):
                model_classif = row[idx]
                classification_data[model_name].append((image_name, model_classif))
                error = 1 if model_classif != user_choice else 0
                error_data[model_name] += error

                if time_idx < len(row) and row[time_idx]:
                    try:
                        time_data[model_name].append(float(row[time_idx]))
                    except ValueError:
                        pass
    
    avg_time_data = {model: sum(times) / len(times) if len(times) > 0 else 0 for model, times in time_data.items()}

    
    # Plot
    # fig, ax = plt.subplots(figsize=(20, 10))
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    logistic_mult = 2 / (1 + np.exp(-(len(model_names) - 1)))
    offset_step = ((1/len(categories)) * logistic_mult) / len(model_names)

    # user choice
    x_uc = list(range(len(user_choice_list)))
    y_uc = [categories.index("Personal Items") if uc == "1" else categories.index("Tools/Electronics") if uc == "2" else categories.index("Trash") if uc == "3" else categories.index("Unclear/Nothing") for _, uc in user_choice_list]
    ax.scatter(x_uc, y_uc, label="User Choice", color="black", marker='x', s=100, linewidths=2)

    # model classifications
    for idx, (model_name, data) in enumerate(classification_data.items()):
        x = []
        y = []
        for i, (image, classif) in enumerate(data):
            if classif in ["1", "2", "3", "4"]:
                y_value = categories[int(classif) - 1]
                x.append(i)
                y.append(categories.index(y_value) + (idx + 1) * offset_step)

        ax.scatter(x, y, label=model_name, color=colors[idx % len(colors)], marker='o', alpha=0.8, s=100)

    # horizontal lines
    for i in range(len(categories)):
        ax.axhline(y=i, color='gray', linestyle='--', linewidth=0.5)

    # labels and Title
    x_labels = [f"{csv_rows[image][1]} ({i+1})" for i, image in enumerate(image_list)]
    ax.set_xticks(range(len(image_list)))
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Image Index and Description')
    ax.set_ylabel('Classification Category')
    ax.set_title(f'Model Classification Results with User Choice Comparison ({image_folder})')
    ax.legend(loc='upper right', framealpha=0.5)

    # total error & avg time for each model
    error_text = "Total Errors: " + " | ".join([f"{model}: {err}" for model, err in error_data.items()])
    time_text = "Average Generation Time (s): " + " | ".join([f"{model}: {round(time, 2)}" for model, time in avg_time_data.items()])
    plt.figtext(0.5, 0.01, f"{error_text}\n{time_text}", wrap=True, horizontalalignment='center', fontsize=12, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.07, 1, 1]) # (adjust to fit text)
    plt.savefig('classification_plot.png')
    plt.show()

