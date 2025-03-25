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


def init_csv(image_list, user_choice_data, categories):
    """
    Initialize the CSV data with image info
    Format:
    image, description, user_choice, model1_classif, model1_response, model1_time, model2_classif, model2_response, model2_time, ...
    """
    csv_rows = {}
    for image_name in image_list:
        uc_info = user_choice_data.get(image_name, {"description": "N/A", "user_choice": "Unclear/Nothing"})
        description = uc_info["description"]
        user_choice = uc_info["user_choice"]
        csv_rows[image_name] = [image_name, description, user_choice]
    return csv_rows


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


def vlm_call(model_name, all_messages, input_prompt, image_path):
    """
    same as single_call, but with a list of messages for sequential calls
    new message is all_messages[-1]['content']
    """
    new_message = {'role': 'user', 'content': input_prompt}
    if image_path: new_message['image'] = image_path
    all_messages.append(new_message)
    
    start_time = time.time()
    response = ollama.chat(
        model=model_name,
        messages=all_messages,
        stream=False,
        options={'temperature': 0}
    )
    time_taken = time.time() - start_time
    
    assistant_message = {
        'role': response['message']['role'],
        'content': response['message']['content']
    }
    all_messages.append(assistant_message)
    return all_messages, time_taken


def classify_response(response, categories):
    """
    Classify the response into one of the classes by counting the occurences of specific words
    """
    classif_counts = {cat: 0 for cat in categories}
    for word in response.split():
        word = word.strip(string.punctuation).lower()
        for cat, words in categories.items():
            if word in words:
                classif_counts[cat] += 1
    max_classes = [k for k, v in classif_counts.items() if v == max(classif_counts.values())]
    classif = max_classes[0] if len(max_classes) == 1 else "Unclear/Nothing"
    return list(categories.keys()).index(classif) + 1



def print_response(model_name, image_name, classif, model_response, time_taken, user_choice, i, description, image_count, categories):
    """
    Prints the log of the response with color coding
    """
    classif_color = f"\033[92m{classif}\033[37m" if classif == user_choice else f"\033[91m{classif}\033[37m"
    colored_response = model_response
    for cat, words in categories.items():
        for word in words:
            colored_response = re.sub(rf'\b{word}\b', f"\033[1;33m{word}\033[0m", colored_response)

    print(f"\n\033[1;34m[{model_name}] \033[37m[{i+1}/{image_count}] {image_name} : Class {classif_color} ({user_choice}) ({time_taken:.2f}s) ({description})\033[0m")
    print(colored_response)


def update_csv_rows(csv_rows, image_name, model_number, classif, model_response, time_taken, model_count):
    """
    Update the CSV rows with the classification data
    """
    len_header = 3 # image_name, description, user_choice
    data_per_model = 3 # classif, response, time
    if len(csv_rows[image_name]) == len_header: # (image_name, description, user_choice)
        for _ in range(model_count * data_per_model): # space for model data
            csv_rows[image_name].append("")
    
    base_idx = len_header + (model_number * data_per_model)
    csv_rows[image_name][base_idx:base_idx + 3] = [classif, model_response, round(time_taken, 3)] # add (classif, response, time)
    return csv_rows

def update_csv_rows_multi(csv_rows, image_name, model_number, classif, model_response, time_taken, model_count, i):
    """
    Update the CSV rows with the classification data
    """
    len_header = 3 # image_name, description, user_choice
    data_per_model = 6 # classif, response, time
    if len(csv_rows[image_name]) == len_header: # (image_name, description, user_choice)
        for _ in range(model_count * data_per_model): # space for model data
            csv_rows[image_name].append("")
    
    base_idx = len_header + (model_number * data_per_model)
    if i == 0:
        csv_rows[image_name][base_idx:base_idx + 6] = [classif, model_response, round(time_taken, 3), "", "", ""] # add (classif, response, time)
    else:
        csv_rows[image_name][base_idx + 3:base_idx + 6] = [classif, model_response, round(time_taken, 3)]
    
    return csv_rows


def process_images(image_list, models, input_prompt, image_folder, csv_rows, categories):
    """
    Main loop to process images with models
    Returns classification data and error data
    
    Converts numeric user choices to category names for proper comparison
    """
    for model in models:
        model_name = model['name']
        print(f"\n=== Processing with Model: {model_name} ===")

        for i, image_name in enumerate(image_list):
            # Get the image path and call the model
            model_response, time_taken = single_call(model_name, input_prompt, os.path.join(image_folder, image_name), model['temperature'])

            # Classify the response
            classif = classify_response(model_response.lower(), categories)
            
            # Print response with color coding
            user_choice = int(csv_rows[image_name][2])
            print_response(model_name, image_name, classif, model_response, time_taken, user_choice, i, csv_rows[image_name][1], len(image_list), categories)
            
            # Update CSV rows
            csv_rows = update_csv_rows(csv_rows, image_name, models.index(model), classif, model_response, time_taken, len(models))
            
    return csv_rows

def process_image_multi(image_list, models, input_prompts, image_folders, csv_rows, categories):
    """
    Main loop to process images with models
    Returns classification data and error data
    
    Converts numeric user choices to category names for proper comparison
    """
    for model in models:
        model_name = model['name']
        print(f"\n=== Processing with Model: {model_name} ===")
        
        # Try with the first prompt, if classification is wrong, try with the second prompt and image
        for i, image_name in enumerate(image_list):
            all_messages = []
            for j, input_prompt in enumerate(input_prompts):
                # Get the image path and call the model
                all_messages, time_taken = vlm_call(model_name, all_messages, input_prompt, os.path.join(image_folders[j], image_name))
                model_response = all_messages[-1]['content']
                print(all_messages)

                # Classify the response
                classif = classify_response(model_response.lower(), categories)

                # Print response with color coding
                user_choice = int(csv_rows[image_name][2])
                print_response(model_name, image_name, classif, model_response, time_taken, user_choice, i, csv_rows[image_name][1], len(image_list), categories)

                # Update CSV rows
                csv_rows = update_csv_rows_multi(csv_rows, image_name, models.index(model), classif, model_response, time_taken, len(models), j)

                if classif == user_choice:
                    break
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


def load_csv(csv_file, categories):
    """
    Load the classification data from a CSV file
    """
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)
    
    category_names = list(categories.keys())
    model_names = []
    model_indices, model_time_indices = {}, {}

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

    for row in rows:
        if len(row) < 3:
            continue
        image_name = row[0]
        description = row[1]
        user_choice_raw = row[2]

        if user_choice_raw.isdigit() and int(user_choice_raw) <= len(category_names):
            idx = int(user_choice_raw) - 1
            if 0 <= idx < len(category_names):
                user_choice = category_names[idx]
            else:
                user_choice = "Unclear/Nothing"
        else:
            user_choice = user_choice_raw

        image_list.append(image_name)
        csv_rows[image_name] = row
        user_choice_list.append((image_name, user_choice))

        for model_name in model_names:
            idx = model_indices[model_name]
            time_idx = model_time_indices[model_name]

            if idx < len(row):
                model_classif_raw = row[idx]

                if model_classif_raw.isdigit() and int(model_classif_raw) <= len(category_names):
                    model_idx = int(model_classif_raw) - 1
                    if 0 <= model_idx < len(category_names):
                        model_classif = category_names[model_idx]
                    else:
                        model_classif = model_classif_raw
                else:
                    model_classif = model_classif_raw

                classification_data[model_name].append((image_name, model_classif))
                error = 1 if model_classif != user_choice else 0
                error_data[model_name] += error

                if time_idx < len(row) and row[time_idx]:
                    try:
                        time_data[model_name].append(float(row[time_idx]))
                    except ValueError:
                        pass

    avg_time_data = {model: sum(times) / len(times) if len(times) > 0 else 0 for model, times in time_data.items()}
    return image_list, csv_rows, user_choice_list, classification_data, error_data, avg_time_data, category_names, model_names

def plot_results(categories, image_folder, csv_file, output_plot):
    """
    Plot the classification results with user choice comparison
    """
    image_list, csv_rows, user_choice_list, classification_data, error_data, avg_time_data, category_names, model_names = load_csv(csv_file, categories)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    logistic_mult = 2 / (1 + np.exp(-(len(model_names) - 1)))
    offset_step = ((1/len(category_names)) * logistic_mult) / len(model_names)

    # user choice
    x_uc = list(range(len(user_choice_list)))
    
    y_uc = []
    for _, uc in user_choice_list:
        if uc in category_names:
            y_uc.append(category_names.index(uc))
        else:
            # If not found, append the last category (usually "Unclear/Nothing")
            y_uc.append(len(category_names) - 1)
    
    ax.scatter(x_uc, y_uc, label="User Choice", color="black", marker='x', s=100, linewidths=2)

    # model classifications
    for idx, (model_name, data) in enumerate(classification_data.items()):
        x = []
        y = []
        for i, (image, classif) in enumerate(data):
            if classif in category_names:
                y_value = classif
                x.append(i)
                y.append(category_names.index(y_value) + (idx + 1) * offset_step)
            else:
                # Skip invalid classifications
                continue

        ax.scatter(x, y, label=model_name, color=colors[idx % len(colors)], marker='o', alpha=0.8, s=100)

    # horizontal lines
    for i in range(len(category_names)):
        ax.axhline(y=i, color='gray', linestyle='--', linewidth=0.5)

    # labels and Title
    x_labels = [f"{csv_rows[image][1]} ({i+1})" for i, image in enumerate(image_list)]
    ax.set_xticks(range(len(image_list)))
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(category_names)))
    ax.set_yticklabels(category_names)
    ax.set_xlabel('Image Index and Description')
    ax.set_ylabel('Classification Category')
    ax.set_title(f'Model Classification Results with User Choice Comparison ({image_folder})')
    ax.legend(loc='upper right', framealpha=0.5)

    # total error & avg time for each model
    error_text = "Total Errors: " + " | ".join([f"{model}: {err}" for model, err in error_data.items()])
    time_text = "Average Generation Time (s): " + " | ".join([f"{model}: {round(time, 2)}" for model, time in avg_time_data.items()])
    plt.figtext(0.5, 0.01, f"{error_text}\n{time_text}", wrap=True, horizontalalignment='center', fontsize=12, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.07, 1, 1]) # (adjust to fit text)
    plt.savefig(output_plot)
    plt.show()