from classify_functions import *

def main():
    set_ = "mini"

    image_folder = f"dataset_vlm_{set_}"
    user_choice_file = f"user_choice_{set_}.csv"
    output_csv = f"results_{set_}.csv"
    output_plot = f"classification_plot_{set_}.png"

    # Uncomment the models to use.
    # You need about 1.2x of vram+ram of the model size (GB) to be able to run it.
    models = [
        # {"name": "granite3.2-vision", "temperature":0}, # 2B (2.3GB)
        # {"name": "gemma3", "temperature": 0}, # 4B (3.3GB)
        {"name": "llava", "temperature": 0}, # 7B (4.7GB)
        # {"name": "llava-llama3", "temperature": 0}, # 8B (5.5GB)
        # {"name": "llama3.2-vision", "temperature": 0}, # 11B (7.9GB)
        # {"name": "llava:13b", "temperature": 0}, # 13B (8GB)
        # {"name": "llava:34b", "temperature": 0}, # 34B (20GB)
    ]

    # input_prompt = "I am a robot carrying out the following task: 'Clear out the desk by putting the items in the blue containers. The personal items should go in box number 1, the tools in box number 2 and rubbish in the trash can.'. This is what I see. Please suggest a first pick-and-place action to begin the task. Be concise, only output the action. If the task is accomplished, answer 'Do nothing'."
    # input_prompt = "You are a robot. Based on what you see, you must identify the object. You might see your robotic arm, but ignore it, focus on the object. After this: if the object should go in the personal items box (number 1), the electronics and tools box (number 2), or if the object is trash and should be thrown in the trash can (number 3). Be concise, only output the action."
    # input_prompt = "You are a robot. Based on what you see, you must identify the object. You might see your robotic arm, but ignore it, focus on the object. After this: if the object should go in the personal items box (number 1), the electronics and tools box (number 2), or if the object is trash and should be thrown in the trash can (number 3). Be concise, identify the object, and you must categorize it in one of the three categories at all costs. Do not mention what it is not, only what it is and its category."
    input_prompt = "You are a robot. Based on what you see, you must identify the object. You might see your robotic arm, but ignore it, focus on the object. After this: determine if the object should go in the household & misc box (number 1) or the electronics and tools box (number 2). Be concise, identify the object, and you must categorize it in one of the two categories. Do not mention what it is not."

    categories = {
        "Household & Misc": ["1", "household", "misc"],
        "Tools/Electronics": ["2", "tools", "electronics"],
        "Unclear/Nothing": ["3", "unclear", "unsure"]
    }
    user_choice_data = read_user_choices(user_choice_file)
    image_list = load_images(image_folder)
    csv_rows = init_csv(image_list, user_choice_data, categories)
    csv_rows = process_images(image_list, models, input_prompt, image_folder, csv_rows, categories)
    save_csv(output_csv, csv_rows, image_list, models)

    plot_results(categories, image_folder, output_csv, output_plot)

if __name__ == "__main__":
    main()