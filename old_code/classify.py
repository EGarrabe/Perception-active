from classify_functions import *

def main():
    image_folder = "dataset_vlm_mini"
    user_choice_file = "user_choice_mini.csv"
    csv_file = "results.csv"

    # Uncomment the models to use.
    # You need about 1.2x of vram+ram of the model size (GB) to be able to run it.
    models = [
        {"name": "llava", "temperature": 0}, # 7B (4.7GB)
        # {"name": "llava-llama3", "temperature": 0}, # 8B (5.5GB)
        # {"name": "llama3.2-vision", "temperature": 0}, # 11B (7.9GB)
        # {"name": "llava:13b", "temperature": 0}, # 13B (8GB)
        # {"name": "llava:34b", "temperature": 0}, # 34B (20GB)
    ]

    # input_prompt = "I am a robot carrying out the following task: 'Clear out the desk by putting the items in the blue containers. The personal items should go in box number 1, the tools in box number 2 and rubbish in the trash can.'. This is what I see. Please suggest a first pick-and-place action to begin the task. Be concise, only output the action. If the task is accomplished, answer 'Do nothing'."
    # input_prompt = "You are a robot. Based on what you see, you must identify the object. You might see your robotic arm, but ignore it, focus on the object. After this: if the object should go in the personal items box (number 1), the electronics and tools box (number 2), or if the object is trash and should be thrown in the trash can (number 3). Be concise, only output the action."
    # T
#    input_prompt = "You are a robot. Based on what you see, you must identify the object. You might see your robotic arm, but ignore it, focus on the object. After this: if the object should go in the personal items box (number 1), the electronics and tools box (number 2), or if the object is trash and should be thrown in the trash can (number 3). Be concise, identify the object, and you must categorize it in one of the three categories at all costs. Do not mention what it is not, only what it is and its category."
    # T (without robot arm mention)
    # input_prompt = "You are a robot. Based on what you see, you must identify the object. After this: if the object should go in the personal items box (number 1), the electronics and tools box (number 2), or if the object is trash and should be thrown in the trash can (number 3). Be concise, identify the object, and you must categorize it in one of the three categories at all costs. Do not mention what it is not, only what it is and its category."
    # V
    # input_prompt = "I am a robot carrying out the following task: 'Clear out the desk by putting the items in the blue containers. The personal items should go in container number 1, the tools should go in container number 2, and rubbish should go in the trash can.'. I have picked up the item in my arm, this is what I see. Please suggest what the item seems to be, and what to do with the item to complete the task. Be concise, only output what the item is and the action."
    # 2 part prompts : 1st part to identify the object, 2nd part to categorize it
    input_prompt = "You are a robot. Based on what you see, you must identify the object. You will see a robotic arm in the image, but IGNORE IT, focus on the object. Be concise, identify the object" 
    input_prompt2 = "You are a robot. Based on a description of what you see, you must identify the object. The object should go in one of the following categories : in the personal items box (number 1), the electronics and tools box (number 2), or if the object is trash and should be thrown in the trash can (number 3). Be concise, you must categorize it in one of the three categories at all costs. Do not mention what it is not, only what it is and its category. Here is the description :\n"

    user_choice_data = read_user_choices(user_choice_file)
    image_list = load_images(image_folder)
    csv_rows, description, user_choice = init_csv(image_list, user_choice_data)
    # csv_rows = process_images(image_list, models, user_choice_data, input_prompt, image_folder, csv_rows, user_choice, description)
    csv_rows = process_images_2(image_list, models, user_choice_data, input_prompt, input_prompt2, image_folder, csv_rows, user_choice, description)
    save_csv(csv_file, csv_rows, image_list, models)

    categories = ["Personal Items", "Tools/Electronics", "Trash", "Unclear/Nothing"]
    plot_results(categories, image_folder, csv_file)

if __name__ == "__main__":
    main()