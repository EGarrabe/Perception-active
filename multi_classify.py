from vlm_functions import *
from classify_functions import *


# DO NOT USE THIS FUNCTION
# it is not done yet
def main():
    return
    image_folder_main = "dataset_vlm"
    image_folder_second = "dataset_vlm_crop"
    user_choice_file = "user_choice.csv"
    output_csv = "results_multi.csv"
    output_plot = "classification_plot_multi.png"

    models = [
        {"name": "llava", "temperatuer":0}
    ]
    input_prompts = ["You are a robot. Based on what you see, you must identify the object. You might see your robotic arm, but ignore it, focus on the object. After this: determine if the object should go in the household & misc box (number 1) or the electronics and tools box (number 2). Be concise, identify the object, and you must categorize it in one of the two categories. Do not mention what it is not.", "You were unable to provide a clear classification. Here is a new image of the same object. Can you identify it now ?"]

    categories = {
        "Household & Misc": ["1", "household", "misc"],
        "Tools/Electronics": ["2", "tools", "electronics"],
        "Unclear/Nothing": ["3", "unclear", "unsure"]
    }

    user_choice_data = read_user_choices(user_choice_file)
    image_list_1 = load_images(image_folder_main)
    image_list_2 = load_images(image_folder_second)
    csv_rows, description, user_choice = init_csv(image_list_1, user_choice_data)
    


if __name__ == '__main__':
    main()