from classify_functions import *


# DO NOT USE THIS FUNCTION
# it is not done yet
def main():
    # return
    image_folder_main = "dataset_vlm"
    image_folder_second = "dataset_vlm_crop"
    user_choice_file = "user_choice.csv"
    output_csv = "results_multi.csv"
    output_plot = "classification_plot_multi.png"

    models = [
        # {"name": "llava", "temperatuer":0}
        {"name": "llava:13b", "temperature": 0}, # 7B (4.7GB)
    ]
    #"You were unable to provide a clear classification. Here is a new image of the same object. Can you identify it now ?"
    input_prompts = ["You are a robot. Based on what you see, you must identify the object. You might see your robotic arm, but ignore it, focus on the object. After this: determine if the object should go in the household & misc box (number 1) or the electronics and tools box (number 2). Be concise, identify the object, and you must categorize it in one of the two categories. Do not mention what it is not.", "You got the incorrect classification, try again with a closeup of the same object :"]

    categories = {
        "Household & Misc": ["1", "household", "misc"],
        "Tools/Electronics": ["2", "tools", "electronics"],
        "Unclear/Nothing": ["3", "unclear", "unsure"]
    }

    user_choice_data = read_user_choices(user_choice_file)
    image_list = load_images(image_folder_main)
    csv_rows = init_csv(image_list, user_choice_data, categories)
    csv_rows = process_image_multi(image_list, models, input_prompts, [image_folder_main, image_folder_second], csv_rows, categories)


    # broken / not checked
    # save_csv(output_csv, csv_rows, image_list, models)
    # plot_results(categories, image_folder_main, output_csv, output_plot)


if __name__ == '__main__':
    main()