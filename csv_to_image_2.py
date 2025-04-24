import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np

def csv_to_image(csv_path, output_folder="output/csv_images", dpi=2000):
    df = pd.read_csv(csv_path)
    
    filename = os.path.basename(csv_path)
    image_name = filename.split("_")[0] + ".jpg"
    image_path = os.path.join("ressources/images_base/", filename)
    print(f"csv -> png : {filename}")

    # insert newlines on every 25 characters
    for col in df.columns:
        df[col] = df[col].apply(lambda x: '\n'.join([str(x)[i:i+25] for i in range(0, len(str(x)), 20)]))

    n_rows, n_cols = df.shape

    # Scale figure size based on content
    fig_width = n_cols * 0.5
    fig_height = n_rows * 0.2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    # fig, ax = plt.subplots(figsize=(20, 12), dpi=300)
    ax.axis('off')

    # # Display the image in the top-left corner
    # if os.path.exists(image_path):
    #     img = Image.open(image_path)
    #     new_ax = fig.add_axes([0.045, fig_height*(n_cols-1), 0.2, 0.2], anchor='NW', zorder=1)  # [x, y, width, height]
    #     new_ax.imshow(img)
    #     new_ax.axis('off')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colLoc='center')

    # Adjust styling
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == 0:
            cell.set_text_props(weight='bold')
        val = str(cell.get_text().get_text()).lower()
        if val == 'true':
            cell.set_facecolor('#d4f7d4')
        elif val == 'false':
            cell.set_facecolor('#f8d4d4')

    # Resize table cells slightly to reduce cramping
    table.scale(1.2, 1.5)

    # Place image
    if os.path.exists(image_path):
        image = Image.open(image_path)
        resize_factor = 0.3  # Adjust this factor to change the size of the image
        target_width = int(resize_factor * image.size[0])
        target_height = int(resize_factor * image.size[1])
        image = image.resize((target_width, target_height))

        fig.figimage(image, 0, 0)

    os.makedirs(output_folder, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(csv_path))[0] + ".png"
    img_path = os.path.join(output_folder, img_name)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

def convert_all_csvs_in_folder(folder_path, output_folder):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_to_image(os.path.join(root, file), output_folder)

if __name__ == "__main__":
    convert_all_csvs_in_folder("output/image_matrices_2", "output/image_pngs_2")
    convert_all_csvs_in_folder("output/model_matrices_2", "output/model_pngs_2")
    convert_all_csvs_in_folder("output/transformation_matrices_2", "output/transformation_pngs_2")
    csv_to_image("output/accuracy_matrix_2.csv")
    csv_to_image("output/responses_2.csv")    # will probably run out of memory and crash, but it will work if you have few responses saved in the csv
