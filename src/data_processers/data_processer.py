import os
from pathlib import Path
from PIL import Image
import imghdr
import pandas as pd


def process_data_and_retrieve_files_df() -> pd.DataFrame:
    files_df_file_name = "files_df.csv"

    if Path(f"./{files_df_file_name}").is_file():
        files_df = pd.read_csv(f"./{files_df_file_name}", sep=",")
        files_df.label = files_df.label.astype(str)
        return files_df

    processed_file_paths = _remove_incorrect_files_and_return_valid_file_list()

    files_df = pd.DataFrame(processed_file_paths, columns=["file", "label"])
    files_df.sample(frac=1).reset_index(drop=True, inplace=True)

    files_df.to_csv(f"./{files_df_file_name}", sep=",")

    return files_df


def _remove_incorrect_files_and_return_valid_file_list() -> list:
    accepted_file_type = ["jpg", "jpeg", "png", "bmp"]
    dir_to_label_map = {
        "Cats": "0",
        "Dogs": "1"
    }

    data_dir = "./data/CatsDogs"
    data_class_dir_list = os.listdir(data_dir)
    file_paths = []

    for data_class in data_class_dir_list:
        image_dir = os.path.join(data_dir, data_class)

        for image in os.listdir(image_dir):
            image_file_path = os.path.join(image_dir, image)

            try:
                img_file_type = imghdr.what(image_file_path)

                if img_file_type not in accepted_file_type:
                    print(f"File type incorrect. Removing image {image_file_path}. File type: {img_file_type}")
                    os.remove(image_file_path)

                    continue

                img = Image.open(image_file_path)
                img = img.convert("RGB")
                img.save(image_file_path)

                file_paths.append([image_file_path, dir_to_label_map[data_class]])
            except:
                print(f"Error opening image file. Excluding and removing image: {image_file_path}")
                os.remove(image_file_path)

    return file_paths
