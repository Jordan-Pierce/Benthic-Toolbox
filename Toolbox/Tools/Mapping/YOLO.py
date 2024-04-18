import os
import glob
import json
import yaml
from tqdm import tqdm
from shutil import copyfile

import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------


def convert_to_yolo_dataset(csv_path, dataset_folder):
    """

    :param csv_path:
    :param dataset_folder:
    :return:
    """
    df = pd.read_csv(csv_path)

    images_folder = os.path.join(dataset_folder, 'images')
    labels_folder = os.path.join(dataset_folder, 'labels')

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    unique_images = df['Image Path'].unique()

    class_mapping = {class_name: idx for idx, class_name in enumerate(df['ScientificName'].unique())}

    # List to store class names
    class_names = []

    # Split images into train and validation sets
    train_images, val_images = train_test_split(unique_images, test_size=0.2, random_state=42)

    # Create train.txt and convert class names to integers
    with open(os.path.join(dataset_folder, 'train.txt'), 'w') as train_file:
        for train_image in train_images:
            train_file.write(
                f"{os.path.abspath(os.path.join(dataset_folder, 'images', os.path.basename(train_image)))}\n")

    # Create val.txt and convert class names to integers
    with open(os.path.join(dataset_folder, 'val.txt'), 'w') as val_file:
        for val_image in val_images:
            val_file.write(f"{os.path.abspath(os.path.join(dataset_folder, 'images', os.path.basename(val_image)))}\n")

    for image_path in tqdm(unique_images):
        image_name = os.path.basename(image_path)

        width = int(df.loc[df['Image Path'] == image_path, 'Width'].values[0])
        height = int(df.loc[df['Image Path'] == image_path, 'Height'].values[0])

        yolo_annotations = []

        for _, row in df[df['Image Path'] == image_path].iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])

            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # Use the class mapping to get the integer value
            class_value = class_mapping[row['ScientificName']]

            yolo_annotation = f"{class_value} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            yolo_annotations.append(yolo_annotation)

            # Add class name to the list
            if row['ScientificName'] not in class_names:
                class_names.append(row['ScientificName'])

        # Writing YOLO annotations to a text file in the "labels" folder
        labels_file_path = os.path.join(labels_folder, f'{os.path.splitext(image_name)[0]}.txt')
        with open(labels_file_path, 'w') as output_file:
            for annotation in yolo_annotations:
                output_file.write(annotation + '\n')

        # Copy image to the dataset's "images" folder
        destination_path = os.path.join(images_folder, image_name)

        if not os.path.exists(destination_path):
            copyfile(image_path, destination_path)

    # Writing the class mapping to a JSON file in the dataset folder
    classes_json_path = os.path.join(dataset_folder, 'classes.json')
    with open(classes_json_path, 'w') as classes_json:
        json.dump(class_mapping, classes_json, indent=3)

    # Writing data.yaml file
    data_yaml_path = os.path.join(dataset_folder, 'data.yaml')
    with open(data_yaml_path, 'w') as data_yaml:
        data_yaml_content = {
            'train': os.path.abspath(os.path.join(dataset_folder, 'train.txt')),
            'val': os.path.abspath(os.path.join(dataset_folder, 'val.txt')),
            'nc': len(list(set(class_names))),
            'names': list(set(class_names))
        }
        yaml.dump(data_yaml_content, data_yaml, default_flow_style=False)

    return images_folder, labels_folder, data_yaml_path, classes_json_path


# ------------------------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Get the root data directory (Data); OCD
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    root = os.path.dirname(os.path.dirname(root)) + "\\Data"
    root = root.replace("\\", "/")
    assert os.path.exists(root)

    # Input dataset folders
    input_folders = glob.glob(f"{root}/Ground_Truth/*")

    # Where to store the output
    output_dir = f"{root}/YOLO"

    # Loop through the existing folders
    for input_folder in input_folders:
        # Input folder name for cor creating output directory
        folder_name = os.path.basename(input_folder)
        output_folder = f"{output_dir}/{folder_name}"
        os.makedirs(output_folder, exist_ok=True)
        # Input folder's CSV file
        csv_file = f"{input_folder}/annotations.csv"
        assert os.path.exists(csv_file)

        # Convert the csv file to YOLO format
        images, labels, data_yaml, class_json = convert_to_yolo_dataset(csv_file, output_folder)
