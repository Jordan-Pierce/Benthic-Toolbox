import os
import sys
import argparse

import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------


def plot_coco_samples(coco_file, output_dir, n_images=5, seed=None):
    """
    :param coco_annotations_file:
    :param output_dir:
    :param n_images:
    :param seed:
    :return:
    """

    if not os.path.exists(coco_file):
        return

    print(f"NOTE: Plotting {n_images} samples from {os.path.basename(coco_file)}")

    # Load COCO annotations JSON file
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(seed)

    # Randomly shuffle image IDs
    image_ids = [img_info['id'] for img_info in coco_data['images']]
    random.shuffle(image_ids)

    # Loop through randomly selected images
    for img_id in image_ids[:n_images]:
        img_info = next(info for info in coco_data['images'] if info['id'] == img_id)
        img_file_name = img_info['file_name']

        # Find annotations for the current image
        img_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == img_id]

        if len(img_annotations) == 0:
            continue

        # Load and plot the image
        img_path = os.path.join(output_dir, img_file_name)
        image = plt.imread(img_path)
        plt.imshow(image)

        # Plot bounding boxes
        for annotation in img_annotations:
            bbox = annotation['bbox']
            bbox_rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                  linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(bbox_rect)

        plt.axis('off')
        plt.title(f"Image ID: {img_id}")

        # Save the plot
        output_file = os.path.join(output_dir, f"image_{img_id}.png")
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def concat_annotations(annotation_files):
    """
    :param annotation_files:
    :return:
    """

    # Annotations
    annotations = pd.DataFrame()

    if not annotation_files:
        return annotations

    # Loop through the annotation files
    for annotation_file in annotation_files:

        if not os.path.exists(annotation_file):
            print(f"ERROR: Annotation file {annotation_file} does not exists")
            sys.exit(1)
        else:
            # Group all the annotations together into a single dataframe
            annotations = pd.concat((annotations, pd.read_csv(annotation_file, index_col=0)))

    # Reset the index
    annotations.reset_index(drop=True, inplace=True)

    return annotations


def to_coco(annotations, class_mapping, coco_file):
    """
    :param annotations:
    :param class_map:
    :param coco_file:
    :return:
    """

    # If no annotations, return
    if annotations.empty:
        return ""

    # To hold the coco formatted data
    images = []
    objects = []

    # Loop through first based on unique images
    for i_idx, image_name in enumerate(annotations['Image Name'].unique()):

        # Get the current annotations for the image
        current_annotations = annotations[annotations['Image Name'] == image_name]

        # Loop through each object
        for _, (a_idx, r) in enumerate(current_annotations.iterrows()):

            if _ == 0:
                # Get the attributes for the first image to serve for all of them
                img_path = r['Image Path']
                height = r['Height']
                width = r['Width']

                images.append(dict(id=i_idx, file_name=img_path, height=height, width=width))

            # Bounding box
            xmin = r['xmin']
            ymin = r['ymin']
            xmax = r['xmax']
            ymax = r['ymax']

            # Instance segmentation mask as polygon (list of vertices representing a rectangle)
            mask_polygon = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]

            data_anno = dict(
                image_id=i_idx,
                id=a_idx,
                category_id=class_mapping[r['ScientificName']],
                bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                segmentation=[mask_polygon],
                area=(xmax - xmin) * (ymax - ymin),
                iscrowd=0)

            objects.append(data_anno)

    # Create the categories
    categories = [{'id': class_id, 'name': class_name} for class_name, class_id in class_mapping.items()]

    # Create COCO format json
    coco_format_json = dict(
        images=images,
        annotations=objects,
        categories=categories)

    # Save to output file
    with open(coco_file, 'w') as out:
        json.dump(coco_format_json, out, indent=3)

    if os.path.exists(coco_file):
        print(f"NOTE: COCO formatted file saved to {coco_file}")
    else:
        print("ERROR: Could not save COCO formatted file")

    return coco_file


def coco(args):
    """
    :param ann_file:
    :param out_file:
    :return:
    """
    print("\n###############################################")
    print("Converting to COCO")
    print("###############################################\n")

    # Set the variables
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Output coco annotation files
    train_file = f"{output_dir}\\train.json"
    valid_file = f"{output_dir}\\valid.json"
    test_file = f"{output_dir}\\test.json"

    # Output class map json file
    class_map_file = f"{output_dir}\\class_map.json"

    # Get the annotations per dataset
    train_annotations = concat_annotations(args.train_files)
    valid_annotations = concat_annotations(args.valid_files)
    test_annotations = concat_annotations(args.test_files)

    # Filter based on list
    if args.only_include:
        train_annotations = train_annotations[train_annotations['ScientificName'].isin(args.only_include)]
        valid_annotations = valid_annotations[valid_annotations['ScientificName'].isin(args.only_include)]
        test_annotations = test_annotations[test_annotations['ScientificName'].isin(args.only_include)]
    # Single object detector (both could be used)
    elif args.single_object_detector:
        train_annotations['ScientificName'] = 'Object'
        valid_annotations['ScientificName'] = 'Object'
        test_annotations['ScientificName'] = 'Object'

    # Combine
    annotations = pd.concat((train_annotations, valid_annotations, test_annotations))

    # Create a class mapping for category id (this means all annotation files must be added)
    class_mapping = {v: i for i, v in enumerate(annotations['ScientificName'].unique())}

    # Create COCO format annotations for each of the datasets
    train_coco = to_coco(train_annotations, class_mapping, train_file)
    valid_coco = to_coco(valid_annotations, class_mapping, valid_file)
    test_coco = to_coco(test_annotations, class_mapping, test_file)

    # Updated class mapping file as expected by training script
    categories = [{'id': class_id, 'name': class_name} for class_name, class_id in class_mapping.items()]

    # Write the JSON data to the output file
    with open(class_map_file, 'w') as output_file:
        json.dump(categories, output_file, indent=3)

    if os.path.exists(class_map_file):
        print(f"NOTE: Class Map JSON file saved to {class_map_file}")
    else:
        print("ERROR: Could not save Class Map JSON file")

    # Plot samples in ground-truth directory
    if args.plot_n_samples:
        n_samples = args.plot_n_samples
        plot_coco_samples(train_coco, f"{output_dir}\\plots\\", n_samples)
        plot_coco_samples(valid_coco, f"{output_dir}\\plots\\", n_samples)
        plot_coco_samples(test_coco, f"{output_dir}\\plots\\", n_samples)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="COCO")

    parser.add_argument("--train_files", type=str, nargs="+",
                        help="Path to the training annotation files")

    parser.add_argument("--valid_files", type=str, nargs="+",
                        help="Path to the validation annotation files")

    parser.add_argument("--test_files", type=str, nargs="+",
                        help="Path to the testing annotation files")

    parser.add_argument("--single_object_detector", action='store_true',
                        help="All labels are replaced with a single class category")

    parser.add_argument("--only_include", type=str, nargs="+",
                        help="A list of class categories to use, others filtered out")

    parser.add_argument("--plot_n_samples", type=int, default=5,
                        help="Plot N samples to show COCO labels on images")

    parser.add_argument("--output_dir", type=str,
                        default=f"{os.path.abspath('../../../Data/COCO/')}",
                        help="Path to the output directory.")

    args = parser.parse_args()

    try:
        coco(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
