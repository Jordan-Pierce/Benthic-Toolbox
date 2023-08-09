import os
import sys
import argparse

import json
import pandas as pd


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

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
        return

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
                category_id=class_mapping[r['Scientific Name']],
                bbox=[xmin, ymin, xmax, ymax],
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

    # Single object detector
    if args.single_object_detector:
        train_annotations['Scientific Name'] = 'Object'
        valid_annotations['Scientific Name'] = 'Object'
        test_annotations['Scientific Name'] = 'Object'

    # Combine
    annotations = pd.concat((train_annotations, valid_annotations, test_annotations))

    # Create a class mapping for category id (this means all annotation files must be added)
    class_mapping = {v: i for i, v in enumerate(annotations['Scientific Name'].unique())}

    # Create COCO format annotations for each of the datasets
    to_coco(train_annotations, class_mapping, train_file)
    to_coco(valid_annotations, class_mapping, valid_file)
    to_coco(test_annotations, class_mapping, test_file)

    # Updated class mapping file as expected by training script
    categories = [{'id': class_id, 'name': class_name} for class_name, class_id in class_mapping.items()]

    # Write the JSON data to the output file
    with open(class_map_file, 'w') as output_file:
        json.dump(categories, output_file, indent=3)

    if os.path.exists(class_map_file):
        print(f"NOTE: Class Map JSON file saved to {class_map_file}")
    else:
        print("ERROR: Could not save Class Map JSON file")


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

    parser.add_argument("--output_dir", type=str,
                        default=f"{os.path.abspath('../../../Data/Ground_Truth/')}",
                        help="Path to the output directory.")

    args = parser.parse_args()

    try:
        coco(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
