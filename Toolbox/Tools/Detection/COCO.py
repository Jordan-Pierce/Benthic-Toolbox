import os
import sys
import argparse

import json
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def coco(args):
    """
    :param ann_file:
    :param out_file:
    :return:
    """
    print("\n###############################################")
    print("Converting to COCO")
    print("###############################################\n")

    if args.annotations:
        annotation_files = args.annotations
        annotations = pd.DataFrame()
    else:
        print(f"ERROR: No annotation files were provided as input")
        sys.exit(1)

    # Set the variables
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Output coco annotation file
    coco_file = f"{output_dir}\\coco_annotations.json"

    # Output class map json file
    class_map_file = f"{output_dir}\\class_map.json"

    # To hold the coco formatted data
    images = []
    objects = []

    # Loop through the annotation files
    for annotation_file in annotation_files:

        if not os.path.exists(annotation_file):
            print(f"ERROR: Annotation file {annotation_file} does not exists")
            sys.exit(1)
        else:
            # Group all the annotations together into a single dataframe
            annotations = pd.concat((annotations, pd.read_csv(annotation_file, index_col=0)))

    # Create a class mapping for category id (this means all annotation files must be added)
    class_mapping = {v: i for i, v in enumerate(annotations['ScientificName'].unique())}

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

    parser.add_argument("--annotations", type=str, nargs="+",
                        help="Path to the annotation files")

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
