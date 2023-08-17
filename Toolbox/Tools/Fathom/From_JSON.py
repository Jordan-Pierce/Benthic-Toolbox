import os
import sys
import argparse
import requests
import traceback

from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

import json
import pandas as pd
import matplotlib.pyplot as plt

from fathomnet.api import images


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def plot_distributions(annotations, media_dir, media_name):
    """
    :param annotations:
    :param media_dir:
    :return:
    """

    # Output a data distribution chart
    plt.figure(figsize=(20, 20))
    annotations['ScientificName'].value_counts().plot(kind='bar')
    plt.title(f"{media_name}")
    plt.savefig(f"{media_dir}/ScientificName.png")
    plt.close()

    # Output a data distribution chart
    plt.figure(figsize=(20, 20))
    annotations['Mapped'].value_counts().plot(kind='bar')
    plt.title(f"{media_name}")
    plt.savefig(f"{media_dir}/Mapped.png")
    plt.close()


def download_file(url, path):
    """

    """
    try:

        # Don't re-download
        if os.path.exists(path):
            return path

        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                # Write the content to the file
                file.write(response.content)

            print(f"NOTE: Saved file to {path}")

            return path
        else:
            raise Exception(f"ERROR: Failed to download file. Status code: {response.status_code}")

    except Exception as e:
        print(f"ERROR: An error occurred: {e}")
        return None


def from_json(args):
    """

    :param args:
    :return:
    """
    print("\n###############################################")
    print("Converting from JSON")
    print("###############################################\n")

    if os.path.exists(args.json_file):
        # Open json file as dict
        with open(args.json_file, 'r') as json_file:
            data = json.load(json_file)
        # Change if needed
        if type(data) != list:
            data = [v for k, v in data.items()]
    else:
        print("ERROR: JSON file provided doesn't exist; please check input")
        sys.exit(1)

    if os.path.exists(args.label_map):
        # Open label map as dict
        with open(args.label_map, 'r') as json_file:
            label_map = json.load(json_file)
    else:
        print("ERROR: Label map file provided doesn't exist; please check input")
        sys.exit(1)

    # Get media name from output dir
    media_dir = f"{args.media_dir}\\"
    media_name = os.path.dirname(media_dir).split("\\")[-1]
    frame_dir = f"{media_dir}frames\\"
    # Create output dir
    os.makedirs(frame_dir, exist_ok=True)

    # Output annotation csv
    annotations_path = f"{media_dir}annotations.csv"
    frames_path = f"{media_dir}frames.csv"

    image_urls = []
    annotations = []

    # Loop though all the objects in json file
    for d in data:

        # Image url
        image_url = d['url']
        ext = os.path.basename(image_url).split(".")[-1]

        # Image Name
        image_name = f"{d['uuid']}.{ext}"

        # Image Path
        image_path = f"{frame_dir}{image_name}"

        # Frame
        frame = image_name.split(".")[0]

        # Dimensions
        width = d['width']
        height = d['height']

        # List to download later
        image_urls.append([image_url, image_path])

        # If for some reason
        if 'boundingBoxes' not in d:
            continue

        # Loop though each bounding box, map class label
        for b in d['boundingBoxes']:

            try:

                # Scientific name
                scientific = b['concept']

                # Provided with the benthic label map
                try:
                    mapped = label_map[scientific]
                except:
                    mapped = "No Label"

                # bbox
                xmin = b['x']
                ymin = b['y']
                xmax = xmin + b['width']
                ymax = ymin + b['height']

                # Add to list, each is a row in dataframe
                annotations.append([media_name, image_name, image_path, frame, width, height,
                                    scientific, mapped,
                                    xmin, ymin, xmax, ymax])

            except Exception as e:
                print(f"WARNING: {e}")

    # Pandas dataframe
    annotations = pd.DataFrame(annotations, columns=['Media', 'Image Name', 'Image Path',
                                                     'Frame', 'Width', 'Height',
                                                     'ScientificName', 'Mapped',
                                                     'xmin', 'ymin', 'xmax', 'ymax'])
    # Output to media directory for later
    print(f"NOTE: Saving {len(annotations)} annotations to {media_dir}")
    annotations.to_csv(annotations_path)

    if os.path.exists(annotations_path):
        print("NOTE: Annotations saved successfully")
    else:
        raise Exception

    # Plot the distributions
    plot_distributions(annotations, media_dir, media_name)

    # Download the frames
    frames = []

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(download_file, url, path) for url, path in image_urls]

        for future in as_completed(futures):
            downloaded_path = future.result()
            if downloaded_path:
                frames.append(downloaded_path)

    # Pandas dataframe
    frames = pd.DataFrame(frames, columns=['Name'])

    # Output to media directory for later
    frames.to_csv(frames_path)

    if os.path.exists(frames_path):
        print("NOTE: Annotations saved successfully")
    else:
        raise Exception


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument("--json_file", type=str,
                        help="Path to json file")

    parser.add_argument("--label_map", type=str,
                        default=f"{os.path.abspath('../../../Data/benthic_label_map.json')}",
                        help="Path to label map file")

    parser.add_argument("--media_dir", type=str,
                        help="Path to the output directory.")

    args = parser.parse_args()

    try:
        from_json(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.print_exc())


if __name__ == "__main__":
    main()
