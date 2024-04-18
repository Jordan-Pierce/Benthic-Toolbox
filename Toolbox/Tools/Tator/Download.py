import os
import sys
import shutil
import argparse
import traceback
from tqdm import tqdm

import json
import tator
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
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


def download_image(api, media_id, frame, media_dir):
    """
    :param api:
    :param media:
    :param frame:
    :param media_dir:
    :return:
    """
    # Location media directory
    frame_dir = f"{media_dir}/frames/"
    os.makedirs(frame_dir, exist_ok=True)

    # Location of file
    path = f"{frame_dir}/{str(frame)}.jpg"

    # If it doesn't already exist, download, move.
    if not os.path.exists(path):
        temp = api.get_frame(media_id, frames=[frame])
        shutil.move(temp, path)

    return path


def get_localizations(localizations, tracks):
    """
    :param localizations:
    :param tracks:
    :return:
    """
    print(f"NOTE: Merging track localizations and localizations")

    track_localizations = []
    clean_localizations = []

    # Create a dictionary for faster lookup
    localizations_dict = {l.id: l for l in localizations}

    for track in tracks:
        attributes = track.attributes
        attributes['track_id'] = track.id
        for localization_id in track.localizations:
            # Use the dictionary for faster lookup
            localization = localizations_dict.get(localization_id)

            if localization:
                t = {'id': localization_id}
                t.update(attributes)
                track_localizations.append(t)

    for track_localization in track_localizations:
        clean_localization = localizations_dict.get(track_localization['id'])

        # If there is a localization (there should be)
        if clean_localization:
            # If the localization doesn't have attributes (legacy)
            if clean_localization.attributes is None:
                clean_localization.attributes = track_localization

            else:
                # Update the localization with attributes from track localization
                clean_localization.attributes.update(track_localization)

            # Save in cleaned list
            clean_localizations.append(clean_localization)

    return clean_localizations


def download(args):
    """
    :param args:
    :return:
    """

    print("\n###############################################")
    print("Download")
    print("###############################################\n")

    # Root data location
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Setting the api given the token, authentication
        token = args.api_token
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        print(f"ERROR: Could not authenticate with provided API Token\n{e}")
        sys.exit(1)

    # Get the label map
    if os.path.exists(args.label_map):
        # Open label map as dict
        with open(args.label_map, 'r') as json_file:
            label_map = json.load(json_file)
    else:
        print("ERROR: Label map file provided doesn't exist; please check input")
        sys.exit(1)

    # Pass the project and media list to variables
    if args.project_id:
        project_id = args.project_id
    else:
        print(f"ERROR: Project ID provided is invalid; please check input")
        sys.exit(1)

    if args.media_ids:
        media_ids = args.media_ids
    else:
        print(f"ERROR: Medias provided is invalid; please check input")
        sys.exit(1)

    try:
        # The type of localization for the project (bounding box, attributes)
        loc_type_id = 440  # Detection Box
        loc_name = api.get_localization_type(loc_type_id).name
        layer_type_id = 228  # AI Experiments
        layer_name = api.get_version(layer_type_id).name
        state_type_id = 438  # State Type
        state_name = api.get_state_type(state_type_id).name
    except Exception as e:
        print(f"ERROR: Could not find the correct localization type in project {project_id}")
        sys.exit(1)

    # Loop through medias
    for media_id in media_ids:

        try:
            # Name used for output
            Media = api.get_media(media_id)
            media_name = Media.name
            media_dir = f"{output_dir}/{media_id}/"
            os.makedirs(media_dir, exist_ok=True)

            print(f"NOTE: Media ID {media_id} corresponds to {Media.name}")

            # Get all localizations, filter for those that are actually bounding boxes
            localizations = api.get_localization_list(project_id, media_id=[media_id])
            print(f"NOTE: Found {len(localizations)} annotations")
            localizations = [l for l in localizations if None not in [l.x, l.y, l.width, l.height]]
            print(f"NOTE: Found {len(localizations)} localizations")

            # Get all the tracks, filter for just those with bounding boxes
            tracks = api.get_state_list(project_id, media_id=[media_id])
            tracks = [t for t in tracks if t.localizations]
            print(f"NOTE: Found {len(tracks)} tracks")

            # Combine the two to create a standardized list of localizations
            localizations = get_localizations(localizations, tracks)
            print(f"NOTE: {len(localizations)} total localizations for media {media_name}")

            # Get the frames associate with localizations
            frames = list(set([l.frame for l in localizations]))
            print(f"NOTE: Found {len(frames)} frames with localizations for media {media_name}")

            if not frames or not localizations:
                print("NOTE: No frames with localizations for this media")
                continue

            # Download the associated frames
            print(f"NOTE: Downloading {len(frames)} frames for {media_name}")
            with ThreadPoolExecutor(max_workers=100) as executor:
                paths = [executor.submit(download_image, api, media_id, frame, media_dir) for frame in frames]
                # Contains a list of frame paths on local machine
                paths = [future.result() for future in paths]

            # Output a dataframe for later
            pd.DataFrame(paths, columns=['Name']).to_csv(f"{media_dir}/frames.csv")
            print(f"NOTE: Downloaded {len(paths)} frames for {media_name}")

            # Reformat the localizations so that they are per-frame, making it easier to create coco files
            annotations = []

            # Loop through each unique frame
            for f_idx, frame in tqdm(enumerate(frames)):

                # Find the localizations
                frame_localizations = [l for l in localizations if l.frame == frame]

                # Loop through the frame localizations
                for frame_localization in frame_localizations:
                    # Tator format
                    x = frame_localization.x
                    y = frame_localization.y
                    w = frame_localization.width
                    h = frame_localization.height

                    # Convert to COCO format
                    xmin = int(x * Media.width)
                    ymin = int(y * Media.height)
                    xmax = int(w * Media.width) + xmin
                    ymax = int(h * Media.height) + ymin

                    # Frame number, localization id
                    frame_number = frame_localization.frame
                    localization_id = frame_localization.id

                    # Legacy attribute need review...
                    if 'Needs Review' in frame_localization.attributes:
                        n = 'Needs Review'
                    elif 'NeedsReview' in frame_localization.attributes:
                        n = 'NeedsReview'
                    else:
                        n = ""

                    if n in frame_localization.attributes:
                        needs_review = frame_localization.attributes[n]
                    else:
                        needs_review = "NULL"

                    # Legacy attribute name...
                    if 'Scientific Name' in frame_localization.attributes:
                        s = 'Scientific Name'
                    elif 'ScientificName' in frame_localization.attributes:
                        s = 'ScientificName'
                    else:
                        s = ""

                    # Get the actual scientific name
                    if s in frame_localization.attributes:

                        # Make sure it's a string
                        scientific = str(frame_localization.attributes[s])

                        # If it doesn't have an actual label...
                        if scientific.lower() in ['0', '', ' ', 'unlabeled' 'undefined']:
                            scientific = "No Label"
                    else:
                        print(f"WARNING: Frame {frame} localization {localization_id} has no label!")
                        scientific = "No Label"

                    try:
                        # Get the benthic mapped label
                        # CommonName use is inconsistent...
                        mapped = label_map[scientific]
                    except:
                        mapped = "No Label"

                    # Row in dataframe
                    annotation = [
                        media_id,
                        os.path.basename(paths[f_idx]),
                        paths[f_idx],
                        frame_number,
                        Media.width,
                        Media.height,
                        scientific,
                        mapped,
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        needs_review
                    ]

                    # Add to list
                    annotations.append(annotation)

            if annotations:
                # Pandas dataframe
                annotations = pd.DataFrame(annotations, columns=['Media', 'Image Name', 'Image Path',
                                                                 'Frame', 'Width', 'Height',
                                                                 'ScientificName', 'Mapped',
                                                                 'xmin', 'ymin', 'xmax', 'ymax',
                                                                 'Needs Review'])
                # Output to media directory for later
                print(f"NOTE: Saving {len(annotations)} annotations to {media_dir}")
                annotations.to_csv(f"{media_dir}/annotations.csv")

                if os.path.exists(f"{media_dir}/annotations.csv"):
                    print("NOTE: Annotations saved successfully")
                else:
                    raise Exception

                # Plot data summaries
                plot_distributions(annotations, media_dir, media_name)

        except Exception as e:
            print(f"ERROR: Could not finish collecting media {media_id} from TATOR")
            print(f"ERROR: {e}")
            traceback.print_exc()


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Download")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int,
                        help="Project ID for desired media")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--label_map", type=str,
                        default=f"{os.path.abspath('../../../Data/benthic_label_map.json')}",
                        help="Path to label map file")

    parser.add_argument("--output_dir", type=str,
                        default=f"{os.path.abspath('../../../Data/Ground_Truth/')}",
                        help="Path to the output directory.")

    args = parser.parse_args()

    try:
        download(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
