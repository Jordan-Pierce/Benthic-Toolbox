import os
import sys
import shutil
import argparse
import traceback
from tqdm import tqdm

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
    annotations['Scientific Name'].value_counts().plot(kind='bar')
    plt.title(f"{media_name}")
    plt.savefig(f"{media_dir}/Scientific Name.png")
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
    :param localizatins:
    :param tracks:
    :return:
    """
    print(f"NOTE: Merging track localizations and localizations")

    # Holds the localizations in the tracks
    track_localizations = []
    # Holds the standard version of localizations
    clean_localizations = []

    for track in tracks:
        # The attributes dict for the current track
        attributes = track.attributes
        attributes['track_id'] = track.id
        for localization in track.localizations:
            # Assign the localization id, add attributes
            t = {'id': localization}
            t.update(attributes)
            track_localizations.append(t)

    for track_localization in track_localizations:
        # Find the localization that corresponds to the track localization by id
        localization = [l for l in localizations if l.id == track_localization['id']]

        # If there is a localization (there should be)
        if localization:
            # Update the localization with attributes from track localization
            clean_localization = localization[0]
            clean_localization.attributes.update(track_localization)
            # Save in cleaned list
            clean_localizations.append(clean_localization)

    return localizations


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

    # Loop through medias
    for media_id in media_ids:

        try:
            # Name used for output
            Media = api.get_media(media_id)
            media_name = Media.name.replace(":", "__").split(".")[0]
            media_dir = f"{output_dir}/{media_name}/"
            os.makedirs(media_dir, exist_ok=True)

            print(f"NOTE: Media ID {media_id} corresponds to {Media.name}")

            # Get all localizations, filter for those that are actually bounding boxes
            localizations = api.get_localization_list(project_id, media_id=[media_id])
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
            for f_idx, frame in enumerate(list(set(frames))):

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
                    frame = frame_localization.frame
                    localization_id = frame_localization.id

                    # For some reason, class categories are different...
                    if 'Scientific Name' in frame_localization.attributes:
                        s = 'Scientific Name'
                    elif 'ScientificName' in frame_localization.attributes:
                        s = 'ScientificName'
                    else:
                        s = ""

                    # Get the actual scientific name
                    if s in frame_localization.attributes:
                        scientific = frame_localization.attributes[s]
                    else:
                        print(f"WARNING: Frame {frame} localization {localization_id} has no label!")
                        scientific = "Unlabeled"

                    # Row in dataframe
                    annotation = [
                        media_name,
                        os.path.basename(paths[f_idx]),
                        paths[f_idx],
                        frame,
                        Media.width,
                        Media.height,
                        scientific,
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                    ]

                    # Add to list
                    annotations.append(annotation)

            # Pandas dataframe
            annotations = pd.DataFrame(annotations, columns=['Media', 'Image Name', 'Image Path',
                                                             'Frame', 'Width', 'Height', 'Scientific Name',
                                                             'xmin', 'ymin', 'xmax', 'ymax'])
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
            print(f"ERROR: Could not finish collecting media {media_id} from Tator")
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

    parser.add_argument("--human_made", type=bool, default=True,
                        help="To download only ground truth by humans")

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
