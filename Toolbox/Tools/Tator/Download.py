import os
import sys
import shutil
import argparse
from tqdm import tqdm

import tator
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from concurrent.futures import ThreadPoolExecutor


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def plot_data(annotations, media_dir, media_name):
    """
    :param annotations:
    :param media_dir:
    :return:
    """

    # Output a data distribution chart
    annotations['ScientificName'].value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.title(f"{media_name}")
    plt.savefig(f"{media_dir}/ScientificName.png")
    plt.close()

    # Output a data distribution chart
    annotations['CommonName'].value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.title(f"{media_name}")
    plt.savefig(f"{media_dir}/CommonName.png")
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

            print(f"NOTE: Collecting annotated media from {media_name}")

            # Get all localizations
            localizations = api.get_localization_list(project_id, media_id=[media_id])

            # Filter for bounding boxes
            localizations = [l for l in localizations if 'ScientificName' in l.attributes]

            # Filter for ground truth, or else all
            if args.human_made:
                print(f"NOTE: Collecting human-made annotated media from {media_name}")
                localizations = [l for l in localizations if l.attributes['ID Analyst'] != 'Algorithm']

            # Get the frames associate with localizations
            frames = [l.frame for l in localizations]

            if not frames or not localizations:
                print("NOTE: No annotated frames for this media")
                continue

            # Download the associated frame
            with ThreadPoolExecutor(max_workers=100) as executor:
                paths = [executor.submit(download_image,
                                         api,
                                         media_id,
                                         frame,
                                         media_dir) for frame in frames]

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

                    # Labels
                    scientific = frame_localization.attributes['ScientificName']
                    common = frame_localization.attributes['CommonName']

                    # Convert to COCO format
                    xmin = int(x * Media.width)
                    ymin = int(y * Media.height)
                    xmax = int(w * Media.width) + xmin
                    ymax = int(h * Media.height) + ymin

                    # Row in dataframe
                    annotation = [
                        media_name,
                        paths[f_idx],
                        frame,
                        scientific,
                        common,
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                    ]

                    # Add to list
                    annotations.append(annotation)

            # Pandas dataframe
            annotations = pd.DataFrame(annotations, columns=['Media', 'Image', 'Frame',
                                                             'ScientificName', 'CommonName',
                                                             'xmin', 'ymin', 'xmax', 'ymax'])
            # Output to media directory for later
            annotations.to_csv(f"{media_dir}/annotations.csv")

            # Plot data summaries
            plot_data(annotations, media_dir, media_name)

        except Exception as e:
            print(f"ERROR: Could not collect media {media_id} from Tator\n{e}")


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
