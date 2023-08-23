import os
import sys
import argparse

import tator
import pandas as pd


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def upload(args):
    """
    :param args:
    :return:
    """
    print("\n###############################################")
    print("Upload")
    print("###############################################\n")

    try:
        # Setting the api given the token, authentication
        token = args.api_token
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        print(f"ERROR: Could not authenticate with provided API Token\n{e}")
        sys.exit(1)

    if os.path.exists(args.annotations):
        annotations = pd.read_csv(args.annotations, index_col=0)
    else:
        print(f"ERROR: Annotation file provided does not exist; please check input")
        sys.exit(1)

    # Pass the project and media list to variables
    project_id = args.project_id
    media_id = args.media_id

    # Media
    Media = api.get_media(media_id)
    media_name = Media.name
    print(f"NOTE: Uploading {len(annotations)} localizations to {media_name}")

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

    # Loop through all the predictions, upload to media
    localizations = []

    for i, r in annotations.iterrows():
        # Tator format of bounding boxes
        x = float(r['xmin'] / Media.width)
        y = float(r['ymin'] / Media.height)
        w = float(r['xmax'] - r['xmin']) / Media.width
        h = float(r['ymax'] - r['ymin']) / Media.height

        # Based on localization 440 - Detection
        frame = r['Frame']
        score = r['Score']

        # TODO figure this one out
        label = r['Label']

        # Standard spec for this bounding box
        spec = {'media_id': media_id,
                'type': loc_type_id,
                'version': layer_type_id,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'frame': frame,
                'ScientificName': "",
                'CommonName': "",
                'Notes': "",
                'Needs Review': True,
                'Score': score}

        # Add to list
        localizations.append(spec)

    try:
        print(f"NOTE: Attempting to upload {len(localizations)} annotations to {media_name}")

        # Upload annotations to Tator
        num_locs = 0

        for r in tator.util.chunked_create(api.create_localization_list,
                                           project_id,
                                           body=localizations):
            num_locs += 1

        print(f"NOTE: Uploaded {num_locs} annotations to {media_name} successfully")

    except Exception as e:
        print(f"ERROR: Could not upload data\n{e}")

    print(".")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Upload")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int,
                        help="Project ID for desired media")

    parser.add_argument("--media_id", type=int,
                        help="ID for desired media")

    parser.add_argument("--annotations", type=str,
                        help="Path to the annotations, or predictions file")

    args = parser.parse_args()

    try:
        upload(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
