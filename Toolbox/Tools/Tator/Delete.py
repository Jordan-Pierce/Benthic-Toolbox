import os
import sys
import argparse

import time
import tator


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def delete(args):
    """

    :param args:
    :return:
    """

    print("\n###############################################")
    print("Detector")
    print("###############################################\n")

    try:
        # Setting the api given the token, authentication
        token = args.api_token
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        print(f"ERROR: Could not authenticate with provided API Token\n{e}")
        sys.exit(1)

    # Project id containing media
    if args.project_id:
        project_id = args.project_id
    else:
        print(f"ERROR: Project ID provided is invalid; please check input")
        sys.exit(1)

    # List of media
    if args.media_ids:
        media_ids = args.media_ids
    else:
        print(f"ERROR: Medias provided is invalid; please check input")
        sys.exit(1)

    try:
        # The type of localization for the project (bounding box, attributes)
        loc_type_id = 440  # Detection Box
        loc_name = api.get_localization_type(loc_type_id).name
        layer_type_id = 465  # AI Experiments
        layer_name = api.get_version(layer_type_id).name
    except Exception as e:
        print(f"ERROR: Could not find the correct localization type in project {project_id}")
        sys.exit(1)

    # Loop through medias
    for media_id in media_ids:

        # Get the media from tator
        media = api.get_media(media_id)
        print(f"NOTE: Media ID {media_id} corresponds to {media.name}")

        # Get the localizations that are Detections, in the layer AI Experiments
        localizations = api.get_localization_list(project=project_id,
                                                  media_id=[media_id],
                                                  type=loc_type_id,
                                                  version=[layer_type_id])

        print(f"NOTE: Found {len(localizations)} localizations for {loc_name} layer {layer_name}")

        # Don't delete all, just those that Need Review
        if not args.delete_all:
            # Grabs the localizations that Need Review, leaving the others behind
            localizations = [l for l in localizations if l.attributes['Needs Review']]

        print(f"NOTE: Targeting {len(localizations)} localizations for {loc_name} type - layer {layer_name}")

        try:
            print(f"NOTE: Deleting localizations in 60 seconds...")
            time.sleep(60)

            # Get just the ids of target locs
            ids = [l.id for l in localizations]

            # Burn baby burn
            response = api.delete_localization_list(project=project_id,
                                                    media_id=[media_id],
                                                    localization_bulk_delete={"ids": ids})
            print(f"NOTE: {response.message}")

        except Exception as e:
            print(f"ERROR: {e}")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Detector")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int,
                        help="Project ID for desired media")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--delete_all", action='store_true',
                        help="Delete all detections (not just 'Needs Review')")

    args = parser.parse_args()

    try:
        delete(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
