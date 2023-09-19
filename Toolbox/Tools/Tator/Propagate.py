import os
import sys
import argparse

import tator


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def get_localizations(localizations, tracks):
    """
    :param localizations:
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


def propagate(args):
    """

    :param args:
    :return:
    """

    print("\n###############################################")
    print("propagate")
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
        layer_type_id = 228  # AI Experiments
        layer_name = api.get_version(layer_type_id).name
        state_type_id = 438  # Tracks for Detections
        state_name = api.get_state_type(state_type_id).name
    except Exception as e:
        print(f"ERROR: Could not find the correct localization type in project {project_id}")
        sys.exit(1)

    # Loop through medias
    for media_id in media_ids:

        # Get the media from tator
        media = api.get_media(media_id)
        print(f"NOTE: Media ID {media_id} corresponds to {media.name}")

        # Get the tracks that are in the layer AI experiments
        tracks = api.get_state_list(project=project_id,
                                    media_id=[media_id],
                                    type=state_type_id,
                                    version=[layer_type_id])

        print(f"NOTE: Found {len(tracks)} tracks for {state_name} layer {layer_name}")

        # Get the localizations that are Detections, in the layer AI Experiments
        localizations = api.get_localization_list(project=project_id,
                                                  media_id=[media_id],
                                                  type=loc_type_id,
                                                  version=[layer_type_id])

        print(f"NOTE: Found {len(localizations)} localizations for {loc_name} layer {layer_name}")

        # Identify target tracks
        target_tracks = []

        for track in tracks:
            if not track.attributes['Needs Review'] and track.attributes['ScientificName'] not in ['', 'Not Set']:
                target_tracks.append(track)

        print(f"NOTE: Found {len(target_tracks)} tracks with labels to propagate")

        # Loop through each of these, change the label with the associate localizations
        for track in target_tracks:

            try:

                response = api.update_localization_list(
                    project=project_id,
                    localization_bulk_update={
                        "ids": track.localizations,
                        "attributes": {"Needs Review": False, "ScientificName": track.attributes["ScientificName"]}
                    }
                )

                print(f"NOTE: Track ID {track.id} - {response['message']}")

            except Exception as e:
                print(f"ERROR: Could not propagate track {track.id}'s label to associated bounding boxes.")
                print(f"ERROR: {response['message']}")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Propagate")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int,
                        help="Project ID for desired media")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    args = parser.parse_args()

    try:
        propagate(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
