import os
import sys
import time
import argparse
from tqdm import tqdm

import tator
import numpy as np


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def delete(args):
    """

    :param args:
    :return:
    """

    print("\n###############################################")
    print("Delete")
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

        if not args.start_frame:
            start_frame = 0
        else:
            start_frame = args.start_frame

        if not args.end_frame:
            end_frame = media.num_frames
        else:
            end_frame = args.end_frame

        # Identify target tracks, localizations
        target_tracks = []
        target_localizations = []
        # Represents the first and last frame with modifications
        # A track may extend pass the end frame, so it's not counted
        first_frame = start_frame
        last_frame = start_frame

        print(f"NOTE: Searching for tracks and localizations within frames [{start_frame, end_frame}]")

        # Loop through all the tracks
        for track in tqdm(tracks):
            # Create a dictionary to map localization IDs to frames
            id_to_frame = {loc.id: loc.frame for loc in localizations}
            # Get all the localizations in the current track
            track_locs = np.array([l for l in track.localizations])
            # From the track localizations, get all the frame IDs efficiently
            track_frames = np.array([id_to_frame[loc] for loc in track_locs if loc in id_to_frame])
            # If the track contains localizations in frames within the window, add as target
            if np.any((track_frames >= start_frame) & (track_frames <= end_frame)):
                target_tracks.append(track)
                # Then add all the localizations for the target track
                target_localizations.extend([l for l in localizations if l.id in track_locs])
                # Update the frame window representing where modifications occurred
                if last_frame < min(track_frames):
                    last_frame = min(track_frames)
                if first_frame >= max(track_frames):
                    first_frame = max(track_frames)

        print(f"NOTE: Found {len(target_tracks)} tracks and {len(target_localizations)} "
              f"within frames [{first_frame}, {last_frame}]")

        try:

            if target_localizations or target_localizations:
                print(f"NOTE: Deleting {len(target_tracks)} tracks and {len(target_localizations)} localizations "
                      f"in 15 seconds...")
                time.sleep(15)

            if target_tracks:
                # Get the ids for the target tracks
                ids = [t.id for t in target_tracks]

                # Burn baby burn
                response = api.delete_state_list(project=project_id,
                                                 media_id=[media_id],
                                                 type=state_type_id,
                                                 version=[layer_type_id],
                                                 state_bulk_delete={"ids": ids})

                print(f"NOTE: {response.message}")

            if target_localizations:
                # Get the ids for the target localizations
                ids = [l.id for l in target_localizations]

                # Burn baby burn
                response = api.delete_localization_list(project=project_id,
                                                        media_id=[media_id],
                                                        type=loc_type_id,
                                                        version=[layer_type_id],
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

    parser = argparse.ArgumentParser(description="Delete")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int,
                        help="Project ID for desired media")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--start_frame", type=int,
                        help="Start frame to propagate")

    parser.add_argument("--end_frame", type=int,
                        help="End frame to propagate")

    args = parser.parse_args()

    try:
        delete(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
