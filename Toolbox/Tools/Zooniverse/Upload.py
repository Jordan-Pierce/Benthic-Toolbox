import os
import sys
import shutil
import argparse
import traceback

import tator
import panoptes_client

import cv2
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def get_moving_frames(nav_data):
    """
    :param nav_data:
    :return:
    """
    frames = []

    for n_idx, n in enumerate(nav_data):
        # Get the heading diff
        curr_head = float(n.attributes['Heading'])
        prev_head = float(nav_data[n_idx - 1].attributes['Heading'])
        diff_head = abs(curr_head - prev_head)
        # Get the easting diff
        curr_east = float(n.attributes['Eastings'])
        prev_east = float(nav_data[n_idx - 1].attributes['Eastings'])
        diff_east = abs(curr_east - prev_east)
        # Get the northing diff
        curr_north = float(n.attributes['Northings'])
        prev_north = float(nav_data[n_idx - 1].attributes['Northings'])
        diff_north = abs(curr_north - prev_north)
        # Find frames that represent where the rov is moving
        if diff_head >= 1.0 or diff_east >= 3.0 or diff_north >= 3.0:
            frames.append(n.frame)

    return frames


def assess_image_quality(image_path, sharp_thresh=40, size_thresh=1):
    """
    :param image_path:
    :return:
    """
    # To determine quality of image
    quality = False

    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the Laplacian variance as a measure of image sharpness
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    # To catch issues with video outage
    if sharpness >= 200:
        sharpness = 0

    # Check image size
    image_size = os.path.getsize(image_path) / (1024 * 1024)

    # Apply threshold conditions (larger than 1mb)
    if sharpness >= sharp_thresh and image_size >= size_thresh:
        quality = True

    return quality


def download_image(api, media_id, frame, media_dir):
    """
    :param api:
    :param media:
    :param frame:
    :param media_dir:
    :return:
    """
    # Location media directory
    frame_dir = f"{media_dir}frames\\"
    os.makedirs(frame_dir, exist_ok=True)

    # Location of file
    path = f"{frame_dir}{str(frame)}.jpg"

    # If it doesn't already exist, download, move.
    if not os.path.exists(path):
        temp = api.get_frame(media_id, frames=[frame])
        shutil.move(temp, path)

    # Delete the image if it's of lower quality
    if not assess_image_quality(path):
        os.remove(path)

    return path


def upload(args):
    """
    :param args:
    :return:
    """

    print("\n###############################################")
    print("Upload to Zooniverse")
    print("###############################################\n")

    # Root data location
    output_dir = f"{args.output_dir}\\"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Login to panoptes
        panoptes_client.Panoptes.connect(username=args.username, password=args.password)
        print(f"NOTE: Authentication to Zooniverse successful for {args.username}")
    except:
        print("ERROR: Could not login to Panoptes")
        sys.exit(1)

    try:
        # Get access to the project
        project = panoptes_client.Project.find(id=args.zoon_project_id)
        print(f"NOTE: Connected to Zooniverse project '{project.title}' successfully")
    except:
        print(f"ERROR: Could not access project {args.zoon_project_id}")
        sys.exit(1)

    try:
        # Setting the api given the token, authentication
        token = args.api_token
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication to TATOR successful for {api.whoami().username}")

    except Exception as e:
        print(f"ERROR: Could not authenticate with provided API Token\n{e}")
        sys.exit(1)

    try:
        # The type of localization for the project (bounding box, attributes)
        tator_project_id = args.tator_project_id
        project_name = api.get_project(id=tator_project_id).name
        state_type_id = 288  # State Type (ROV)
        state_name = api.get_state_type(state_type_id).name
    except Exception as e:
        print(f"ERROR: Could not find the correct localization type in project {args.tator_project_id}")
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
            media = api.get_media(media_id)
            media_name = media.name
            media_dir = f"{output_dir}{media_id}\\"
            os.makedirs(media_dir, exist_ok=True)
            print(f"NOTE: Media ID {media_id} corresponds to {media_name}")

            # Get the frames associated with nav data
            nav_data = api.get_state_list(project=tator_project_id, media_id=[media_id], type=state_type_id)
            print(f"NOTE: Found {len(nav_data)} frames with nav data for media {media_name}")

            # Filter out frames where there is no movement from previous frame
            frames = get_moving_frames(nav_data)
            print(f"NOTE: Found {len(frames)} frames ({len(nav_data)}) with movement for media {media_name}")

            # Download the associated frames
            print(f"NOTE: Downloading {len(frames)} frames for {media_name}")
            with ThreadPoolExecutor(max_workers=80) as executor:
                # Submit the jobs
                paths = [executor.submit(download_image, api, media_id, frame, media_dir) for frame in frames]
                # Execute, store paths
                paths = [future.result() for future in tqdm(paths)]

            # Dataframe for output
            dataframe = []

            for p_idx, path in enumerate(paths):

                # Make sure the path exists
                if not os.path.exists(path):
                    continue

                # Add to dataframe
                dataframe.append([media_id,
                                  media_name,
                                  p_idx,
                                  os.path.basename(path),
                                  path,
                                  media.height,
                                  media.width])

            # Output a dataframe for later
            dataframe = pd.DataFrame(dataframe, columns=['Media ID', 'Media Name',
                                                         'Frame', 'Frame Name',
                                                         'Path', 'Height', 'Width'])
            # Save the dataframe
            dataframe.to_csv(f"{media_dir}\\frames.csv", index=False)
            print(f"NOTE: Downloaded {len(dataframe)} of {len(frames)} frames for {media_name}")

        except Exception as e:
            print(f"ERROR: Could not finish downloading media {media_id} from TATOR")
            traceback.print_exc()
            sys.exit(1)

        if args.upload:
            try:
                print(f"NOTE: Uploading media {media_name} to Zooniverse")
                # Create subject set, link to project
                subject_set = panoptes_client.SubjectSet()
                subject_set.links.project = project
                subject_set.display_name = str(media_id)
                subject_set.save()
                # Reload the project
                project.reload()

                # Convert the dataframe to a dict
                subject_dict = dataframe.to_dict(orient='records')
                # Create a new dictionary with 'Path' as keys and other values as values
                subject_meta = {d['Path']: {k: v for k, v in d.items() if k != 'Path'} for d in subject_dict}

                # Create subjects from the meta
                new_subjects = []

                for filename, metadata in tqdm(subject_meta.items()):
                    # Create the subject
                    subject = panoptes_client.Subject()
                    # Link subject to project
                    subject.links.project = project
                    subject.add_location(filename)
                    # Update meta
                    subject.metadata.update(metadata)
                    # Save
                    subject.save()
                    # Append
                    new_subjects.append(subject)

                # Add the list of subjects to set
                subject_set.add(new_subjects)
                # Save
                subject_set.save()
                project.save()

            except Exception as e:
                print(f"ERROR: Could not finish uploading media {media_id} to Zooniverse")
                traceback.print_exc()
                sys.exit(1)

            try:
                # Attaching the new subject set to all the active workflows
                workflow_ids = project.__dict__['raw']['links']['active_workflows']

                # If there are active workflows, link them to the next subject sets
                for workflow_id in tqdm(workflow_ids):
                    # Create Workflow object
                    workflow = panoptes_client.Workflow(workflow_id)
                    workflow_name = workflow.__dict__['raw']['display_name']
                    # Add the subject set created previously
                    print(f"\nNOTE: Adding subject set {subject_set.display_name} to workflow {workflow_name}")
                    workflow.add_subject_sets([subject_set])
                    # Save
                    workflow.save()
                    project.save()

            except Exception as e:
                print(f"ERROR: Could not link media {media_id} to project workflows")
                traceback.print_exc()
                sys.exit(1)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Upload to Zooniverse")

    parser.add_argument("--username", type=str,
                        default=os.getenv('ZOONIVERSE_USERNAME'),
                        help="Zooniverse username")

    parser.add_argument("--password", type=str,
                        default=os.getenv('ZOONIVERSE_PASSWORD'),
                        help="Zooniverse password")

    parser.add_argument("--zoon_project_id", type=int, default=21853,  # click-a-coral
                        help="Zooniverse project ID")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--tator_project_id", type=int, default=70,
                        help="Tator Project ID")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--upload", action='store_true',
                        help="Upload media to Zooniverse (debugging")

    parser.add_argument("--output_dir", type=str,
                        default=f"{os.path.abspath('../../../Data/Zooniverse/')}",
                        help="Path to the output directory.")

    args = parser.parse_args()

    try:
        upload(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
