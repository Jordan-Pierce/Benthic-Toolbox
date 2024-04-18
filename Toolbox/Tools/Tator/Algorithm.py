import os
import sys
import glob
import json
import datetime
import argparse
import traceback
from tqdm import tqdm


import tator
import random
import numpy as np
import pandas as pd

import mmcv
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmyolo.registry import VISUALIZERS

from mmengine.utils import track_iter_progress

from boxmot import BYTETracker

from Toolbox.Tools.SAM import *

import warnings

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def get_now():
    """
    :return:
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    return now


def get_color(index):
    """
    :param index:
    :return:
    """
    # Set the random seed to ensure consistent colors for the same index
    random.seed((42, index))

    # Generate random RGB values
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    return (red, green, blue)


def get_tracks(frame, scores, bboxes, labels, tracker):
    """
    :param scores:
    :param bboxes:
    :param labels:
    :param tracker:
    :return:
    """
    # Input to tracker has to be N X (x, y, x, y, scores, labels)
    detections = np.concatenate((bboxes, scores[:, np.newaxis], labels[:, np.newaxis]), axis=1)

    # Pass to tracker to update
    tracks = tracker.update(detections, frame)

    if len(tracks):
        # Return as N X (x, y, x, y, id, scores, labels, indices of true detections)
        scores = tracks[:, 5]
        bboxes = tracks[:, 0:4].astype('int')
        labels = tracks[:, 6].astype('int')
        track_ids = tracks[:, 4].astype('int')
    else:
        # Nothing was tracked, return Nones
        scores = np.array([])
        bboxes = np.array([])
        labels = np.array([])
        track_ids = np.array([])

    return scores, bboxes, labels, track_ids, tracker


def get_num_tracks(tracker):
    """
    :param tracker:
    :return:
    """
    num_tracks = len(tracker.tracked_stracks) + \
                 len(tracker.removed_stracks) + \
                 len(tracker.lost_stracks)

    return num_tracks


def algorithm(args):
    """
    :param args:
    :return:
    """
    print("\n###############################################")
    print("Algorithm")
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

    # List of media
    if args.media_ids:
        media_ids = args.media_ids
    else:
        print(f"ERROR: Medias provided is invalid; please check input")
        sys.exit(1)

    # Create run directory
    if os.path.exists(args.run_dir):
        run_dir = f"{args.run_dir}\\"
        run_name = os.path.dirname(run_dir).split("\\")[-1]
        print(f"NOTE: Using run {run_name}")
    else:
        print(f"ERROR: Run directory doesn't exist; please check input")
        sys.exit(1)

    try:
        # Find the config file
        config = glob.glob(f"{run_dir}*.py")[0]
        config_name = os.path.basename(config).split(".")[0]
        print(f"NOTE: Using config file {config_name}")
    except:
        print(f"ERROR: Config file doesn't exist; please check input")
        sys.exit(1)

    try:
        # Find the checkpoint file
        checkpoints = glob.glob(f"{run_dir}*.pth")
        checkpoint = [c for c in checkpoints if os.path.basename(c) == f'epoch_{args.epoch}.pth'][0]
        checkpoint_name = os.path.basename(checkpoint).split(".")[0]
        print(f"NOTE: Using checkpoint file {checkpoint_name}")
    except:
        print(f"ERROR: Checkpoint file doesn't exist; please check input")
        sys.exit(1)

    try:
        # Find the class map json file
        class_map = f"{run_dir}class_map.json"

        with open(class_map, 'r') as input_file:
            class_map = json.load(input_file)

        # Update class map format
        class_map = {d['id']: d['name'] for d in class_map}
    except:
        print(f"ERROR: Class Map JSON file provided doesn't exist; please check input")
        sys.exit(1)

    # Root where all output is saved
    output_dir = f"{args.output_dir}\\"

    # ------------------------------------------------
    # Model config
    # ------------------------------------------------
    print(f"NOTE: Setting up model...")

    # Either GPU or CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"NOTE: Using device {device}")

    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, palette='random', device=device)
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    # Create the local visualizer from cfg
    visualizer = VISUALIZERS.build(model.cfg.visualizer)

    classes = visualizer.dataset_meta['class_label'] = model.cfg.train_dataloader.dataset.metainfo['classes']
    colors = model.cfg.train_dataloader.dataset.metainfo['palette']
    colors = [tuple(c) for c in colors]

    # ------------------------------------------------
    # Track
    # ------------------------------------------------
    if args.track:
        # Create tracker object
        print("NOTE: Tracking enabled")
        tracker = BYTETracker(track_thresh=args.pred_threshold * 1.1,
                              match_thresh=.9,
                              frame_rate=30 // args.every_n)

    # ------------------------------------------------
    # Segment
    # ------------------------------------------------
    if args.segment:
        # Create SAM predictor object
        print("NOTE: Segmenting enabled")
        sam_predictor = get_sam_predictor("vit_l", device)

    # Loop through medias
    for media_id in media_ids:

        # For upload
        localizations = []
        # For local archive
        predictions = []

        # ------------------------------------------------
        # Download media
        # ------------------------------------------------
        try:
            # Get the video handler
            media = api.get_media(media_id)
            ext = media.name.split(".")[-1]
            output_media_dir = f"{output_dir}Algorithm_{get_now()}_{media_id}\\"
            output_video_path = f"{output_media_dir}{media_id}.{ext}"
            os.makedirs(output_media_dir, exist_ok=True)
            print(f"NOTE: Downloading {media.name}...")
        except Exception as e:
            print(f"ERROR: Could not get media from {media_id}")
            sys.exit(1)

        # Download the video
        for progress in tator.util.download_media(api, media, output_video_path):
            print(f"NOTE: Download progress: {progress}%")

        # Check that video was downloaded
        if os.path.exists(output_video_path):
            print(f"NOTE: Media {media.name} downloaded successfully")
        else:
            print(f"ERROR: Media {media.name} did not download successfully; skipping")
            continue

        # Do inference on each video.
        print(f"NOTE: Doing inference on {media.name}")

        # Create the video handler
        video_reader = mmcv.VideoReader(output_video_path)
        video_writer = None

        # Start at
        if args.start_at:
            start_at = args.start_at
        else:
            start_at = 0

        # End at
        if args.end_at:
            end_at = args.end_at
        else:
            end_at = video_reader.frame_cnt

        if args.show_video:
            # Create a path for the predictions video
            output_pred_path = output_video_path.split(".")[0] + "_algorithm.mp4"
            # To output the video with predictions super-imposed
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_pred_path, fourcc, video_reader.fps,
                                           (video_reader.width, video_reader.height))

        # Loop through all the frames in the video
        for f_idx, frame in enumerate(track_iter_progress(video_reader)):

            # Make predictions on every N frames within window
            if f_idx % args.every_n == 0 and start_at <= f_idx <= end_at:

                # Make predictions
                result = inference_detector(model, frame, test_pipeline=test_pipeline)

                # Parse out predictions
                scores = result.pred_instances.cpu().detach().scores.numpy()
                bboxes = result.pred_instances.cpu().detach().bboxes.numpy()
                labels = result.pred_instances.cpu().detach().labels.numpy()

                # Filter based on threshold
                indices = np.where(scores >= args.pred_threshold)[0]
                scores = scores[indices]
                bboxes = bboxes[indices]
                labels = labels[indices]

                # Placeholders if not tracking, segmenting
                tracks = np.array([0] * len(indices))
                masks = np.array([])
                segmentations = np.array([])

                if args.track:
                    # Get the updates results from tracker
                    scores, bboxes, labels, tracks, tracker = get_tracks(frame,
                                                                         scores,
                                                                         bboxes,
                                                                         labels,
                                                                         tracker)
                if args.segment:
                    # Get the masks for each bbox
                    masks, segmentations = get_segments(sam_predictor,
                                                        frame,
                                                        bboxes)

                # Record the predictions in tator format
                for i_idx in range(len(bboxes)):
                    # Score, Label for tator
                    score = round(float(scores[i_idx]), 3)
                    label = class_map[labels[i_idx]]
                    # Update this with multi-category
                    scientific = "Not Set"

                    # Local archive format
                    xmin = int(bboxes[i_idx][0])
                    ymin = int(bboxes[i_idx][1])
                    xmax = int(bboxes[i_idx][2])
                    ymax = int(bboxes[i_idx][3])

                    # Tator format of bounding boxes
                    x = float(max(0.0, min(1.0, xmin / video_reader.width)))
                    y = float(max(0.0, min(1.0, ymin / video_reader.height)))
                    w = float(max(0.0, min(1.0, (xmax - xmin) / video_reader.width)))
                    h = float(max(0.0, min(1.0, (ymax - ymin) / video_reader.height)))

                    # For tator upload
                    loc = {'media_id': media.id,
                           'type': loc_type_id,
                           'version': layer_type_id,
                           'x': x,
                           'y': y,
                           'width': w,
                           'height': h,
                           'frame': f_idx,
                           'track_id': int(tracks[i_idx]),
                           'attributes': {
                               'ScientificName': scientific,
                               'CommonName': "",
                               'Notes': "",
                               'Needs Review': True,
                               'Score': score}
                           }
                    # For local archive
                    pred = [f_idx, label, score, xmin, ymin, xmax, ymax]

                    # Add to lists
                    localizations.append(loc)
                    predictions.append(pred)

                if args.show_video:
                    # Create a new 'result' after filtering, tracking, segmenting
                    # (Only used for local visualizations)

                    if args.track:
                        # Stash tracked object IDs in visualizer metadata
                        # (Only used for local visualizations)
                        tracks -= 1

                        print(f" Tracking: {[i for i in tracks]}")
                        num_tracks = get_num_tracks(tracker)
                        object_tracks = np.arange(0, num_tracks)
                        object_tracks = [f'OBJ {i}' for i in object_tracks]
                        object_colors = [get_color(i) for i in object_tracks]

                        # Pass to the visualizer
                        visualizer.dataset_meta['classes'] = object_tracks
                        visualizer.dataset_meta['palette'] = object_colors

                    # bbox predictions
                    pred_instances = InstanceData(metainfo=result.metainfo)
                    pred_instances.bboxes = torch.from_numpy(bboxes).to(device)
                    pred_instances.scores = torch.from_numpy(scores).to(device)
                    pred_instances.labels = torch.from_numpy(labels).to(device)

                    if args.segment:
                        pred_instances.masks = torch.from_numpy(masks).to(device)

                    result = DetDataSample(pred_instances=pred_instances)

                    try:
                        # Add result to frame
                        visualizer.add_datasample(
                            name='video',
                            image=frame,
                            data_sample=result,
                            draw_gt=False,
                            show=False,
                            wait_time=0.01,
                            pred_score_thr=args.pred_threshold)
                    except:
                        pass

                    # Get the updated frame with visuals
                    frame = visualizer.get_image()

                    # Display predictions as they are happening
                    cv2.namedWindow('video', 0)
                    mmcv.imshow(frame, 'video', 1)

                    # Write the frame to video file
                    video_writer.write(frame)

            # Exit the frame loop
            if f_idx >= end_at:
                break

        if args.show_video:
            # Release the video handler
            video_writer.release()
            video_writer = None

            # Close the viewing window
            cv2.destroyAllWindows()

        try:
            # Close the video writer
            video_reader = None
            # Delete the original video
            os.remove(output_video_path)
        except Exception as e:
            print(f"WARNING: Could not delete media\n{e}")

        # ------------------------------------------------
        # Save and Upload localizations for this media
        # ------------------------------------------------

        # Save merged annotations as "predictions.csv"
        predictions_path = f"{output_media_dir}predictions.csv"
        predictions = pd.DataFrame(predictions, columns=['Frame', 'Label', 'Score', 'xmin', 'ymin', 'xmax', 'ymax'])
        predictions.to_csv(predictions_path)
        print(f"NOTE: Merged predictions saved to {os.path.basename(predictions_path)}")

        if args.upload:
            # Create the localizations in the video.
            print(f"NOTE: Uploading {len(localizations)} detections on {media.name}...")

            # Specify the batch size
            batch_size = 500

            # Calculate the number of batches needed
            num_batches = int(np.ceil(len(localizations) / batch_size))

            # Initialize a list to store all localization IDs
            localization_ids = []

            # Outer loop to upload batches of localizations
            for batch_num in tqdm(range(num_batches)):
                # Calculate the start and end indices for the current batch
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, len(localizations))

                # Extract the current batch of localizations
                batch = localizations[start_index:end_index]

                try:

                    # Upload the current batch of localizations
                    for response in tator.util.chunked_create(api.create_localization_list, project_id, body=batch):
                        localization_ids.extend(response.id)

                except Exception as e:
                    print(f"ERROR: {e}")

            print(f"NOTE: Successfully created {len(localization_ids)} localizations in total on {media.name}!")

        if args.upload and args.track:
            # Associate the localization ids with track ids
            tracks = [l['track_id'] for l in localizations]
            tracks = np.array(list(zip(tracks, localization_ids)))
            num_tracks = len(np.unique(tracks.T[0]))

            print(f"NOTE: Uploading {num_tracks} tracks on {media.name}...")
            states = []
            for track_id in np.unique(tracks.T[0]):
                state = {'type': state_type_id,
                         'version': layer_type_id,
                         'localization_ids': tracks.T[1][np.where(tracks.T[0] == track_id)].tolist(),
                         'media_ids': [media_id],
                         'attributes':
                             {'ScientificName': "Not Set",
                              'Needs Review': True,
                              'Notes': ""}
                         }

                states.append(state)

            # Specify the batch size
            batch_size = 500

            # Calculate the number of batches needed
            num_batches = int(np.ceil(len(states) / batch_size))

            # Create an empty list to store the state IDs for this batch
            state_ids = []

            # Outer loop to upload batches of states
            for batch_num in tqdm(range(num_batches)):
                # Calculate the start and end indices for the current batch
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, len(states))

                # Extract the current batch of states
                batch = states[start_index:end_index]

                # Upload the current batch of states
                for response in tator.util.chunked_create(api.create_state_list, project_id, body=batch):
                    state_ids.extend(response.id)

            # After the loop, you can work with `all_state_ids`, which contains all the state IDs from all batches.
            print(f"NOTE: Successfully created {len(state_ids)} tracks in total on {media.name}!")

    print(f"NOTE: Completed inference on {len(media_ids)} medias.")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Algorithm")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int,
                        help="Project ID for desired media")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory containing the run")

    parser.add_argument("--epoch", type=int, required=True,
                        help="Epoch N checkpoint to use")

    parser.add_argument("--every_n", type=int, default=30,
                        help="Make predictions on every N frames")

    parser.add_argument("--start_at", type=int, default=0,
                        help="Frame to start making predictions at")

    parser.add_argument("--end_at", type=int, default=0,
                        help="Frame to end making predictions at")

    parser.add_argument('--pred_threshold', type=float, default=0.3,
                        help='Prediction confidence threshold')

    parser.add_argument('--track', action='store_true',
                        help='Track objects')

    parser.add_argument('--segment', action='store_true',
                        help='Segment objects')

    parser.add_argument('--show_video', action='store_true',
                        help='Show video, and save it to the predictions directory')

    parser.add_argument('--upload', action='store_true',
                        help='Upload predictions to Tator')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.abspath("../../../Data/Predictions/"),
                        help='Directory to save output from model')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'

    try:
        algorithm(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
