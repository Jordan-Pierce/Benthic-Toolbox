import os
import sys
import glob
import json
import argparse
import traceback

import pandas as pd

import torch
from mmdet.apis import DetInferencer

import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def merge_predictions(prediction_paths, class_map, output_dir):
    """
    :param annotation_paths:
    :param class_map:
    :param output_directory:
    :return:
    """

    predictions = []

    for path in prediction_paths:
        with open(path, "r") as file:
            annotations = json.load(file)

        # Frame name
        frame = os.path.basename(path).split(".")[0]
        # Confidence Scores
        scores = [s for s in annotations['scores']]
        # Labels, mapped to class category (string)
        labels = [class_map[l] for l in annotations['labels']]

        # Get the bounding boxes, then parse
        bboxes = [bbox for bbox in annotations['bboxes']]
        xmin = [b[0] for b in bboxes]
        ymin = [b[1] for b in bboxes]
        xmax = [b[2] for b in bboxes]
        ymax = [b[3] for b in bboxes]

        for _ in range(len(scores)):
            predictions.append([frame, labels[_], scores[_], xmin[_], ymin[_], xmax[_], ymax[_]])

    # Save merged annotations as "predictions.csv"
    predictions_path = f"{output_dir}predictions.csv"
    predictions = pd.DataFrame(predictions, columns=['Frame', 'Label', 'Score', 'xmin', 'ymin', 'xmax', 'ymax'])

    if os.path.exists(predictions_path):
        previous_predictions = pd.read_csv(predictions_path, index_col=0)
        predictions = pd.concat((previous_predictions, predictions))

    predictions.to_csv(predictions_path)
    print(f"NOTE: Merged predictions saved to {os.path.basename(predictions_path)}")


def inference(args):
    """

    :param args:
    :return:
    """
    print("\n###############################################")
    print("Inference")
    print("###############################################\n")

    # Check that config file exists
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

    # Directory where media is stored
    if os.path.exists(args.media_dir):
        media_dir = args.media_dir
        media_name = os.path.basename(media_dir)
    else:
        print(f"ERROR: Media directory provided doesn't exist; please check input")
        sys.exit(1)

    # Collect the images from frames directory
    if os.path.exists(f"{media_dir}\\frames\\"):
        frame_dir = f"{media_dir}\\frames\\"
        frame_paths = glob.glob(f"{frame_dir}*.*")
        print(f"NOTE: Found {len(frame_paths)} frames in {media_name}")
    else:
        print(f"ERROR: Could not locate 'frames' directory in {media_dir}")
        sys.exit(1)

    # Output directory
    output_dir = f"{args.output_dir}\\{media_name}_{run_name}\\"
    os.makedirs(output_dir, exist_ok=True)

    # Either GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"NOTE: Using device {device}")

    try:
        # Load inference object
        inferencer = DetInferencer(model=config,
                                   weights=checkpoint,
                                   palette='coco',
                                   show_progress=True)

        print(f"NOTE: Weights {os.path.basename(checkpoint)} loaded successfully")

    except Exception as e:
        print(f"ERROR: Could not load model\n{e}")
        sys.exit(1)

    try:
        print(f"NOTE: Making predictions for {media_name}")
        # Dict of predictions, and visualization
        results = inferencer(frame_paths,
                             out_dir=output_dir,
                             no_save_pred=False,
                             pred_score_thr=args.pred_threshold)

        print(f"NOTE: Merging predictions")
        merge_predictions(glob.glob(f"{output_dir}preds\\*.json"), class_map, output_dir)

    except Exception as e:
        print(f"ERROR: Could not make prediction on {media_name}\n{e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory containing the run")

    parser.add_argument("--epoch", type=int, required=True,
                        help="Epoch checkpoint to use")

    parser.add_argument('--media_dir', type=str, required=True,
                        help='Directory predictions will be made on')

    parser.add_argument('--pred_threshold', type=float, default=0.25,
                        help='Prediction confidence threshold')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.abspath("../../../Data/Predictions/"),
                        help='Directory to save output from model')

    args = parser.parse_args()

    try:
        inference(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
