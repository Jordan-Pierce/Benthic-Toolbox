import os
import sys
import requests

import cv2
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from segment_anything import SamPredictor
from segment_anything import sam_model_registry


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, bboxes):
        self.data = bboxes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def download_checkpoint(url, path):
    """

    """
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                # Write the content to the file
                file.write(response.content)
            print(f"NOTE: Downloaded file successfully")
            print(f"NOTE: Saved file to {path}")
        else:
            print(f"ERROR: Failed to download file. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred: {e}")


def get_sam_predictor(model_type="vit_l", device='cpu'):
    """

    """
    # URL to download pre-trained weights
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/"

    # The path containing the weights
    sam_root = f".\\Weights"
    os.makedirs(sam_root, exist_ok=True)

    # Mapping between the model type, and the checkpoint file name
    sam_dict = {"vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth"}

    if model_type not in list(sam_dict.keys()):
        print(f"ERROR: Invalid model type provided; choices are:\n{list(sam_dict.keys())}")
        sys.exit(1)

    # Checkpoint path to model
    path = f"{sam_root}\\{sam_dict[model_type]}"

    # Check to see if the weights of the model type were already downloaded
    if not os.path.exists(path):
        print("NOTE: SAM model checkpoint does not exist; downloading")
        url = f"{sam_url}{sam_dict[model_type]}"
        # Download the file
        download_checkpoint(url, path)

    # Loading the mode, returning the predictor
    sam_model = sam_model_registry[model_type](checkpoint=path)
    sam_model.to(device=device)
    sam_predictor = SamPredictor(sam_model)

    return sam_predictor


def get_segments(sam_predictor, image, bboxes):
    """
    :param image:
    :param bboxes:
    :return:
    """

    # To hold the segmentations as polygons and masks
    masks = []
    segmentations = []

    # Set the image in sam predictor
    sam_predictor.set_image(image)

    # Create into a tensor
    transformed_boxes = torch.tensor(bboxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(transformed_boxes, image.shape[:2])

    # Create a data loader containing the transformed boxes
    custom_dataset = CustomDataset(transformed_boxes)
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

    # Loop through batches of boxes, faster
    for batch_idx, batch in enumerate(data_loader):

        try:
            # After setting the current image, get masks for each point / bbox
            mask, _, _ = sam_predictor.predict_torch(point_coords=None,
                                                     point_labels=None,
                                                     boxes=batch,
                                                     multimask_output=False)

            # Convert the mask to numpy, cpu
            mask = mask.cpu().numpy().squeeze()

            # Find the contours of the mask
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            # Get the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the segmentation mask for object
            segmentation = largest_contour.flatten().tolist()
            segmentation = np.array(segmentation).reshape(-1, 2)

            # Store masks, segmentations
            masks.append(mask)
            segmentations.append(segmentation)

        except Exception as e:
            print(f"ERROR: SAM model could not make predictions\n{e}")
            masks = []
            segmentations = []

    return np.array(masks), np.array(segmentations)
