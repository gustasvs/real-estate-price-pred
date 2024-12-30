import torch
import numpy as np

from matplotlib import pyplot as plt
from model.google_vit_model import get_vit_model
from helpers.data_loader import process_sample_images

from config.settings import AGGREGATION_METHOD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, feature_extractor = get_vit_model(aggregation_method=AGGREGATION_METHOD)

model.load_state_dict(torch.load("models/vit_regression_model.pth"))

model.to(device)


def compute_predicted_price(images, metadata):

    processed_sample_images = process_sample_images(images, feature_extractor)

    instance = [processed_sample_images, np.array(metadata)]

    sample = [
        [
            instance[0].to(device, dtype=torch.float32),  # Assuming instance[0] is already a tensor
            torch.tensor(instance[1], device=device, dtype=torch.float32) if isinstance(instance[1], np.ndarray) else instance[1].to(device, dtype=torch.float32)
        ]
    ]

    prediction = model(sample)

    return prediction.item()


