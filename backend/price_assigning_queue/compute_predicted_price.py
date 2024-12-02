

import torch

from matplotlib import pyplot as plt
from model.google_vit_model import get_vit_model
from helpers.data_loader import process_sample

from helpers.processed_data import processed_data

from config.settings import AGGREGATION_METHOD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images, prices = processed_data(535)

model, feature_extractor = get_vit_model(aggregation_method=AGGREGATION_METHOD)

model.load_state_dict(torch.load("models/vit_regression_model.pth"))


def compute_predicted_price(images):
    processed_sample = process_sample(images, feature_extractor)

    prediction = model(processed_sample.to(device))

    return prediction.item()


