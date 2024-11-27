
import torch

from matplotlib import pyplot as plt
from model.google_vit_model import get_vit_model
from helpers.data_loader import get_data_loaders

from helpers.processed_data import processed_data

from config.settings import AGGREGATION_METHOD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images, prices = processed_data(535)

model, feature_extractor = get_vit_model(aggregation_method=AGGREGATION_METHOD)

model.load_state_dict(torch.load("models/vit_regression_model.pth"))

model.to(device)

train_loader, val_loader = get_data_loaders(images, prices, feature_extractor)

predicted_prices = []
actual_prices = []
image_samples = []

from visualisation_gui import visualise_results, tensor_to_pil

def calculate_predictions(val_loader, model, device):
    model.eval()
    with torch.no_grad():
        for pixel_values_list, prices in val_loader:
            outputs = model([instance.to(device) for instance in pixel_values_list])
            for batch, predicted, actual in zip(pixel_values_list, outputs, prices):
                # Populate the global lists with the necessary data
                predicted_prices.append(predicted.item())
                actual_prices.append(actual.item())

                print("Predicted: ", predicted.item())
                print("Actual: ", actual.item())
                print("-" * 20)
                # Convert tensors to PIL images and add to image_samples
                # image_samples.append([tensor_to_pil(img_tensor) for img_tensor in batch])
                image_samples.append([tensor_to_pil(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for img_tensor in batch])

calculate_predictions(val_loader, model, device)

visualise_results(image_samples, predicted_prices, actual_prices)
