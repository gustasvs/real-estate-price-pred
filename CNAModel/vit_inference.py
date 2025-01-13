
import torch
import numpy as np

from PIL import Image
from PIL import ImageTk

from matplotlib import pyplot as plt
from model.google_vit_model import get_vit_model
from helpers.data_loader import get_data_loaders

from helpers.processed_data import processed_data

from config.settings import AGGREGATION_METHOD, SAMPLES_TO_USE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images, prices = processed_data(SAMPLES_TO_USE)

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
        for sample, prices in val_loader:
                sample = [
                    [
                        instance[0].to(
                            device, dtype=torch.float32
                        ),  # Assuming instance[0] is already a tensor
                        (
                            torch.tensor(
                                instance[1], device=device, dtype=torch.float32
                            )
                            if isinstance(instance[1], np.ndarray)
                            else instance[1].to(device, dtype=torch.float32)
                        ),
                    ]
                    for instance in sample
                ]

                outputs = model(sample)  # Now passing the correct structured sample

                for batch_idx, (instances, predicted, actual) in enumerate(
                    zip(sample, outputs, prices)
                ):
                    n_images = len(
                        instances[0]
                    )
                    # plt.figure(figsize=(n_images * 5, 5))
                    batch_images = []
                    
                    for idx, img_tensor in enumerate(instances[0]):
                        
                        img_tensor.clamp_(0, 1)  # Clamp to [0, 1]
                        img_tensor = img_tensor.cpu().squeeze()
                        if img_tensor.dim() == 3:
                            img_tensor = img_tensor.permute(
                                1, 2, 0
                            )  # Permute to (H, W, C) for imshow

                        # img_tensor = (
                        #     img_tensor + 1
                        # ) / 2  # Rescale from [-1, 1] to [0, 1]

                        img_tensor = Image.fromarray((img_tensor.numpy() * 255).astype(np.uint8))
                        batch_images.append(img_tensor)
                        # batch_images.append(tensor_to_pil(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))


                    image_samples.append(batch_images)

                    # Populate the global lists with the necessary data
                    predicted_prices.append(predicted.item())
                    actual_prices.append(actual.item())

                    print("Predicted: ", predicted.item())
                    print("Actual: ", actual.item())
                    print("-" * 20)
                    

calculate_predictions(val_loader, model, device)

visualise_results(image_samples, predicted_prices, actual_prices)
