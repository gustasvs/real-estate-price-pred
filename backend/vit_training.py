import numpy as np

import torch
from torch.optim import Adam

import matplotlib.pyplot as plt

import time

import transformers
transformers.logging.set_verbosity_error()

from config.settings import BATCH_SIZE, ENABLE_DEV_LOGS, LEARNING_RATE, LOSS_FUNCT, EPOCHS, BATCHES_TO_AGGREGATE

from model.google_vit_model import get_vit_model

from helpers.weighted_loss import compute_bin_weights, WeightedMSELoss

from helpers.data_loader import get_data_loaders

from helpers.processed_data import processed_data

model, feature_extractor = get_vit_model(aggregation_method="mean")

# Define the custom dataset

def r2_score(outputs, prices):
    # Calculate the total sum of squares
    total_variance = torch.sum((prices - prices.mean())**2)
    
    # Calculate the residual sum of squares
    residual_variance = torch.sum((prices - outputs)**2)
    
    # Compute R2 score
    r2 = 1 - (residual_variance / total_variance)
    
    return r2.item()

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0
    start_time = time.time()
    total_batches = len(dataloader)

    accuracies = []

    aggregated_outputs = []
    aggregated_prices = []
    # optimizer.zero_grad()

    for batch_idx, (sample, prices) in enumerate(dataloader, 1):

        sample = [instance.to(device) for instance in sample]

        if ENABLE_DEV_LOGS: print("one input in batch shape: ", sample[0].shape)  # Should print [8, 2, 3, 224, 224] for a batch size of 8

        prices = prices.to(device)

        if ENABLE_DEV_LOGS: print("prices shape (should be batch size): ", prices.shape)


        outputs = model(sample)

        # print("outputs: ", outputs)
        # print("prices: ", prices)

        loss = LOSS_FUNCT(outputs, prices)
        loss.backward()

        aggregated_outputs.append(outputs.to("cpu"))
        aggregated_prices.append(prices.to("cpu"))
        

        if (batch_idx + 1) % BATCHES_TO_AGGREGATE == 0 or batch_idx == total_batches:

            optimizer.step()
            optimizer.zero_grad()

            # print("aggregated_outputs: ", aggregated_outputs)
            # print("aggregated_prices: ", aggregated_prices)

            aggregated_outputs = torch.cat(aggregated_outputs)
            aggregated_prices = torch.cat(aggregated_prices)

            aggregated_accuracy = r2_score(aggregated_outputs, aggregated_prices)

            accuracies.append(aggregated_accuracy)
            # print("aggregated_outputs: ", aggregated_outputs)
            # print("aggregated_prices: ", aggregated_prices)
            # print("aggregated_accuracy: ", aggregated_accuracy)
            # print("*" * 20)

            aggregated_outputs = []
            aggregated_prices = []

        
            elapsed_time = time.time() - start_time
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] - Batch [{batch_idx}/{total_batches}]: "
                f"Elapsed Time = {elapsed_time:.2f}s"
                f" - Loss = {loss.item():.4f}"
                f" - R2 Score = {aggregated_accuracy:.4f}"
            )
            # print last output and real outputs
            # print("Last output: ", outputs[-1].item())
            # print("Real output: ", prices[-1].item())

    plt.plot(accuracies)
    plt.xlabel("Batch")
    plt.ylabel("R2 Score")
    plt.title("R2 Score vs Batch")
    plt.show()

    if total_batches > 0:
        return running_loss / total_batches
    else:
        return 0


def validate(model, dataloader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for pixel_values_list, prices in dataloader:
            pixel_values_list = [instance.to(device) for instance in pixel_values_list]

            prices = prices.to(device)
            
            outputs = model(pixel_values_list)
            loss = LOSS_FUNCT(outputs.squeeze(), prices)
            val_loss += loss.item()
    return val_loss / len(dataloader)

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Batch size: {BATCH_SIZE}, aggregation multiplier: {BATCHES_TO_AGGREGATE}")
print("-" * 20)
print(f"Simulated batch size: {BATCH_SIZE * BATCHES_TO_AGGREGATE}")
print("-" * 20)

model.to(device)

count = 535

images, prices = processed_data(count)

bins, bin_weights = compute_bin_weights(prices, num_bins=10)

plt.plot(bin_weights)
plt.xlabel("Price Bins")
plt.ylabel("Weight")
plt.title("Price Bins vs Weights")
plt.show()

LOSS_FUNCT = WeightedMSELoss(bins, bin_weights, device=device)


train_loader, val_loader = get_data_loaders(images, prices, feature_extractor)


optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
epochs = EPOCHS
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
    val_loss = validate(model, val_loader, device)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "models/vit_regression_model.pth")

# After training completion, for visualizing predictions
def show_predictions(val_loader, model, device):
    model.eval()
    with torch.no_grad():
        for pixel_values_list, prices in val_loader:
            outputs = model([instance.to(device) for instance in pixel_values_list])
            for batch_idx, (batch, predicted, actual) in enumerate(zip(pixel_values_list, outputs, prices)):
                n_images = len(batch)  # Get the number of images in the batch
                plt.figure(figsize=(n_images * 5, 5))  # Adjust figure size based on number of images
                for idx, img_tensor in enumerate(batch):
                    plt.subplot(1, n_images, idx + 1)
                    img_tensor = img_tensor.cpu().squeeze()
                    if img_tensor.dim() == 3:  # Check if the image has three dimensions (C, H, W)
                        img_tensor = img_tensor.permute(1, 2, 0)  # Permute to (H, W, C) for imshow
                    plt.imshow(img_tensor.numpy())
                    plt.axis('off')  # Turn off axis numbers and ticks
                plt.suptitle(f"Sample {batch_idx + 1}: Predicted: ${predicted.item():.2f}, Actual: ${actual.item():.2f}", fontsize=16)
                plt.show()

# Call the function to show predictions after the last epoch
show_predictions(val_loader, model, device)
