import numpy as np

import torch
from torch.optim import Adam

import matplotlib.pyplot as plt

import transformers
transformers.logging.set_verbosity_error()

from config.settings import BATCH_SIZE, ENABLE_DEV_LOGS, LEARNING_RATE, LOSS_FUNCT, EPOCHS, BATCHES_TO_AGGREGATE, WEIGHTED_BIN_COUNT, AGGREGATION_METHOD

from model.google_vit_model import get_vit_model
from model.r2_score import r2_score

from helpers.weighted_loss import compute_bin_weights, WeightedMSELoss

from helpers.data_loader import get_data_loaders

from helpers.processed_data import processed_data

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Batch size: {BATCH_SIZE}, aggregation multiplier: {BATCHES_TO_AGGREGATE}")
print("-" * 20)
print(f"Simulated batch size: {BATCH_SIZE * BATCHES_TO_AGGREGATE}")
print("-" * 20)

# model, feature_extractor = get_vit_model(aggregation_method=AGGREGATION_METHOD)
# model.load_state_dict(torch.load("models/vit_regression_model.pth")) # load weights
# model.to(device)


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    epoch_loss = 0
    total_batches = len(dataloader)

    accuracies = []

    aggregated_outputs = []
    aggregated_prices = []
    # optimizer.zero_grad()

    dataloader = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for mini_batch_idx, (sample, prices) in enumerate(dataloader, 1):

        sample = [instance.to(device) for instance in sample]

        if ENABLE_DEV_LOGS: print("one input in batch shape: ", sample[0].shape)  # Should print [8, 2, 3, 224, 224] for a batch size of 8

        prices = prices.to(device)

        if ENABLE_DEV_LOGS: print("prices shape (should be batch size): ", prices.shape)


        outputs = model(sample)

        loss = LOSS_FUNCT(outputs, prices)
        loss.backward()

        epoch_loss += loss.item()

        aggregated_outputs.append(outputs.to("cpu"))
        aggregated_prices.append(prices.to("cpu"))
        

        if (mini_batch_idx + 1) % BATCHES_TO_AGGREGATE == 0 or mini_batch_idx == total_batches:

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

            dataloader.set_postfix({
                "Avg Loss": epoch_loss / mini_batch_idx,
                "Batch Loss": loss.item(),
                "R2 Score": aggregated_accuracy
            })

    # plt.plot(accuracies)
    # plt.xlabel("Batch") 
    # plt.ylabel("R2 Score")
    # plt.title("R2 Score vs Batch")
    # plt.show()

    if total_batches > 0:
        return epoch_loss / total_batches
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


# count = 535
count = 300

images, prices = processed_data(count)
bins, bin_weights = compute_bin_weights(prices, num_bins=WEIGHTED_BIN_COUNT)

# plt.plot(bin_weights)
# plt.xlabel("Price Bins")
# plt.ylabel("Weight")
# plt.title("Price Bins vs Weights")
# plt.show()

def train_loop(epochs, model, train_loader, val_loader, optimizer, device):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        best_val_loss = min(best_val_loss, val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return best_val_loss

results = []

models = {
    "google/vit-base-patch16-224": {"embedding_size": 768, "head_only": False},
    # "google/vit-base-patch16-224-in21k": {"embedding_size": 768, "head_only": False},
    # "facebook/dinov2-small": {"embedding_size": 384, "head_only": False},
    # "google/vit-base-patch16-224 head-only": {"embedding_size": 768, "head_only": True},
    # "google/vit-base-patch16-224-in21k head-only": {"embedding_size": 768, "head_only": True},
    # "facebook/dinov2-large head-only": {"embedding_size": 1024, "head_only": True},
    # "facebook/dinov2-small head-only": {"embedding_size": 384, "head_only": True}
}

# aggregation_methods = ["mean", "sum", "attention"]
aggregation_methods = ["mean"]

results = [[0 for _ in range(len(aggregation_methods))] for _ in range(len(models))]

model, feature_extractor = None, None

for i, aggregation_method in enumerate(aggregation_methods):
    for j, (model_name, model_info) in enumerate(models.items()):

        # results[j][i] = np.random.rand()
        # continue

        train_only_head = model_info['head_only']
        actual_model_name = model_name.replace(" head-only", "") if train_only_head else model_name
        embedding_size = model_info['embedding_size']

        print(f"Training model: {actual_model_name}, aggregation method: {aggregation_method}, train_only_head: {train_only_head}")

        model, feature_extractor = get_vit_model(aggregation_method=aggregation_method, model_name=actual_model_name, train_only_head=train_only_head, embedding_size=embedding_size)
        model.to(device)
        
        train_loader, val_loader = get_data_loaders(images, prices, feature_extractor)

        LOSS_FUNCT = WeightedMSELoss(bins, bin_weights, device=device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        val_loss = train_loop(EPOCHS, model, train_loader, val_loader, optimizer, device)
        results[j][i] = val_loss


plt.figure(figsize=(13, 9))
heatmap = plt.imshow(results, cmap='magma', interpolation='nearest')
plt.xticks(range(len(aggregation_methods)), aggregation_methods, rotation=45, ha="right")
plt.yticks(range(len(models)), [name for name in models])
plt.colorbar(heatmap)
plt.title("Training Loss by Model and Aggregation Method")
plt.xlabel("Aggregation Method")
plt.ylabel("Model")
plt.show()


torch.save(model.state_dict(), "models/vit_regression_model.pth")


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
