# vit google model
# ViTMultiImageRegressionModel processes each image in the sample seperately
# ImageAggregator aggregates embeddings from all images in the sample
# CustomViTHead processes the aggregated embeddings and runs though custom head consisting of two linear layers
# Output is a single value for each sample in the batch


import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel

# so config can be imported
import sys
from pathlib import Path
current_file_path = Path(__file__).absolute()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import BATCH_SIZE, ENABLE_DEV_LOGS



base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

class ImageAggregator(nn.Module):
    def __init__(self, aggregation_method="mean"):
        super(ImageAggregator, self).__init__()
        self.aggregation_method = aggregation_method

    def forward(self, embeddings):
        if self.aggregation_method == "mean":
            return torch.mean(embeddings, dim=0)
        elif self.aggregation_method == "sum":
            return torch.sum(embeddings, dim=0)
        elif self.aggregation_method == "max":
            return torch.max(embeddings, dim=0)[0]
        else:
            return torch.mean(embeddings, dim=0)  # fallback to mean


class CustomViTHead(nn.Module):
    def __init__(self):
        super(CustomViTHead, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x.squeeze(-1)
        # return torch.sigmoid(x.squeeze(-1))


# Combine base model with custom head
class ViTMultiImageRegressionModel(nn.Module):
    def __init__(self, base_model, aggregator, custom_head):
        super(ViTMultiImageRegressionModel, self).__init__()
        self.base_model = base_model
        self.aggregator = aggregator
        self.custom_head = custom_head

    def forward(self, batch):
        batch_embeddings = []
        
        # create embeddings for all images in one sample in batch
        for sample in batch:
            # You only need to stack if instance_images is a list of tensors, otherwise directly use the tensor.
            instance_embeddings = [self.base_model(pixel_values=image.unsqueeze(0)).last_hidden_state[:, 0, :]
                                   for image in sample]
            instance_embeddings = torch.cat(instance_embeddings, dim=0)
            batch_embeddings.append(instance_embeddings)
        
        aggregated_embeddings = torch.stack([
            self.aggregator(instance) for instance in batch_embeddings
        ])

        if ENABLE_DEV_LOGS: print("Aggregated embeddings shape: ", aggregated_embeddings.shape)
        
        # Pass aggregated embedding through regression head
        output = self.custom_head(aggregated_embeddings)
        if ENABLE_DEV_LOGS: print("Final output shape: ", output.shape)
        return output


def get_vit_model(
        aggregation_method="mean"
):
    aggregator = ImageAggregator(aggregation_method=aggregation_method)
    custom_head = CustomViTHead()
    model = ViTMultiImageRegressionModel(base_model, aggregator, custom_head)

    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    return model, feature_extractor

