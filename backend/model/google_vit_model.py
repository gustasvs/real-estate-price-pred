# vit google model
# ViTMultiImageRegressionModel processes each image in the sample seperately
# ImageAggregator aggregates embeddings from all images in the sample
# CustomViTHead processes the aggregated embeddings and runs though custom head consisting of two linear layers
# Output is a single value for each sample in the batch


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTImageProcessor, ViTModel

from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModel

# so config can be imported
import sys
from pathlib import Path
current_file_path = Path(__file__).absolute()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import ENABLE_DEV_LOGS

embedding_layer_size = 768

base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


# base_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
# feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
# base_model = AutoModel.from_pretrained('facebook/dinov2-large')
# embedding_layer_size = 1024

# feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
# base_model = AutoModel.from_pretrained('facebook/dinov2-small')
# embedding_layer_size = 384

# for param in base_model.parameters():
#     param.requires_grad = False

class ImageAggregator(nn.Module):
    def __init__(self, aggregation_method="mean", embedding_layer_size=768):
        super(ImageAggregator, self).__init__()
        self.aggregation_method = aggregation_method
        if self.aggregation_method == "attention":
            self.attention_weights = nn.Linear(3 * embedding_layer_size, 3)

    def forward(self, embeddings):
        if self.aggregation_method == "mean":
            return torch.mean(embeddings, dim=0)
        elif self.aggregation_method == "sum":
            return torch.sum(embeddings, dim=0)
        elif self.aggregation_method == "max":
            return torch.max(embeddings, dim=0)[0]
        elif self.aggregation_method == "attention":
        
            mean_agg = torch.mean(embeddings, dim=0)
            sum_agg = torch.sum(embeddings, dim=0)
            max_agg = torch.max(embeddings, dim=0)[0]
            concatenated_agg = torch.cat([mean_agg, sum_agg, max_agg], dim=0)
            attention_scores = self.attention_weights(concatenated_agg)
            attention_scores = F.softmax(attention_scores, dim=0)
            
            # print("Attention scores: ", attention_scores)

            final_agg = (
                attention_scores[0] * mean_agg +
                attention_scores[1] * sum_agg +
                attention_scores[2] * max_agg
            )
            return final_agg
        

        else:
            return torch.mean(embeddings, dim=0)  # fallback to mean


class CustomViTHead(nn.Module):
    def __init__(self, embedding_layer_size=768):
        super(CustomViTHead, self).__init__()
        self.fc1 = nn.Linear(embedding_layer_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)



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
            attention_maps = []
            instance_embeddings = []
            
            # import cv2
            # import matplotlib.pyplot as plt

            for image in sample:
                outputs = self.base_model(pixel_values=image.unsqueeze(0))
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                instance_embeddings.append(embeddings)

            #     # Handle attentions
            #     attentions = outputs.attentions[-1]  # Last layer attentions
            #     avg_attention_map = attentions.mean(dim=1)  # Average across heads
            #     cls_attention = avg_attention_map[:, 0, 1:]  # Focus on [CLS]-to-patch attentions

            #     # Reshape the attention to match the patch grid
            #     patch_size = int(cls_attention.size(-1) ** 0.5)  # Assumes square patch grid
            #     cls_attention = cls_attention.view(patch_size, patch_size)

            #     attention_map_resized = cv2.resize(cls_attention.cpu().detach().numpy(), (224, 224))
            #     attention_maps.append(attention_map_resized)

            # # Visualization
            # for img, attn_map in zip(sample, attention_maps):
            #     attention_map_normalized = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())  # Normalize for visualization
            #     plt.imshow(img.permute(1, 2, 0).cpu().numpy())  # Image
            #     plt.imshow(attention_map_normalized, cmap='jet', alpha=0.5)  # Attention map overlay
            #     plt.axis('off')
            #     plt.show()


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
        aggregation_method="mean",
        model_name="google/vit-base-patch16-224",
        train_only_head=False,
        embedding_size=768
):
    base_model = None
    feature_extractor = None
    if "google" in model_name:
        base_model = ViTModel.from_pretrained(model_name)
        feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    elif "facebook" in model_name:
        base_model = AutoModel.from_pretrained(model_name)
        feature_extractor = AutoImageProcessor.from_pretrained(model_name)

    if train_only_head:
        for param in base_model.parameters():
            param.requires_grad = False

    aggregator = ImageAggregator(aggregation_method=aggregation_method, embedding_layer_size=embedding_size)
    custom_head = CustomViTHead(embedding_layer_size=embedding_size)
    model = ViTMultiImageRegressionModel(base_model, aggregator, custom_head)

    return model, feature_extractor

