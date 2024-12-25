# vit google model
# ViTMultiImageRegressionModel processes each image in the sample seperately
# ImageAggregator aggregates embeddings from all images in the sample
# CustomViTHead processes the aggregated embeddings and runs though custom head consisting of two linear layers
# Output is a single value for each sample in the batch


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTImageProcessor, ViTModel, DetrImageProcessor, DetrForObjectDetection, AutoModelForObjectDetection

from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModel, AutoModelForImageClassification

# import cv2
# import matplotlib.pyplot as plt

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
    def __init__(self, embedding_layer_size=768, additional_metadata_count=5):
        super(CustomViTHead, self).__init__()


        # image embeddings layers
        self.fc_image = nn.Linear(embedding_layer_size, 128)
        self.dropout_image = nn.Dropout(0.2)

        # additional metadata layers
        self.fc_features = nn.Linear(additional_metadata_count, 64)
        self.dropout_features = nn.Dropout(0.2)

        # combined layers
        self.fc_combined = nn.Linear(128 + 64, 96)
        self.fc_final = nn.Linear(96, 1)

    def forward(self, aggregated_image_embeddings, additional_metadata):
        
        # Process image embeddings
        img_out = nn.functional.relu(self.fc_image(aggregated_image_embeddings))
        img_out = self.dropout_image(img_out)

        # Process additional features
        md_out = nn.functional.relu(self.fc_features(additional_metadata))
        md_out = self.dropout_features(md_out)

        # Concatenate and process combined data
        combined = torch.cat([img_out, md_out], dim=-1)
        combined = nn.functional.relu(self.fc_combined(combined))
        output = self.fc_final(combined)
        return output.squeeze(-1)

class ViTMultiImageRegressionModel(nn.Module):
    def __init__(self, base_model, aggregator, custom_head):
        super(ViTMultiImageRegressionModel, self).__init__()
        self.base_model = base_model
        self.aggregator = aggregator
        self.custom_head = custom_head

    def forward(self, batch):
        batch_embeddings = []
        batch_metadata = []

        for sample in batch:

            sample_images = sample[0]
            sample_additional_metadata = sample[1]

            # print("Sample images shape: ", sample_images.shape)
            # print("Sample additional metadata shape: ", sample_additional_metadata.shape)

            instance_embeddings = []

            for image in sample_images:
                outputs = self.base_model(pixel_values=image.unsqueeze(0))
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                instance_embeddings.append(embeddings)

            instance_embeddings = torch.cat(instance_embeddings, dim=0)
            batch_embeddings.append(instance_embeddings)

            batch_metadata.append(sample_additional_metadata)  
        
        aggregated_image_embeddings = torch.stack([
            #* iterates over batches NOT over images in a single sample (as it might seem with first glance)
            self.aggregator(instance) for instance in batch_embeddings
        ])


        outputs = []
        for embeddings, metadata in zip(aggregated_image_embeddings, batch_metadata):
            # print("Aggregated embeddings shape: ", embeddings.shape)
            # print("Metadata shape: ", metadata)
            output = self.custom_head(embeddings, metadata)
            outputs.append(output)

            # print("Output: ", output)

        final_output = torch.stack(outputs)
        # print("Final output shape: ", final_output.shape)

        return final_output


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
    elif "facebook/detr" in model_name:
        feature_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        base_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    elif "facebook" in model_name:
        base_model = AutoModel.from_pretrained(model_name)
        feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    elif "resnet" in model_name:
        base_model = ResNetForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    elif "WinKawaks" in model_name:
        base_model = AutoModelForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    elif "hustvl/yolos" in model_name:
        feature_extractor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        base_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")


    if train_only_head:
        for param in base_model.parameters():
            param.requires_grad = False

    aggregator = ImageAggregator(aggregation_method=aggregation_method, embedding_layer_size=embedding_size)
    custom_head = CustomViTHead(embedding_layer_size=embedding_size, additional_metadata_count=6)
    model = ViTMultiImageRegressionModel(base_model, aggregator, custom_head)

    return model, feature_extractor

