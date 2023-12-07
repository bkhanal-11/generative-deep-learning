import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import config as CFG
from modules import ImageEncoder, TextEncoder, ProjectionHead

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return  total_loss

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.tokenizer = self.text_encoder.get_tokenzier()     
        
        self.clip_loss = CLIPLoss()
        
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))   

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        keys_to_keep = ['input_ids', 'token_type_ids', 'attention_mask']
        filtered_dict = {k: batch[k] for k in keys_to_keep if k in batch}
        text_features = self.text_encoder(
            filtered_dict
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        loss = self.clip_loss(image_embeddings, text_embeddings, self.logit_scale)
        
        return loss


if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    clip = CLIPModel()
    loss = clip(batch)
    print(f"Calculated Loss: {loss}")