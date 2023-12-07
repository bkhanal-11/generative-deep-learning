import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import config as CFG


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(self, 
                 model_name=CFG.model_name, 
                 pretrained=CFG.pretrained, 
                 trainable=CFG.trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained)
        self.data_config = timm.data.resolve_model_data_config(self.model)

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    """
    Encode text (caption) to a fixed size vector
    """
    def __init__(self, 
                 model_name=CFG.text_encoder_model, 
                 pretrained=CFG.pretrained, 
                 trainable=CFG.trainable):
        super().__init__()
        
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable
            
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
        
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, x):
        model_output = self.model(**x)
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, x["attention_mask"])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
    
    def get_tokenzier(self):
        return self.tokenizer

class ProjectionHead(nn.Module):
    """
    Projects fixed size vectors (768 for both for image and text)
    to 512
    """
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    