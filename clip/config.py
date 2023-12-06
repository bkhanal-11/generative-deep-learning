import torch

debug = False
image_path = "flickr-image-dataset/flickr30k_images/flickr30k_images"
captions_path = "flickr-image-dataset/captions.csv"

# Hyperparameters
batch_size = 32
num_workers = 16
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 1
factor = 0.8
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-trained Image and Text encoder models
model_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
image_embedding = 768
text_encoder_model = "sentence-transformers/bert-base-nli-mean-tokens"
text_embedding = 768
text_tokenizer = "sentence-transformers/bert-base-nli-mean-tokens"
max_length = 100

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 512 
dropout = 0.1