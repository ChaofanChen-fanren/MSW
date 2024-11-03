import open_clip
from open_clip import create_model_and_transforms
from open_clip import tokenizer
from open_clip import get_model_preprocess_cfg
from torchinfo import summary
import torch
import random
import numpy as np
from PIL import Image

models, _, preprocess = create_model_and_transforms('ViT-H-14', pretrained='../pretrained_models/open_clip_pytorch_model.bin')
# models, _, preprocess = create_model_and_transforms('RN50', pretrained='../pretrained_models/RN50.pt', jit=True)
# dict(size, mode, mean, std, interpolation, bicubic, resize_mode, shortest, fill_color)
preprocess_cfg = get_model_preprocess_cfg(models)
print(preprocess_cfg['size'])
# print(models.visual)

# ##########################################################################################
# #                   Print model text_dim and image_size
# ##########################################################################################
# models = models.to('cpu')
# print(models.visual.image_size)
# print(models.transformer.width)
# ##########################################################################################
# #                   Print Structure for visual and text
# ##########################################################################################
# # print model name structure for visual
# summary(models.visual, input_size=(1, 3, 224, 224))
# # print model name structure for text
# summary(models.transformer, input_size=(77, 3, 1024))

##########################################################################################
#                   Test CLIP Encode_Image Function
##########################################################################################
# Class CLIP model encode_image function only return visual.image_feature so set model.visual.output_tokens
# models.visual.output_tokens = True
# image_feature, patch_tokens = models.visual(torch.ones(size=(3, 3, 224, 224)))
# print(image_feature.shape, patch_tokens.shape)

##########################################################################################
#                   Test CLIP token_embedding
##########################################################################################
tokenizer = open_clip.get_tokenizer('ViT-H-14')
embedding = models.token_embedding(tokenizer('hello world'))
print(embedding.shape)