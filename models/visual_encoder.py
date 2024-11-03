import clip
import torch

clip_model, _, preprocess = clip.create_model_and_transforms(
    model_name="ViT-H-14", pretrained="../pretrained_models/open_clip_pytorch_model.bin", img_size=224
)
image_feature, patch_tokens = clip_model.encode_image(image=torch.randn(1, 3, 448, 448), out_layers=[6, 12, 18, 24])
print(len(image_feature[0]), len(patch_tokens))
# [batch, 257, 1024]
print(image_feature.shape)
# list([1, 257, 1280]...) len = 7

