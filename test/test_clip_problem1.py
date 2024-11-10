import open_clip
import torch
from PIL import Image
model, _, preprocess = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="../pretrained_models/open_clip_pytorch_model.bin")
# image_path = "./cat.png"
image_path = "./022.png"
image = Image.open(image_path).convert("RGB")
# description = [
#     "a photo of a cat",
#     "a photo of a tiger",
#     "a photo of a dog"
# ]
description = [
    "a photo of a normal screw",
    "a photo of a broken screw",
    "a photo of an abnormal screw",
    "a photo of a defect screw"
]

tokenizer = open_clip.get_tokenizer(model_name="ViT-H-14")
text = tokenizer(description)
image = preprocess(image).unsqueeze(0)


print(image.shape)
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

print(image_features.shape, text_features.shape)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
print(similarity)


import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.imshow(Image.open(image_path))
plt.axis("off")

plt.subplot(2, 1, 2)
plt.barh(description, similarity[:, 0], color='skyblue')
plt.xlabel("probability")
plt.xlim(0, 1)
plt.show()
