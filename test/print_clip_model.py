from open_clip import list_models, get_model_config
from open_clip import list_pretrained, list_pretrained_models_by_tag

# 打印模型的config
models_name = "ViT-H-14"
if models_name in list_models():
    print(get_model_config(model_name=models_name))

# 打印所有的预训练模型名
print(list_pretrained())
# 通过tag知道有哪些预训练模型名
print(list_pretrained_models_by_tag(tag="openai"))
