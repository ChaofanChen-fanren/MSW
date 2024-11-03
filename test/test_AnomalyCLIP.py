from models import AnomalyCLIP

from types import SimpleNamespace
import torch
learn_prompt_cfg = {
    'dataset': 'visa',
    'feature_dim': 768,
    'n_ctx': 12,
    'device': 'cpu'
}
args = SimpleNamespace()
args.n_ctx = 12
args.features_list = [6, 12, 18, 24]
model = AnomalyCLIP(
    clip_model_name="ViT-L-14",
    clip_pretrained_path="../pretrained_models", learn_prompt_cfg=learn_prompt_cfg, args=args, device='cpu')
# model = AnomalyCLIP(
#     clip_model_name="ViT-H-14",
#     clip_pretrained_path="pretrained_models/open_clip_pytorch_model.bin", learn_prompt_cfg=learn_prompt_cfg ,args=args , device='cpu')
items = {}
items["img"] = torch.ones(1, 3, 224, 224)
items["cls_name"] = "candle"
text_probs, anomaly_maps = model(items, with_adapter=False)
# text_probs, anomaly_maps = model(items, only_train_adapter=True, with_adapter=True)
