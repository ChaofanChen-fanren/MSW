import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from models import clip
from models.clip import get_model_config
from models.text_encoder import TextEncoder
from models.modules import CovLayer, LinearLayer, Adapter
from models.prompt_learner import PromptLearnerNormal, PromptLearnerAbnormal
from common import positions_list
from torch.nn import functional as F


@dataclass
class LearnPromptCfg:
    dataset: str = 'visa'  # dataset name
    feature_dim: int = 768  # feature dimension (text and visual)
    n_ctx: int = 12
    device: str = 'cpu'


"""
    AnomalyCLIP:
    1. clip_model: clip model
    2. text_encoder: text encoder
    3. decoder_conv: decode image features by different shape convolution
    4. decoder_linear: decode image features by full connection
    5. normal_prompt_learner: learn normal prompts
    6. abnormal_prompt_learner: learn abnormal prompts
    7. adapter: adapter layer
"""


class AnomalyCLIP(nn.Module):
    def __init__(self,
                 clip_model_name: str,  # model name for clip eg. RN50, RN101, ViT-H-14
                 clip_pretrained_path: str,  # pretrained path for specific clip model name
                 learn_prompt_cfg: LearnPromptCfg,  # config for learnable prompts
                 args,  # additional parameter required
                 device,  # 'cpu' or 'cuda'
                 ) -> None:
        super().__init__()
        self.args = args

        clip_image_size = get_model_config(clip_model_name)['vision_cfg']['image_size']
        # create clip model by model name and pretrained path
        # self.clip_model, _, self.preprocess = clip.create_model_and_transforms(
        #     model_name=clip_model_name, pretrained=clip_pretrained_path, img_size=clip_image_size, jit=True,
        # )

        # openai
        self.clip_model, _, self.preprocess = clip.create_model_and_transforms(
            model_name=clip_model_name, pretrained="openai", img_size=clip_image_size,
        )
        self.clip_model.eval()  # frozen clip model parameters
        self.tokenizer = clip.get_tokenizer(clip_model_name)  # get tokenizer by clip model name
        self.text_encoder = TextEncoder(self.clip_model)
        self.text_encoder.eval()
        
        # TODO: rewrite feature list
        self.feature_list = self.args.features_list
        self.decoder_conv = CovLayer(1024, 768, 3)  # decode image features by different shape convolution
        self.decoder_linear = LinearLayer(1024, 768, 4)  # decode image features by full connection

        # TODO rewrite learnable prompt
        if isinstance(learn_prompt_cfg, dict):
            learn_prompt_cfg = LearnPromptCfg(**learn_prompt_cfg)

        self.normal_prompt_learner = PromptLearnerNormal(
            tokenizer=self.tokenizer,
            token_embedding=self.clip_model.token_embedding,
            dataset=learn_prompt_cfg.dataset,
            feature_dim=learn_prompt_cfg.feature_dim,
            n_ctx=learn_prompt_cfg.n_ctx,
        )
        self.abnormal_prompt_learner = PromptLearnerAbnormal(
            tokenizer=self.tokenizer,
            token_embedding=self.clip_model.token_embedding,
            dataset=learn_prompt_cfg.dataset,
            feature_dim=learn_prompt_cfg.feature_dim,
            n_ctx=learn_prompt_cfg.n_ctx,
        )

        self.adapter = Adapter(768)

    def forward(self, items, with_adapter=False, only_train_adapter=False, position=None):
        image = items["img"]
        class_name = items["cls_name"]

        with torch.no_grad():
            image_features, patch_tokens = self.clip_model.encode_image(
                image,
                self.feature_list
            )

        if with_adapter:
            image_features = self.adapter(image_features)
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if only_train_adapter:
            with torch.no_grad():
                normal_prompts = self.normal_prompt_learner(image_features, class_name)
                normal_tokenized_prompts = self.normal_prompt_learner.get_tokenized_prompts(class_name)

                abnormal_prompts = self.abnormal_prompt_learner(image_features, class_name)
                abnormal_tokenized_prompts = self.abnormal_prompt_learner.get_tokenized_prompts(class_name)

                normal_text_features = self.text_encoder(
                    normal_prompts[0], normal_tokenized_prompts
                )
                normal_text_features = normal_text_features / normal_text_features.norm(
                    dim=-1, keepdim=True
                )

                abnormal_text_features = self.text_encoder(
                    abnormal_prompts[0], abnormal_tokenized_prompts
                )
                abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(
                    dim=-1, keepdim=True
                )

                normal_text_features = normal_text_features.mean(dim=0, keepdim=True)
                normal_text_features = normal_text_features / normal_text_features.norm()
                normal_text_features = normal_text_features.unsqueeze(1)

        else:

            normal_prompts = self.normal_prompt_learner(image_features, class_name)
            normal_tokenized_prompts = self.normal_prompt_learner.get_tokenized_prompts(class_name)

            abnormal_prompts = self.abnormal_prompt_learner(image_features, class_name)
            abnormal_tokenized_prompts = self.abnormal_prompt_learner.get_tokenized_prompts(class_name)

            normal_text_features = self.text_encoder(
                normal_prompts[0], normal_tokenized_prompts
            )
            normal_text_features = normal_text_features / normal_text_features.norm(
                dim=-1, keepdim=True
            )

            abnormal_text_features = self.text_encoder(
                abnormal_prompts[0], abnormal_tokenized_prompts
            )
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(
                dim=-1, keepdim=True
            )

            normal_text_features = normal_text_features.mean(dim=0, keepdim=True)
            normal_text_features = normal_text_features / normal_text_features.norm()
            normal_text_features = normal_text_features.unsqueeze(1)

        ab_position = []
        if position is not None:
            ab_position = position

        if len(ab_position) > 0:
            tmp_abnormal_text_features = []
            for ab_p in ab_position:
                position_idx = positions_list(ab_p)
                tmp_abnormal_text_features.append(abnormal_text_features[position_idx::9])

            abnormal_text_features = torch.cat(tmp_abnormal_text_features, dim=0)

        abnormal_text_features = abnormal_text_features.mean(dim=0, keepdim=True)
        abnormal_text_features = abnormal_text_features / abnormal_text_features.norm()
        abnormal_text_features = abnormal_text_features.unsqueeze(1)

        text_features = torch.cat([normal_text_features, abnormal_text_features], dim=1)
        text_features = text_features / text_features.norm()
        text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)

        anomaly_maps = None

        if not only_train_adapter:

            patch_tokens_qkv = self.decoder_linear(patch_tokens[::2])
            patch_tokens_vv = self.decoder_conv(patch_tokens[1::2])

            anomaly_maps = []
            for layer in range(len(patch_tokens_qkv)):
                patch_tokens_qkv[layer] = patch_tokens_qkv[layer] / patch_tokens_qkv[
                    layer
                ].norm(dim=-1, keepdim=True)

                anomaly_map = (
                        100.0 * patch_tokens_qkv[layer] @ text_features.transpose(-2, -1)
                )

                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                    size=224,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

            for layer in range(len(patch_tokens_vv)):
                patch_tokens_vv[layer] = patch_tokens_vv[layer] / patch_tokens_vv[
                    layer
                ].norm(dim=-1, keepdim=True)

                anomaly_map = (
                        100.0 * patch_tokens_vv[layer] @ text_features.transpose(-2, -1)
                )

                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                    size=224,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

        return text_probs, anomaly_maps
