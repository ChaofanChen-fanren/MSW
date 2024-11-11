import numpy as np
import torch
import torch.nn as nn
from models.text_encoder import TextEncoder
from models.modules import CovLayer, LinearLayer, Adapter
from models import clip
from models.prompt_learner import PromptLearnerNormal, PromptLearnerAbnormal
from common import positions_list
from torch.nn import functional as F
from .AnomalyCLIP import LearnPromptCfg


class FiLo(nn.Module):
    def __init__(self,
                 learn_prompt_cfg: LearnPromptCfg,  # config for learnable prompts,
                 args,
                 device) -> None:
        super().__init__()

        self.args = args

        self.device = device

        self.clip_model, _, self.preprocess = clip.create_model_and_transforms(
            args.clip_model, args.image_size, pretrained="openai"
        )
        self.clip_model.eval()

        self.tokenizer = clip.get_tokenizer(args.clip_model)

        self.decoder_cov = CovLayer(1024, 768, 3)
        self.decoder_linear = LinearLayer(1024, 768, 4)
        self.text_encoder = TextEncoder(self.clip_model)
        self.text_encoder.eval()

        if isinstance(learn_prompt_cfg, dict):
            learn_prompt_cfg = LearnPromptCfg(**learn_prompt_cfg)

        self.normal_prompt_learner = PromptLearnerNormal(
            tokenizer=self.tokenizer,
            token_embedding=self.clip_model.token_embedding,
            dataset=learn_prompt_cfg.dataset,
            feature_dim=learn_prompt_cfg.feature_dim,
            n_ctx=learn_prompt_cfg.n_ctx,
            device=device
        )
        self.abnormal_prompt_learner = PromptLearnerAbnormal(
            tokenizer=self.tokenizer,
            token_embedding=self.clip_model.token_embedding,
            dataset=learn_prompt_cfg.dataset,
            feature_dim=learn_prompt_cfg.feature_dim,
            n_ctx=learn_prompt_cfg.n_ctx,
            device=device
        )
        # self.normal_prompt_learner = PromptLearner_normal(
        #     obj_list,
        #     status_normal,
        #     self.clip_model,
        #     self.tokenizer,
        #     768,
        #     args.n_ctx,
        #     args.device
        # )
        #
        # self.abnormal_prompt_learner = PromptLearner_abnormal(
        #     obj_list,
        #     status_abnormal,
        #     self.clip_model,
        #     self.tokenizer,
        #     768,
        #     args.n_ctx,
        #     args.device
        # )
        self.adapter = Adapter(768)

    def forward(self, items, with_adapter=False, only_train_adapter=False, positions=None):
        image = items["img"].to(self.device)
        class_name = items["cls_name"][0]

        with torch.no_grad():
            image_features, patch_tokens = self.clip_model.encode_image(
                image, self.args.features_list
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
        if positions != None:
            ab_position = positions

        if len(ab_position) > 0:
            tmp_abnormal_text_features = []
            for ab_p in ab_position:
                position_idx = positions_list.index(ab_p)
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
            patch_tokens_vv = self.decoder_cov(patch_tokens[1::2])

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
                    size=self.args.image_size,
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
                    size=self.args.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

        return text_probs, anomaly_maps