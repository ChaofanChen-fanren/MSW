import torch
import torch.nn as nn
from collections import OrderedDict
from common import PromptTemplate


class PromptLearner_abnormal(nn.Module):
    def __init__(self,
                 tokenizer,
                 token_embedding,
                 dataset: str,
                 feature_dim: int,
                 n_ctx: int,
                 ):
        super().__init__()
        vis_dim = ctx_dim = feature_dim

        # context vectors 可学习的上下文参数
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        self.ctx = nn.Parameter(ctx_vectors)

        nn.init.normal_(ctx_vectors, std=0.02)
        learner_prompt = " ".join(["X"] * n_ctx)

        self.meta_net = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("gelu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
            ])
        )

        # PromptTemplate
        self.token_prefix = {}
        self.token_suffix = {}
        self.tokenized_prompts = {}
        self.prompt_templates = PromptTemplate(dataset=dataset, abnormal=True, position=True)
        self.cls_map = self.prompt_templates.cls_map
        # prompt embedding
        for cls_name, prompt_template_list in self.prompt_templates.get_prompt().items():
            prompt_template_list = [learner_prompt + " " + p for p in prompt_template_list]  # [w_1][w_2]...[w_{n_ctx}]
            with torch.no_grad():
                self.tokenized_prompts[cls_name] = tokenizer(prompt_template_list)
                embedding = token_embedding(tokenizer(prompt_template_list))  # shape [len(status), 77(token_number), ctx_dim]
            self.token_prefix[cls_name] = embedding[:, :1, :]  # [len(status), 1, ctx_dim]
            self.token_suffix[cls_name] = embedding[:, 1 + n_ctx:, :]  # [len(status), 77-n_ctx-1, ctx_dim]

        # self.token_prefix.requires_grad = False
        # self.token_suffix.requires_grad = False

    def get_tokenized_prompts(self, class_name):
        return self.tokenized_prompts[self.cls_map[class_name]]


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )
        return prompts

    def forward(self, im_features, class_name):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        class_name = self.cls_map[class_name]

        ctx = self.ctx
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        prefix = self.token_prefix[class_name]  # (batch, len(status), 1, ctx_dim)
        suffix = self.token_suffix[class_name]  # (batch, len(status), 77 - n_ctx -1,  ctx_dim)

        n_cls = prefix.shape[0]

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn(token_number), ctx_dim)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)  # (batch, n_cls, n_tkn(token_number), ctx_dim)

        return prompts


class PromptLearner_normal(nn.Module):
    def __init__(self,
                 tokenizer,
                 token_embedding,
                 dataset: str,
                 feature_dim: int,
                 n_ctx: int,
                 ):
        super().__init__()
        vis_dim = ctx_dim = feature_dim

        # context vectors 可学习的上下文参数
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        self.ctx = nn.Parameter(ctx_vectors)

        nn.init.normal_(ctx_vectors, std=0.02)
        learner_prompt = " ".join(["X"] * n_ctx)

        self.meta_net = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("gelu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
            ])
        )

        # PromptTemplate
        self.token_prefix = {}
        self.token_suffix = {}
        self.prompt_templates = PromptTemplate(dataset=dataset, abnormal=False, position=False)
        self.tokenized_prompts = {}
        self.cls_map = self.prompt_templates.cls_map
        # prompt embedding
        for cls_name, prompt_template_list in self.prompt_templates.get_prompt().items():
            prompt_template_list = [learner_prompt + " " + p for p in prompt_template_list]  # [w_1][w_2]...[w_{n_ctx}]
            self.tokenized_prompts[cls_name] = tokenizer(prompt_template_list)
            with torch.no_grad():
                embedding = token_embedding(tokenizer(prompt_template_list))  # shape [len(status), 77(token_number), ctx_dim]
            self.token_prefix[cls_name] = embedding[:, :1, :]  # [len(status), 1, ctx_dim]
            self.token_suffix[cls_name] = embedding[:, 1 + n_ctx:, :]  # [len(status), 77-n_ctx-1, ctx_dim]

        # self.token_prefix.requires_grad = False
        # self.token_suffix.requires_grad = False

    def get_tokenized_prompts(self, class_name):
        return self.tokenized_prompts[self.cls_map[class_name]]

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )
        return prompts

    def forward(self, im_features, class_name):
        class_name = self.cls_map[class_name]

        ctx = self.ctx
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        prefix = self.token_prefix[class_name]  # (batch, len(status), 1, ctx_dim)
        suffix = self.token_suffix[class_name]  # (batch, len(status), 77 - n_ctx -1,  ctx_dim)

        n_cls = prefix.shape[0]

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn(token_number), ctx_dim)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)  # (batch, n_cls, n_tkn(token_number), ctx_dim)

        return prompts

# # [v_1][v_2]...[v_{n_ctx}][STATE][CLASS]
# class PromptLearner_normal(nn.Module):
#     def __init__(self, classnames, status, clip_model, tokenizer, dim, n_ctx, device):
#         super().__init__()
#         vis_dim = dim
#         ctx_dim = dim
#
#         # context vectors 可学习的上下文参数
#         ctx_vectors = torch.empty(n_ctx, ctx_dim)
#         self.ctx = nn.Parameter(ctx_vectors)
#
#         nn.init.normal_(ctx_vectors, std=0.02)
#         prompt = " ".join(["X"] * n_ctx)
#
#         self.meta_net = nn.Sequential(
#             OrderedDict([
#                 ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
#                 ("gelu", nn.ReLU(inplace=True)),
#                 ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
#             ])
#         )
#
#         # classnames
#         classnames = [name.replace("_", " ") for name in classnames]
#         self.tokenizer_prompt = {}
#         embedding = {}
#
#         self.token_prefix = {}
#         self.token_suffix = {}
#
#         for class_name in classnames:
#             class_name = cls_map[class_name]
#
#             p = [
#                 prompt + " " + status_i.format(class_name) + "." for status_i in status
#             ]
#
#             self.tokenizer_prompt[class_name] = tokenizer(p)  # shape [len(status), 77(token_number)]
#             self.tokenizer_prompt[class_name].requies_grad = False
#
#             with torch.no_grad():
#                 embedding[class_name] = clip_model.token_embedding(
#                     self.tokenizer_prompt[class_name]
#                 )  # shape [len(status), 77(token_number), ctx_dim]
#
#             self.token_prefix[class_name] = embedding[class_name][:, :1, :].to(device)  # [len(status), 1, ctx_dim]
#             self.token_prefix[class_name].requires_grad = False
#
#             self.token_suffix[class_name] = embedding[class_name][:, 1 + n_ctx:, :].to(
#                 device)  # [len(status), 77-n_ctx-1, ctx_dim]
#             self.token_suffix[class_name].requires_grad = False
#
#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]
#
#         prompts = torch.cat(
#             [
#                 prefix,
#                 ctx,
#                 suffix,
#             ],
#             dim=1,
#         )
#         return prompts
#
#     def forward(self, im_features, class_name):
#
#         ctx = self.ctx
#         bias = self.meta_net(im_features)  # (batch, ctx_dim)
#         bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
#         ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
#         ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
#
#         prefix = self.token_prefix[class_name]  # (batch, len(status), 1, ctx_dim)
#         suffix = self.token_suffix[class_name]  # (batch, len(status), 77 - n_ctx -1,  ctx_dim)
#
#         n_cls = prefix.shape[0]
#
#         prompts = []
#         for ctx_shifted_i in ctx_shifted:
#             ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
#             pts_i = self.construct_prompts(
#                 ctx_i, prefix, suffix
#             )  # (n_cls, n_tkn(token_number), ctx_dim)
#             prompts.append(pts_i)
#
#         prompts = torch.stack(prompts)  # (batch, n_cls, n_tkn(token_number), ctx_dim)
#
#         return prompts


# # [w_1][w_2]...[w_{n_ctx}][STATE][CLASS] with [ANOMALY CLASS] at [POS]
# class PromptLearner_abnormal(nn.Module):
#     def __init__(
#             self,
#             classnames,
#             status,
#             clip_model,
#             tokenizer,
#             dim,
#             n_ctx,
#             device,
#             positions=None,
#     ):
#         super().__init__()
#         vis_dim = dim
#         ctx_dim = dim
#
#         print("Initializing a generic context")
#         ctx_vectors = torch.empty(n_ctx, ctx_dim)
#
#         nn.init.normal_(ctx_vectors, std=0.02)
#         prompt = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#
#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
#
#         self.meta_net = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
#                     ("relu", nn.ReLU(inplace=True)),
#                     ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
#                 ]
#             )
#         )
#         if positions == None:
#             self.positions = [
#                 "top left",
#                 "top",
#                 "top right",
#                 "left",
#                 "center",
#                 "right",
#                 "bottom left",
#                 "bottom",
#                 "bottom right",
#             ]
#         else:
#             self.positions = positions
#
#         classnames = [name.replace("_", " ") for name in classnames]
#
#         self.tokenized_prompts = {}
#         embedding = {}
#
#         self.token_prefix = {}
#         self.token_suffix = {}
#
#         for origin_class_name in classnames:
#             class_name = cls_map[origin_class_name]
#
#             p = [
#                 prompt + " " + status_i.format(class_name) + " at " + position + "."
#                 for status_i in status[origin_class_name]
#                 for position in self.positions
#             ]
#
#             # print(p)
#
#             self.tokenized_prompts[class_name] = tokenizer(p)
#             with torch.no_grad():
#                 embedding[class_name] = clip_model.token_embedding(
#                     self.tokenized_prompts[class_name]
#                 )
#
#             self.token_prefix[class_name] = embedding[class_name][:, :1, :].to(device)
#             self.token_prefix[class_name].requires_grad = False
#
#             self.token_suffix[class_name] = embedding[class_name][:, 1 + n_ctx:, :].to(device)
#             self.token_suffix[class_name].requires_grad = False
#
#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
#
#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (dim0, 1, dim)
#                 ctx,  # (dim0, n_ctx, dim)
#                 suffix,  # (dim0, *, dim)
#             ],
#             dim=1,
#         )
#
#         return prompts
#
#     def forward(self, im_features, class_name):
#
#         ctx = self.ctx
#         bias = self.meta_net(im_features)  # (batch, ctx_dim)
#         bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
#         ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
#         ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
#
#         prefix = self.token_prefix[class_name]
#         suffix = self.token_suffix[class_name]
#
#         n_cls = prefix.shape[0]
#
#         prompts = []
#         for ctx_shifted_i in ctx_shifted:
#             ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
#             pts_i = self.construct_prompts(
#                 ctx_i, prefix, suffix
#             )  # (n_cls, n_tkn, ctx_dim)
#             prompts.append(pts_i)
#
#         prompts = torch.stack(prompts)
#         return prompts


# if __name__ == "__main__":
    # from common import mvtec_obj_list, status_normal
    # import open_clip
    # import torch
    #
    # clip_model, _, preprocess = open_clip.create_model_and_transforms(
    #     model_name="ViT-H-14-378-quickgelu",
    #     pretrained='/Users/chenchaofan/study/CLIP/openclip/open_clip_pytorch_model.bin'
    # )
    # clip_model.eval()
    # normal_learner = PromptLearner_normal(
    #     classnames=mvtec_obj_list,
    #     status=status_normal,
    #     clip_model=clip_model,
    #     tokenizer=open_clip.get_tokenizer("ViT-H-14"),
    #     dim=1024,
    #     n_ctx=12,
    #     device="cpu",
    # )
    # # normal_learner(torch.ones(4, 1024), ["pill", "screw", "tile", "wood"])
    # normal_learner(torch.ones(1, 1024), "pill")
