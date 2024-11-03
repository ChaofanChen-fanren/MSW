from ref.builder import build_vision_projector
from types import SimpleNamespace
import torch
config = SimpleNamespace()
config.mm_hidden_size = 1024
config.mm_projector_type = 'mlp3x_gelu'

projector = build_vision_projector(config)
srcs = [torch.randn(size=(3, 256, 256)) for _ in range(4)]
data = projector(srcs)
