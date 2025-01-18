import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    pass

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cFc = nn.Linear(config.nEmbed, 4 * config.nEmbed)
        self.gelu = nn.GELU(approximate="tanh")
        self.cProj = nn.Linear(4 * config.nEmbed, config.nEmbed)

    def forward(self, x):
        x = self.cFc(x)
        x = self.gelu(x)
        x = self.cProj(x)

        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.nEmbed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.nEmbed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # tokens communicate with each other
        x = x + self.MLP(self.ln2(x)) # think individualy where tokens gathered information during self attention

        return x

@dataclass
class GPTConfig:
    blockSize: int = 256
    vocabSize: int = 65
    nLayer: int = 6
    nHead: int = 6
    nEmbed: int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocabSize, config.nEmbed),
            wpe = nn.Embedding(config.blockSize, config.nEmbed),
            h = nn.ModuleList([Block(config) for _ in range(config.nLayer)]),
            lnF = nn.LayerNorm(config.nEmbed)
        ))

        self.lmHead = nn.Linear(config.nEmbed, config.vocabSize, bias=False)