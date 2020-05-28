import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, data_dir, class_list, vocab_path):
        self.model_name = "TextRCNN"
        super().__init__(data_dir, class_list, vocab_path)

        self.dropout = 1.0
        self.num_epoches = 10
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300
        self.hidden_size = 256
        self.num_layers = 1


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.num_vocab, config.embed, padding_idx=config.num_vocab - 1
        )
        self.lstm = nn.LSTM(
            config.embed,
            config.hidden_size,
            config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout,
        )
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
