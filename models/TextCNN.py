import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, data_dir, class_list, vocab_path):
        self.model_name = 'TextCNN'
        super().__init__(data_dir, class_list, vocab_path)

        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epoches = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.num_vocab, config.embed, padding_idx=config.num_vocab - 1
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, config.num_filters, (k, config.embed))
                for k in config.filter_sizes
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(
            config.num_filters * len(config.filter_sizes), config.num_classes
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
