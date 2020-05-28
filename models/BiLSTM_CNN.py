import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, data_dir, class_list, vocab_path):
        self.model_name = 'BiLSTM_CNN'
        super().__init__(data_dir, class_list, vocab_path)

        self.num_epoches = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300
        # LSTM parameters
        self.dropout_LSTM = 0
        self.hidden_size = 256
        self.num_layers = 1
        # CNN parameters
        self.dropout_CNN = 0.5
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(
            config.num_vocab, config.embed, padding_idx=config.num_vocab - 1
        )

        self.lstm = nn.LSTM(
            config.embed,
            config.hidden_size,
            config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout_LSTM,
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, config.num_filters, (k, config.embed + 2*config.hidden_size))
                for k in config.filter_sizes
            ]
        )
        self.dropout_CNN = nn.Dropout(config.dropout_CNN)

        self.fc = nn.Linear(
            config.num_filters * len(config.filter_sizes), config.num_classes
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)  # [batch_size, seq_len, 2*hidden_size + embed]
        out = out.unsqueeze(1)  # [batch_size, in_channel=1, height=seq_len, width=2*hidden_size+embed]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout_CNN(out)
        out = self.fc(out)
        return out
