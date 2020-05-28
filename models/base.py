import os
import torch


class BaseConfig:
    def __init__(self, data_dir, class_list, vocab_path):
        self.train_path = os.path.join(data_dir, 'TrainData')
        self.dev_path = os.path.join(data_dir, 'ValidData')
        self.test_path = os.path.join(data_dir, 'TestData')

        self.class_list = class_list
        self.num_classes = len(self.class_list)

        self.vocab_path = vocab_path
        self.num_vocab = 0  # set when training

        self.save_path = os.path.join('saved_dict', self.model_name + '.ckpt')
        self.log_path = os.path.join('log', self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
