import os
import time
import torch
import numpy as np
from importlib import import_module
import argparse

from train_eval import train, init_network, predict
from utils import build_dataset, build_iterator, get_time_diff

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, TextRCNN, TextRNN_Att, DPCNN')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    model_name = args.model
    x = import_module('models.' + model_name)
    if args.word:
        vocab_path = 'vocab-word.pkl'
        data_dir = 'data'
    else:
        vocab_path = 'vocab-char.pkl'
        data_dir = 'Data'
    class_list = sorted(label for label in os.listdir(os.path.join(data_dir, 'TrainData')) if not label.startswith('.'))

    config = x.Config(data_dir, class_list, vocab_path)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_diff(start_time)
    print("Time usage:", time_dif)

    config.num_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter)

    predictions = predict(config, model, test_iter)
    with open(model_name + 'result.txt', 'w', encoding='UTF-8') as f:
        for i, label in enumerate(predictions):
            print(f'{i}.txt {label}')
