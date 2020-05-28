import os
import time
from datetime import timedelta
from glob import iglob
import pickle as pkl
import torch
import jieba


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = "<UNK>", "<PAD>"  # 未知字，padding符号


def seg_text(data_dir):
    """
    Use jieba to cut all text.
    """
    for file_path in iglob(os.path.join(data_dir, "**", "*.txt"), recursive=True):
        with open(file_path, "r") as f:
            text = f.readline()
        with open(file_path, "w") as f:
            f.write(" ".join(jieba.cut(text)))


def build_vocab(data_dir, tokenizer, max_size, min_freq):
    """
    Build vocabulary from the one line files.
    """
    vocab_dic = {}
    for file_path in iglob(os.path.join(data_dir, "*", "*")):
        with open(file_path, "r", encoding="UTF-8") as f:
            content = f.readline()
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

    vocab_list = sorted(
        [_ for _ in vocab_dic.items() if _[1] >= min_freq],
        key=lambda x: x[1],
        reverse=True,
    )[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, word_level=True):
    if word_level:
        tokenizer = lambda x: x.split()  # word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        vocab = build_vocab(
            config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1
        )
        pkl.dump(vocab, open(config.vocab_path, "wb"))
    print(f"Vocab size: {len(vocab)}")

    label_to_id = {label: idx for idx, label in enumerate(config.class_list)}

    def load_dataset(data_dir, pad_size=32):
        contents = []
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            label_id = label_to_id.get(label, -1)
            for file_path in iglob(os.path.join(label_dir, "*")):
                with open(file_path, "r", encoding="UTF-8") as f:
                    content = f.readline()
                    token = tokenizer(content)
                    seq_len = len(token)
                    if pad_size:
                        if len(token) < pad_size:
                            token.extend([PAD] * (pad_size - len(token)))
                        else:
                            token = token[:pad_size]
                            seq_len = pad_size
                    words_line = [vocab.get(word, vocab.get(UNK)) for word in token]
                contents.append((words_line, label_id, seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                self.index * self.batch_size: (self.index + 1) * self.batch_size
            ]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_diff(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_vocab(data_dir, word_level=False):
    train_dir = os.path.join(data_dir, "TrainData")
    vocab_path = "vocab-word.pkl" if word_level else "vocab-char.pkl"

    if word_level:
        seg_text(data_dir)
        tokenizer = lambda x: x.split(" ")
    else:  # char-level
        tokenizer = lambda x: [y for y in x]
    word_to_id = build_vocab(train_dir, tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(word_to_id, open(vocab_path, "wb"))


if __name__ == "__main__":
    if not os.path.exists("saved_dict"):
        os.mkdir("saved_dict")
    get_vocab("Data", word_level=False)
