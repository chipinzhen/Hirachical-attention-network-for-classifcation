from torch.utils.data import dataset
import torch
import os
from functools import reduce


class IMDB_Dataset(dataset.Dataset):
    def __init__(self, DataPath, max_sentence_length=40, min_count=2, max_sentence_in_doc=10):
        super(IMDB_Dataset, self).__init__()
        self.max_sentence_length = max_sentence_length
        self.min_count = min_count
        self.max_sentence_in_doc = max_sentence_in_doc
        self.data, self.labels = self.process(DataPath)

    def __getitem__(self, item):
        return torch.Tensor(self.data[item]), self.labels[item]

    def process(self, DataPath):
        with open(DataPath, encoding='utf-8') as f:
            data = f.read().splitlines()
        data = [d.split("		")[-1].split() + [d.split("		")[2]] for d in data]
        data = sorted(data, key=lambda x: len(x), reverse=True)
        labels = [int(data[-1]) - 1 for data in data]
        data = [d[0:-1] for d in data]
        self.word2id_dict = self.generate_word2id_dict(data)
        for i, d in enumerate(data):
            data[i] = " ".join(d).split("<sssss>")
            for j, sentence in enumerate(data[i]):
                data[i][j] = sentence.split()
        data = self.convert_data2id(data)

        return data, labels

    def generate_word2id_dict(self, data):
        word_freq = {}
        for d in data:
            for word in d:
                word_freq[word] = word_freq.get(word, 0) + 1
        word2id = {"<pad>": 0, "<unk>": 1}
        for word in word_freq:
            if word_freq[word] < self.min_count:
                continue
            else:
                word2id[word] = len(word2id)
        return word2id

    def convert_data2id(self, data):
        for i, doc in enumerate(data):
            for j, sentence in enumerate(doc):
                for k, word in enumerate(sentence):
                    data[i][j][k] = self.word2id_dict.get(word, 1)
                data[i][j] = data[i][j][0:self.max_sentence_length] + [1] * (self.max_sentence_length - len(data[i][j]))
        for i, doc in enumerate(data):
            data[i] = data[i][0:self.max_sentence_in_doc] + [[0] * self.max_sentence_length] * (
                    self.max_sentence_in_doc - len(data[i]))
        return data

    def get_word2id_dict(self):
        return self.word2id_dict

    def get_class_num(self):
        return len(set(self.labels))

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = IMDB_Dataset('./imdb/imdb-test.txt.ss')
    testDataLoader = DataLoader(dataset, batch_size=2)
    for i, batch_data in enumerate(testDataLoader):
        print(batch_data)
        break
