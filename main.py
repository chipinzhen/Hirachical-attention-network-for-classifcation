import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from HAN_Model import HAN_Model
from data.IMDB_Dataset import IMDB_Dataset
from torch.utils.data import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
use_gpu = True if torch.cuda.is_available() else False

TrainingDataset = IMDB_Dataset('./data/imdb/imdb-test.txt.ss')
TrainingDataLoader = DataLoader(TrainingDataset, batch_size=64)

vocab_size = len(TrainingDataset.get_word2id_dict())
class_num = TrainingDataset.get_class_num()
print(class_num)
model = HAN_Model(vocab_size=vocab_size, embedding_size=64, gru_size=64, class_num=class_num, word_level_num_layers=1,
                  sentence_level_num_layers=1, batch_first=True)
model.to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

Epoch_num = 10
loss_list = []
for epoch in range(Epoch_num):
    model.train()
    for i, (data, label) in enumerate(TrainingDataLoader):
        data = data.to(device).long()
        label = label.to(device).long()
        label = label.squeeze()
        output = model(data, gpu=use_gpu)
        loss = lossFunction(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
