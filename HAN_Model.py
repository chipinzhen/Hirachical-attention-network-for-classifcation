import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class HAN_Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, gru_size, word_level_num_layers, sentence_level_num_layers,
                 class_num, batch_first):
        super(HAN_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_level_gru = nn.GRU(input_size=embedding_size, hidden_size=gru_size, num_layers=word_level_num_layers,
                                     bidirectional=True, batch_first=batch_first)
        self.word_attention_vector = torch.rand(2 * gru_size, 1, requires_grad=True)
        if torch.cuda.is_available():
            self.word_attention_vector = self.word_attention_vector.to('cuda:0')
        self.word_dense = nn.Linear(2 * gru_size, 2 * gru_size)
        self.sentence_level_gru = nn.GRU(input_size=2 * gru_size, hidden_size=gru_size, num_layers=1,
                                         bidirectional=True, batch_first=batch_first)
        self.sentence_attention_vector = torch.rand(2 * gru_size, 1, requires_grad=True)
        if torch.cuda.is_available():
            self.sentence_attention_vector = self.sentence_attention_vector.to('cuda:0')
        self.sentence_dense = nn.Linear(2 * gru_size, 2 * gru_size)
        self.fc = nn.Linear(2 * gru_size, class_num)

    def forward(self, x, gpu=False):
        sentence_num = x.shape[1]
        sentence_length = x.shape[2]

        x = x.view([-1, sentence_length])
        x_embedding = self.embedding(x)
        word_outputs, word_hidden = self.word_level_gru(x_embedding)
        attention_word_outputs = torch.tanh(self.word_dense(word_outputs))
        weights = torch.matmul(attention_word_outputs, self.word_attention_vector)
        weights = F.softmax(weights, dim=1)
        x = x.unsqueeze(2)

        if gpu:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float))

        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)

        sentence_vector = torch.sum(word_outputs * weights, dim=1).view([-1, sentence_num, word_outputs.shape[-1]])
        sentence_outputs, sentence_hidden = self.sentence_level_gru(sentence_vector)
        attention_sentence_outputs = torch.tanh(self.sentence_dense(sentence_outputs))
        weights = torch.matmul(attention_sentence_outputs, self.sentence_attention_vector)
        weights = F.softmax(weights, dim=1)
        x = x.view(-1, sentence_num, x.shape[1])
        x = torch.sum(x, dim=2).unsqueeze(2)
        if gpu:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float).cuda())
        else:
            weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        document_vector = torch.sum(sentence_outputs * weights, dim=1)
        output = self.fc(document_vector)
        return output


if __name__ == "__main__":
    han_model = HAN_Model(vocab_size=30000, embedding_size=200, gru_size=50, class_num=4, word_level_num_layers=1,
                          sentence_level_num_layers=1, batch_first=True)
    x = torch.Tensor(np.zeros([16, 50, 100])).to('cuda:0').long()
    han_model.to('cuda:0')
    x[0][0][0:10] = 1
    output = han_model(x)
    print(output.shape)
    for param in han_model.parameters():
        print(param)