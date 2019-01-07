from __future__ import unicode_literals, print_function, division
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import *

MAX_LENGTH = 50
EPOCHS = 5
HIDDEN_RNN = 50
EMBEDDING = 50
LR = 0.01
LR_DECAY = 0.5


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 8, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)
        self.out_per = nn.Linear(hidden_size, output_size)
        self.out_loc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        try:
            output, hidden = self.gru(output, hidden)
        except RuntimeError:
            print("lol")
        per = self.softmax(self.out_per(output[0]))
        loc = self.softmax(self.out_loc(output[0]))
        output = torch.cat([per, loc])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=50):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[0, 0]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


if __name__ == '__main__':
    print("Using transducer 1")
    print('Learning rate {}'.format(LR))
    print('Learning rate decay {}'.format(LR_DECAY))
    print('Hidden layer {}'.format(HIDDEN_RNN))

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Processed_Corpus/Corpus.TRAIN.processed.txt'
    annoted_file = sys.argv[2] if len(sys.argv) > 2 else 'data/Annotation/TRAIN.annotations.txt'

    data = read_processed_file(input_file)
    event_ids, id_events = get_ids(data)

    encoder = EncoderRNN(input_size=len(event_ids), hidden_size=HIDDEN_RNN)
    decoder = DecoderRNN(hidden_size=HIDDEN_RNN, output_size=MAX_LENGTH)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.NLLLoss()
    ann_dic = dic_annotations_file(annoted_file, input_file)
    vecs_dic = {}
    sentences_dic = {}
    for num, sentence in data:
        vecs_dic[num] = []
        for word in sentence:
            vecs_dic[num].append([])
            vecs_dic[num][-1].append(get_words_id(word["ID"], event_ids))
            vecs_dic[num][-1].append(get_words_id(word["LEMMA"], event_ids))
            vecs_dic[num][-1].append(get_words_id(word["TAG"], event_ids))
            vecs_dic[num][-1].append(get_words_id(word["POS"], event_ids))
            vecs_dic[num][-1].append(get_words_id(word["HEAD"], event_ids))
            vecs_dic[num][-1].append(get_words_id(word["DEP"], event_ids))
            vecs_dic[num][-1].append(get_words_id(word["IOB"], event_ids))
            vecs_dic[num][-1].append(get_words_id(word["TYPE"], event_ids))
            vecs_dic[num][-1] = torch.tensor(vecs_dic[num][-1])
        vecs_dic[num] = torch.stack(vecs_dic[num])
    for num, sentence in vecs_dic.items():
        if num in ann_dic:
            t = []
            per_locs = ann_dic[num]
            for per_loc in per_locs:
                per, loc = per_loc
                t.append(torch.tensor([int(per["ID"]), int(loc["ID"])]))
            target = torch.stack(t)

        else:
            target = torch.tensor([0, 0]).unsqueeze(0)
        train(sentence, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
