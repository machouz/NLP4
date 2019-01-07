import time
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
import sys
from utils import *

EPOCHS = 5
HIDDEN_RNN = [50, 50]
EMBEDDING = 50
BATCH_SIZE = 100
LR = 0.01
LR_DECAY = 0.5


def Timer(start):
    while True:
        now = time.time()
        yield now - start
        start = now


def get_words_id(word, words_id):
    if word not in words_id:
        return words_id["UUUNKKK"]
    return words_id[word]


class LiveModel(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm, id_size, lemma_size, tag_size, pos_size, head_size,
                 dep_size, iob_size, type_size, tagset_size=1, bidirectional=True):
        super(LiveModel, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_lstm = hidden_lstm
        self.id_embeddings = nn.Embedding(id_size, embedding_dim)
        self.lemma_embeddings = nn.Embedding(lemma_size, embedding_dim)
        self.tag_embeddings = nn.Embedding(tag_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(pos_size, embedding_dim)
        self.head_embeddings = nn.Embedding(head_size, embedding_dim)
        self.dep_embeddings = nn.Embedding(dep_size, embedding_dim)
        self.iob_embeddings = nn.Embedding(iob_size, embedding_dim)
        self.type_embeddings = nn.Embedding(type_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(8 * embedding_dim, hidden_lstm[0] // 2, bidirectional=bidirectional, batch_first=True)

        self.lstm2 = nn.LSTM(hidden_lstm[0], hidden_lstm[1] // 2, bidirectional=bidirectional, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_lstm[1], tagset_size)
        self.init_hidden()

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden1 = (torch.randn(2, batch_size, self.hidden_lstm[0] // 2),
                        torch.randn(2, batch_size, self.hidden_lstm[0] // 2))
        self.hidden2 = (torch.randn(2, batch_size, self.hidden_lstm[1] // 2),
                        torch.randn(2, batch_size, self.hidden_lstm[1] // 2))

    def detach_hidden(self):
        self.hidden1 = (self.hidden1[0].detach(), self.hidden1[1].detach())
        self.hidden2 = (self.hidden2[0].detach(), self.hidden2[1].detach())

    def forward(self, id, lemma, tag, pos, head, dep, iob, type):
        sentence_len = id.shape[-1]
        id = self.id_embeddings(id)
        lemma = self.lemma_embeddings(lemma)
        tag = self.tag_embeddings(tag)
        pos = self.pos_embeddings(pos)
        head = self.head_embeddings(head)
        dep = self.dep_embeddings(dep)
        iob = self.iob_embeddings(iob)
        type = self.type_embeddings(type)
        block = torch.stack([id, lemma, tag, pos, head, dep, iob, type]).view(1, sentence_len, -1)
        lstm_out, self.hidden1 = self.lstm1(
            block)
        lstm_out, self.hidden2 = self.lstm2(
            lstm_out)
        lstm_out = lstm_out.reshape(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
        tag_space = lstm_out.sum(dim=0)
        return tag_space


class BasicNet(nn.Module):
    def __init__(self, image_size=66, layers=[100, 50, 2]):
        super(BasicNet, self).__init__()
        self.fc0 = nn.Linear(image_size, layers[0])
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])

    def forward(self, sent, per, loc):
        x = torch.cat([per, loc, sent])
        x = self.fc0(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)





def dic2tensor(dic):
    return torch.tensor([
        get_words_id(dic["ID"], ids_id),
        get_words_id(dic["LEMMA"], lemmas_id),
        get_words_id(dic["TAG"], tags_id),
        get_words_id(dic["POS"], poss_id),
        get_words_id(dic["HEAD"], heads_id),
        get_words_id(dic["DEP"], deps_id),
        get_words_id(dic["IOB"], iobs_id),
        get_words_id(dic["TYPE"], types_id),
    ], dtype=torch.float)


if __name__ == '__main__':
    print("Using transducer 1")
    print('Learning rate {}'.format(LR))
    print('Learning rate decay {}'.format(LR_DECAY))
    print('Hidden layer {}'.format(HIDDEN_RNN))
    print('Batch size {}'.format(BATCH_SIZE))

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Processed_Corpus/Corpus.TRAIN.processed.txt'
    annoted_file = sys.argv[2] if len(sys.argv) > 2 else 'data/Annotation/TRAIN.annotations.txt'

    data = read_processed_file(input_file)
    ids_id, lemmas_id, tags_id, poss_id, heads_id, iobs_id, deps_id, types_id = get_ids(data)

    model_sent = LiveModel(EMBEDDING, HIDDEN_RNN, id_size=len(ids_id), lemma_size=len(lemmas_id), tag_size=len(tags_id),
                           pos_size=len(poss_id), head_size=len(heads_id),
                           dep_size=len(deps_id), iob_size=len(iobs_id), type_size=len(types_id))
    model = BasicNet()
    ann_dic = dic_annotations_file(annoted_file, input_file)
    vecs_dic = []
    sentences_dic = {}
    for num, sentence in data:
        vecs_dic.append({})
        sentences_dic[num] = sentence
        vecs_dic[-1]["SENT"] = num
        vecs_dic[-1]["ID"] = []
        vecs_dic[-1]["LEMMA"] = []
        vecs_dic[-1]["TAG"] = []
        vecs_dic[-1]["POS"] = []
        vecs_dic[-1]["HEAD"] = []
        vecs_dic[-1]["DEP"] = []
        vecs_dic[-1]["IOB"] = []
        vecs_dic[-1]["TYPE"] = []
        for word in sentence:
            vecs_dic[-1]["ID"].append(get_words_id(word["ID"], ids_id))
            vecs_dic[-1]["LEMMA"].append(get_words_id(word["LEMMA"], lemmas_id))
            vecs_dic[-1]["TAG"].append(get_words_id(word["TAG"], tags_id))
            vecs_dic[-1]["POS"].append(get_words_id(word["POS"], poss_id))
            vecs_dic[-1]["HEAD"].append(get_words_id(word["HEAD"], heads_id))
            vecs_dic[-1]["DEP"].append(get_words_id(word["DEP"], deps_id))
            vecs_dic[-1]["IOB"].append(get_words_id(word["IOB"], iobs_id))
            vecs_dic[-1]["TYPE"].append(get_words_id(word["TYPE"], types_id))
    for sentence in vecs_dic:
        for per in sentences_dic[sentence["SENT"]]:
            for loc in sentences_dic[sentence["SENT"]]:
                B = model_sent(torch.tensor(sentence["ID"], dtype=torch.long),
                               torch.tensor(sentence["LEMMA"], dtype=torch.long),
                               torch.tensor(sentence["TAG"], dtype=torch.long),
                               torch.tensor(sentence["POS"], dtype=torch.long),
                               torch.tensor(sentence["HEAD"], dtype=torch.long),
                               torch.tensor(sentence["DEP"], dtype=torch.long),
                               torch.tensor(sentence["IOB"], dtype=torch.long),
                               torch.tensor(sentence["TYPE"], dtype=torch.long))
                t_per = dic2tensor(per)
                t_loc = dic2tensor(loc)
                C = model(B, t_per, t_loc)
                print C
