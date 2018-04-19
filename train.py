import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

from utils import save_checkpoint
from readata import readtrain, prepare_embedding, prepare
import pickle

NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
WEIGHT_DECAY = 1e-4
SAVE_EVERY = 1
DROPOUT = .3

EMBEDDING_DIM = 100
HIDDEN_DIM = 100
LEARNING_RATE = 0.01

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #Creamos un LSTM con tamaño de entrada de embedding y estados de salida ocultos hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
                autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

    def forward(self,sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train():
    print('Reading data...')
    data = readtrain()
    print('Preparing data...')
    tag_to_index, word_to_index, index_to_tag, index_to_word = prepare_embedding(data)
    with open('data.pickle', 'wb') as f:
        pickle.dump([tag_to_index, word_to_index, index_to_tag, index_to_word], f, pickle.HIGHEST_PROTOCOL)


    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_index), len(tag_to_index))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


    filename = "epoch"
    #Train
    print('Train with', len(data), 'examples.')
    for epoch in range(300):
        loss_sum = 0
        for sentence, tags in data:
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = prepare(sentence, word_to_index)
            targets = prepare(tags, tag_to_index)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss_sum += loss.data[0]
            #print('Loss, epoch', epoch, '=', loss.data[0])
            loss.backward()
            optimizer.step()

        if epoch % SAVE_EVERY:
            print("epoch = %d, loss = %f" % (epoch, loss_sum / len(data)))
        else:
            save_checkpoint(filename+str(epoch), model, epoch, loss_sum / len(data))

if __name__ == '__main__':
    train()

