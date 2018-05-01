import pickle

import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from tqdm import tqdm  # Wrap any iterator to show a progress bar.
from utils import save_checkpoint, print_hyperparameters
from readata import readtrain, readdev, prepare_embedding, prepare, get_loader
from csfeatures import morphVec
from config import *



class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #Creamos un LSTM con tamaño de entrada de embedding y estados de salida ocultos hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // NUM_DIRS, 
                dropout = DROPOUT, 
                num_layers = NUM_LAYERS, 
                bidirectional = BIDIRECTIONAL,
                )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        #self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=DROPOUT)
        self.softmax = nn.LogSoftmax()
    
#    def init_hidden(self):
#        return (autograd.Variable(torch.zeros(NUM_LAYERS * NUM_DIRS,1,self.hidden_dim // NUM_DIRS )),
#                autograd.Variable(torch.zeros(NUM_LAYERS * NUM_DIRS,1,self.hidden_dim // NUM_DIRS)))

    def init_hidden(self, batch):
            return (autograd.Variable(torch.randn(NUM_LAYERS*NUM_DIRS, batch, self.hidden_dim // NUM_DIRS)),
                    autograd.Variable(torch.randn(NUM_LAYERS*NUM_DIRS, batch, self.hidden_dim // NUM_DIRS)))

#    def forward(self,sentence):
#        embeds = self.word_embeddings(sentence)
#
#        embeds = self.dropout(embeds) 
#        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
#        lstm_out = self.dropout(lstm_out) 
#        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
#        tag_scores = F.log_softmax(tag_space, dim=1)
#        return tag_scores

    def _get_lstm_features(self, names, lengths):
        self.hidden = self.init_hidden(names.size(-1))
        embeds = self.word_embeddings(names)  
        print(embeds)
        print(lengths)
        packed_input = pack_padded_sequence(embeds, lengths)  
        packed_output, (ht, ct) = self.lstm(packed_input, self.hidden)  
        lstm_out, _ = pad_packed_sequence(packed_output)  
        lstm_feats = self.hidden2tag(lstm_out)
        output = self.softmax(lstm_feats)  
        return output

    def forward(self, name, lengths):
        return self._get_lstm_features(name, lengths)


