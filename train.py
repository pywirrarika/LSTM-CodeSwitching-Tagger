import pickle

import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm  # Wrap any iterator to show a progress bar.
from utils import save_checkpoint, print_hyperparameters
from readata import readtrain, readdev, prepare_embedding, prepare, mini_batch, get_loader
from csfeatures import morphVec
from predict import predict
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
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=DROPOUT)
    
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(NUM_LAYERS * NUM_DIRS,1,self.hidden_dim // NUM_DIRS )),
                autograd.Variable(torch.zeros(NUM_LAYERS * NUM_DIRS,1,self.hidden_dim // NUM_DIRS)))

    def forward(self,sentence):
        embeds = self.word_embeddings(sentence)

        embeds = self.dropout(embeds) 
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = self.dropout(lstm_out) 
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train():
    torch.initial_seed()
    filename = "models/model"

    print_hyperparameters()
    # Loding data and preprocesing
    print('Reading data...')
    data_raw = readtrain()
    data_raw_dev = readdev()

    print('Preparing data...')

    tag_to_index, word_to_index, index_to_tag, index_to_word = prepare_embedding(data_raw)
    idxs = [tag_to_index, word_to_index, index_to_tag, index_to_word]
    

    loader = get_loader(data_raw, idxs)
    print(loader)
    
    data = []
    for sentence, tags in data_raw:
        sentence_in = prepare(sentence, word_to_index)
        targets = prepare(tags, tag_to_index)
        data.append((sentence_in, targets))

    data_dev = data_raw_dev
#    data_dev = []
#    for sentence, tags in data_raw_dev:
#        sentence_in = prepare(sentence, word_to_index)
#        targets = prepare(tags, tag_to_index)
#        data_dev.append((sentence_in, targets))
#

    #data_batches = mini_batch(idxs, data)
#    data_dev_batches = mini_batch(idxs, data_dev)
    print('Training Data size', len(data))
    print('Dev data size', len(data_dev))
 
    # Save indexes to data.pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump([tag_to_index, word_to_index, index_to_tag, index_to_word], f, pickle.HIGHEST_PROTOCOL)

    
    # Create an instance of the NN
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_index), len(tag_to_index))
    loss_function = nn.NLLLoss()
    if OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    #Train
    print('Train with', len(data), 'examples.')
    for epoch in range(EPOCHS):
        print(f'Starting epoch {epoch}.')
        loss_sum = 0
        for sentence, tags in tqdm(data):
            model.zero_grad()
            model.hidden = model.init_hidden()
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, tags)
            loss_sum += loss.data[0]
            #print('Loss, epoch', epoch, '=', loss.data[0])
            loss.backward()
            optimizer.step()

        if epoch % SAVE_EVERY:
            print("epoch = %d, loss = %f" % (epoch, loss_sum / len(data)))
        else:
            save_checkpoint(filename+str(epoch), model, epoch, loss_sum / len(data))

        #Evaluate on dev
        if not epoch % SHOW_ACCURACY:
            accuracy = predict(data_dev, model_=model, idxs=idxs)
            print("Accuracy:", accuracy)


if __name__ == '__main__':
    train()

