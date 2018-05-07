import pickle

import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from tqdm import tqdm  # Wrap any iterator to show a progress bar.
from utils import save_checkpoint, print_hyperparameters
from readdata import readtrain, readdev, prepare_embedding, prepare, get_loader, sort_batch
from csfeatures import morphVec
from predict import predict
from config import *
from model import LSTMTagger

#def sort_batch(batch, ys, lengths):
#    print('In sort_batch, lengths:',lengths)
#    seq_lengths, perm_idx = lengths.sort(0, descending=True)
#    seq_tensor = batch[perm_idx]
#    targ_tensor = ys[perm_idx]
#    return seq_tensor.transpose(0, 1), targ_tensor, seq_lengths

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
    

    dataset = get_loader(data_raw, idxs)
   

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
    print('Soruce size:', len(word_to_index))
    print('Target size:', len(tag_to_index))
    loss_function = nn.NLLLoss(size_average=True)

    if USE_CUDA:
        model = model.cuda()
    if OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    #Train
    print('Train with', len(data), 'examples.')
    for epoch in range(EPOCHS):
        print(f'Starting epoch {epoch}.')
        loss_sum = 0
        y_true = list()
        y_pred = list()
        for batch, lengths, targets, lengths2 in dataset:
            model.zero_grad()
            #print('Original batch size:',batch.size())
            batch, targets, lengths = sort_batch(batch, targets, lengths)
            pred = model(autograd.Variable(batch), lengths.cpu().numpy())
            #_, preds = torch.max(pred, 1)
            #print('Target size:',targets.size())
            #print('Prediction size:',pred.size())
            loss = loss_function(pred.view(-1, pred.size()[2]), autograd.Variable(targets).view(-1, 1).squeeze(1))
            loss.backward()
            optimizer.step()
            loss_sum += loss.data[0]
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())

        #acc = accuracy_score(y_true, y_pred)
        loss_total=loss_sum / len(dataset)

        #print('Accuracy on test:', acc, 'loss:', loss_total)
        print('>>> Loss:', loss_total)

#        for sentence, tags in tqdm(data):
#            model.zero_grad()
#            model.hidden = model.init_hidden()
#            tag_scores = model(sentence)
#            loss = loss_function(tag_scores, tags)
#            loss_sum += loss.data[0]
#            loss.backward()
#            optimizer.step()
#
#        if epoch % SAVE_EVERY:
#            print("epoch = %d, loss = %f" % (epoch, loss_sum / len(data)))
#        else:
#            save_checkpoint(filename+str(epoch), model, epoch, loss_sum / len(data))
#
#        #Evaluate on dev
#        if not epoch % SHOW_ACCURACY:
#            accuracy = predict(data_dev, model_=model, idxs=idxs)
#            print("Accuracy:", accuracy)
#

if __name__ == '__main__':
    train()

