import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import pickle
import sys

from utils import load_checkpoint
from readdata import readdev, prepare_embedding, prepare, EOS, SOS
from config import *
from model import LSTMTagger

def predict(data, model_name='', model_=None, idxs=None):

    if idxs:
        [tag_to_index, word_to_index, index_to_tag, index_to_word] = idxs
    else:
        with open('data.pickle', 'rb') as f:
           [tag_to_index, word_to_index, index_to_tag, index_to_word] = pickle.load(f)

    if model_name:
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_index), len(tag_to_index))
        load_checkpoint(model_name, model)
    elif not model_:
        raise ValueError('No model specified.')
    else:
        model = model_

    def out_gen(tag_scores, index_to_tag):
        tags = []
        for word in tag_scores:
            maxval = -10000
            maxind = 0
            for i in range(word.size()[0]):
                if maxval < word.data[i]:
                    maxind = i
                    maxval = word.data[i]
            tags.append(index_to_tag[maxind])
        return tags


    correct = 0
    total = 0
    for line in data:
        inputs = prepare(line[0], word_to_index)
        tag_scores = model(inputs)
        tags = out_gen(tag_scores, index_to_tag)
        for pred, gold in zip(tags, line[1]):
            if gold in [EOS, SOS]:
                pass
            if pred == gold:
                correct += 1
            total += 1

    return correct/total

if __name__ == '__main__':
    data = readdev()
    accuracy = predict(data, model_name=sys.argv[1])
    print("Accuracy:", accuracy)

