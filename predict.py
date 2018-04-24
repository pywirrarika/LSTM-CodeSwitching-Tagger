import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import pickle
import sys

from utils import load_checkpoint
from readata import readdev, prepare_embedding, prepare, EOS, SOS
from train import EMBEDDING_DIM, HIDDEN_DIM, LSTMTagger
data = readdev()

with open('data.pickle', 'rb') as f:
   [tag_to_index, word_to_index, index_to_tag, index_to_word] = pickle.load(f)

#print(sys.argv[1])
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_index), len(tag_to_index))
load_checkpoint(sys.argv[1], model)

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
    #print(line[0])
    inputs = prepare(line[0], word_to_index)
    tag_scores = model(inputs)
    #print(tag_scores)
    tags = out_gen(tag_scores, index_to_tag)
    for pred, gold in zip(tags, line[1]):
        if gold in [EOS, SOS]:
            pass
        if pred == gold:
            correct += 1
        total += 1
    #print(tags)
    #print(line[1])

print("Accuracy:", correct/total)

