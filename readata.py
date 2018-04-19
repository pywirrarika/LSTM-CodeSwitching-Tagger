import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F



PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
OOV = "<OOV>" # Out-of-vocabularie

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
OOV_IDX = 3

def readtrain():
    return read('train.enes.txt')

def readdev():
    return read('test_enes.txt')

def read(filename):
    F = open(filename, 'r').read().split('\n')
    data = []
    for line in F:
        wline = [SOS]
        tline = [SOS]
        for word in line.split():
            (word, tag)=word.split('///')
            wline.append(word)
            tline.append(tag)
        if len(wline) != len(tline):
            print("Tokens and words are not equal")
            continue
        if not wline or not tline:
            print("Line empty")
            continue
        wline.append(EOS)
        tline.append(EOS)
        data.append((wline, tline))
    return data
        
# Con esta función hacemos el embedding por palabra de nuestros datos y creamos valores
# enteros para palabras y tags.
def prepare_embedding(data):
    index_to_tag = {}
    index_to_word = {}
    word_to_index = {PAD: PAD_IDX, EOS: EOS_IDX, OOV: OOV_IDX}
    tag_to_index = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
 
    for phrase, tags in data:
        for word in phrase:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
        for tag in tags:
            if tag not in tag_to_index:
                tag_to_index[tag] = len(tag_to_index)
    for key, value in tag_to_index.items():
        index_to_tag[value] = key
    for key, value in word_to_index.items():
        index_to_word[value] = key
    return tag_to_index, word_to_index, index_to_tag, index_to_word

# Esta función generará secuencias con embeddings listos para usarse
# a partir del diccionarios definido con prepare_embedding
def prepare(seq, to_index):
    emb = []
    for word in seq:
        try:
            emb.append(to_index[word])
        except KeyError:
            emb.append(OOV_IDX)

    tensor = torch.LongTensor(emb)
    return autograd.Variable(tensor)



