import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from csfeatures import morphVec
from config import *

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
OOV = "<OOV>" # Out-of-vocabularie

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
OOV_IDX = 3

def readtrain():
    return read(TRAIN_DATA)

def readdev():
    return read(DEV_DATA)

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

#def make_feature_vector(word, idx):
#    idxs = [float(idx)] + morphVec(word)
#    tensor = torch.FloatTensor(idxs)
#    return Variable(tensor)

# Esta función generará secuencias con embeddings listos para usarse
# a partir del diccionarios definido con prepare_embedding
def prepare(seq, to_index):
    emb = []
    for word in seq:
        try:
            idx=to_index[word]
        except KeyError:
            idx=OOV_IDX
        emb.append(idx)

    #tensor = torch.LongTensor(emb)
    return emb #autograd.Variable(tensor)

def pad_sequences(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor

def sort_batch(batch, ys, lengths):
    seq_lengths, perm_idx = lengths.sort(0,descending=True)
    seq_tensor = batch[perm_idx]
    targ_tensor = ys[perm_idx]
    return seq_tensor.transpose(0, 1), targ_tensor, seq_lengths

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, idxs):
        """Reads source and target and prepare data"""

        [self.tag_to_index, self.word_to_index, self.index_to_tag, self.index_to_word] = idxs

        self.src_seqs = []
        self.trg_seqs = []
        seq_lengths = list() 

        # Vectorize the input data. 
        for sentence, tags in data:
            self.src_seqs.append(self.prepare(sentence, self.word_to_index))
            self.trg_seqs.append(self.prepare(tags, self.tag_to_index))

        seq_lengths = torch.LongTensor([len(s) for s in self.src_seqs])

        self.src_seqs = torch.LongTensor(pad_sequences(self.src_seqs, seq_lengths))
        self.trg_seqs = torch.LongTensor(pad_sequences(self.trg_seqs, seq_lengths))
        
        self.num_total_seqs = len(data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.src_seqs[index],self.trg_seqs[index]

    def __len__(self):
        return self.src_seqs.size(0)

    def prepare(self, seq, to_index, wfeatures=0):
        emb = []
        for word in seq:
            try:
                idx=to_index[word]
            except KeyError:
                idx=OOV_IDX
            if wfeatures:
                vec = make_feature_vector(word, idx)
                emb.append(vec)
            else:
                emb.append(idx)

        tensor = torch.LongTensor(emb)
        #return autograd.Variable(tensor)
        return tensor

def collate_fn(data):

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    #data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, torch.LongTensor(src_lengths), trg_seqs, torch.LongTensor(trg_lengths)

def get_loader(data_raw, idxs):

    # build a custom dataset
    dataset = Dataset(data_raw, idxs)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              collate_fn=collate_fn)

    return data_loader

if __name__ == '__main__':
    var = [make_feature_vector('Hola', 12), make_feature_vector('Hello', 82), make_feature_vector('lindo', 23), ]
    print(var)
    print(torch.LongTensor(var))

