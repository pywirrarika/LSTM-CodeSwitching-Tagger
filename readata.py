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
    return read('data/train.enes.txt')

def readdev():
    return read('data/test_enes.txt')

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

def make_feature_vector(word, idx):
    idxs = [float(idx)] + morphVec(word)
    tensor = torch.FloatTensor(idxs)
    return Variable(tensor)

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

    tensor = torch.LongTensor(emb)
    return autograd.Variable(tensor)

def prepare_in(seq, to_index, wfeatures=0):
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
    return autograd.Variable(tensor)

# Organize minibaches
def mini_batch(idxs, data):
    batch_x = []
    batch_y = []
    batch_len = 0 # maximum sequence length of a mini-batch
    [tag_to_index, word_to_index, index_to_tag, index_to_word] = idxs 
    for x, y in data:
        print(x, y)
        seq_len = len(x)
        if len(batch_x) == 0: # the first line has the maximum sequence length
            batch_len = seq_len
        pad = [PAD_IDX] * (batch_len - seq_len)
        batch_x.append(x + pad)
        batch_y.append(y + pad)
        if len(batch_x) == BATCH_SIZE:
            data.append((Var(LongTensor(batch_x)), LongTensor(batch_y))) # append a mini-batch
            batch_x = []
            batch_y = []
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, idxs):
        """Reads source and target."""

        [self.tag_to_index, self.word_to_index, self.index_to_tag, self.index_to_word] = idxs

        self.src_seqs = []
        self.trg_seqs = []
        for sentence, tags in data:
            self.src_seqs.append(self.prepare(sentence, self.word_to_index))
            self.trg_seqs.append(self.prepare(tags, self.tag_to_index))

        #self.data = data
        self.num_total_seqs = len(data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        #[src, trg] = self.data[index]
        #src = prepare_in(src, sefl.word_to_index)
        #trg = prepare_in(trg, sefl.word_to_index)
        
        return self.src_seqs[index], self.trg_seqs[index]

    def __len__(self):
        return self.num_total_seqs

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
        return autograd.Variable(tensor)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths

def get_loader(data_raw, idxs):
    """Returns data loader for custom dataset.

    Args:
        src_path: txt file path for source domain.
        trg_path: txt file path for target domain.
        src_word2id: word-to-id dictionary (source domain).
        trg_word2id: word-to-id dictionary (target domain).
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
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

