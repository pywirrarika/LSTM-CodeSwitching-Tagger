import torch
from config import *

def load_checkpoint(filename, model = None, g2c = False):
    print("loading model...")
    if g2c: # load weights into CPU
        checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, verbose=False):
    if verbose:
        print("saving model...")
    checkpoint = {}
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["loss"] = loss
    torch.save(checkpoint, filename + ".epoch%d" % epoch)
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))

def print_hyperparameters():

    print("-"*30)
    print("Number of Layers",NUM_LAYERS)
    print("Dropout Rate",DROPOUT)
    print("Bidirectional?",BIDIRECTIONAL)
    print("Weight of decay",WEIGHT_DECAY)
    print("Save every epoch",SAVE_EVERY)
    print("Show accuracy each epoch",SHOW_ACCURACY)
    print("Epochs",EPOCHS)
    print("Batch size",BATCH_SIZE)
    print("Emmbeding dim",EMBEDDING_DIM)
    print("Hidden dim",HIDDEN_DIM)
    print("Lerning rate",LEARNING_RATE)
    print("Optimizer",OPTIMIZER)
    print("-"*30)
