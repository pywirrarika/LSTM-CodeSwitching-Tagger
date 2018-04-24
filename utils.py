import torch

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
