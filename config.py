
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
WEIGHT_DECAY = 1e-4
SAVE_EVERY = 5
SHOW_ACCURACY = 1
EPOCHS = 100
BATCH_SIZE = 32 
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
LEARNING_RATE = 0.0005 #SDG 0.02 ADAM 0.0005
OPTIMIZER = 'ADAM' #ADAM, SDG

