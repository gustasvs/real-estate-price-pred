import torch.nn as nn

ENABLE_DEV_LOGS = False

BATCH_SIZE = 2
BATCHES_TO_AGGREGATE = 4

LEARNING_RATE = 1e-4

EPOCHS = 10

LOSS_FUNCT = nn.MSELoss()
