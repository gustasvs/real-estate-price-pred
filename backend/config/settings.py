import torch.nn as nn

ENABLE_DEV_LOGS = False

BATCH_SIZE = 1
BATCHES_TO_AGGREGATE = 32

LEARNING_RATE = 1e-4

EPOCHS = 4

LOSS_FUNCT = nn.MSELoss()
