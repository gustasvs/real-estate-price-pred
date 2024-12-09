import torch.nn as nn

ENABLE_DEV_LOGS = False
DEMO_MODE = True
# DEMO_MODE = False

BATCH_SIZE = 1
BATCHES_TO_AGGREGATE = 32

LEARNING_RATE = 1e-4
EPOCHS = 4
LOSS_FUNCT = nn.MSELoss()

# AGGREGATION_METHOD = "attention"
AGGREGATION_METHOD = "mean"

WEIGHTED_BIN_COUNT = 20 # by using 1 bin, weighted loss would be the same as MSE loss