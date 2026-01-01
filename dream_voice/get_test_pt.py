import torch
import numpy as np

# generate pt file for testing training loop
# input shape is [16, 108]; output shape is [N, 16, 108]
dummy_input = torch.randn(1001, 1, 16, 108)  # (Batch, Channels, Features, Time)
dummy_output = torch.randn(1001, 1, 16, 108)    # (Batch, Features, Time)

print(dummy_input.shape)  # should be (1001, 1, 16, 108)
torch.save(dummy_input, f'dream_voice/test_data/train_loop_test_X_1000.pt')
torch.save(dummy_output, f'dream_voice/test_data/train_loop_test_Y_1000.pt')