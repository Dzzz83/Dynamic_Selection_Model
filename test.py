import torch
import clip
import ot
import numpy as np

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}") # Should be True if you have a GPU
print(f"CLIP Available: {clip.available_models()[0]}")
print("Optimal Transport (POT) is ready!")