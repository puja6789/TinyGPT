import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_blocks import Block


# Force CPU device
device = torch.device("cpu")

print("Torch version:", torch.__version__)
print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())  # Just for info
print("GPU name:", "None (CPU mode)")

corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
print (text)

words = list(set(text.split()))
print(words)

vocab_size = len(words)
print(vocab_size)