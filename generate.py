import torch
import config
from model import TinyGPT
from data import word2idx, idx2word, vocab_size

model = TinyGPT(vocab_size)
model.load_state_dict(torch.load("tinygpt.pth", map_location="cpu"))
model.eval()

context = torch.tensor([[word2idx["hello"]]])

out = model.generate(context, max_new_tokens=15)

print("Generated text:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))