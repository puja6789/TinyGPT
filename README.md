# 🚀 TinyGPT – Mini Transformer from Scratch

> > A minimal GPT-style Transformer model built from scratch using PyTorch, capable of generating coherent text after training.
---

## 🧠 Overview

TinyGPT is a lightweight GPT-style language model built from scratch using PyTorch.
The project focuses on understanding the internal working of Transformer-based architectures such as self-attention and sequence modeling.

---

## ✨ Features

* Multi-head self-attention mechanism
* Positional encoding
* Decoder-only Transformer architecture
* Custom training loop
* Text generation from input prompts

---

## ⚙️ Tech Stack

* Python
* PyTorch
* NLP (Natural Language Processing)

---

## 📂 Project Structure

```bash
TinyGPT/
│── train.py        # Model training script
│── model.py        # Transformer architecture
│── generate.py     # Text generation
│── config.py       # Hyperparameters
│── data.py         # Dataset processing
```

---

## ⚙️ Installation

```bash
pip install torch
```

---

## ▶️ How to Run

```bash
# Train the model
python train.py

# Generate text
python generate.py
```

---

## 📊 Sample Output

```bash
Input: hello

Output:
hello friends how are you 
<END> the tea is very hot 
<END> my name is Aarohi
```

---

## 📉 Training Progress

```bash
Step 0 | Loss: 3.9708
Step 100 | Loss: 1.3313
Step 200 | Loss: 0.6241
Step 300 | Loss: 0.2309
Step 400 | Loss: 0.2534
Step 500 | Loss: 0.1664
Step 600 | Loss: 0.1649
Step 700 | Loss: 0.1431
Step 800 | Loss: 0.1278
Step 900 | Loss: 0.0930
Step 1000 | Loss: 0.1281
Step 1100 | Loss: 0.1619
Step 1200 | Loss: 0.0810
Step 1300 | Loss: 0.1245
Step 1400 | Loss: 0.1756
Model saved successfully!
```

---

## ⚙️ How It Works

1. Input text is tokenized into sequences
2. Positional encoding is added to maintain order
3. Transformer processes input using self-attention
4. Model predicts next token step-by-step
5. Final output is generated as coherent text

---

## 🎯 Project Goal

To gain a deep understanding of how GPT-style models work internally by implementing core Transformer components from scratch.

---

## 🚀 Future Improvements

* Add web interface using Streamlit or Flask
* Deploy model for live demo
* Train on larger dataset for better accuracy
* Improve text coherence and performance

---

## 📌 Why This Project Matters

This project demonstrates strong understanding of:

* Transformer architecture
* Deep learning fundamentals
* NLP model development from scratch

---

## 👩‍💻 Author

Puja Kumari
