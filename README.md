# TinyGPT

TinyGPT is a **mini GPT-style language model** implemented in **PyTorch**.  
It uses transformer blocks with multi-head self-attention and feed-forward networks to perform **next-word prediction** on a small corpus of text.  

This project is ideal for learning and experimenting with GPT and transformers.

---

## 📂 Project Structure

TinyGPT/
├── data/ # Optional: store corpus or text files
├── models/ # Optional: save trained model checkpoints
├── demo.py # Main script to train and generate text
├── transformer_blocks.py # Transformer block implementation
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── .gitignore # Files to ignore in Git


> Remove temporary files like `tempCodeRunnerFile.py` to keep the repository clean.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/puja6789/TinyGPT.git
cd TinyGPT

2Install dependencies:

pip install -r requirements.txt
Optional: create a virtual environment first for isolated dependencies.

##Usage

Run the demo script to train the model and generate text:

python demo.py

The script trains TinyGPT on a small text corpus.
After training, it can generate text starting from a given word.

📌 Example Output
Generated text:
hello friends how are you <END>

Output may vary depending on training steps and random initialization.

📝 Notes
Uses word-level tokenization.
Each Transformer block contains:
Multi-head self-attention
Feed-forward network
Layer normalization + residual connections
Dataset is small for demonstration; larger datasets improve results.
