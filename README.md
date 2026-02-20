# Recurrent Neural Networks in Practice  
## RNN vs LSTM vs GRU — AG News Classification (PyTorch)

By **Jose Azzi** and **Joud Senan**

**Presentation (Canva):** https://www.canva.com/design/DAHBsWkLf1k/pR6fWh87KIbJS3gGqTgAPQ/edit?utm_content=DAHBsWkLf1k&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

This project applies recurrent neural networks to an NLP task using the **AG News** dataset. We implement, train, and evaluate **RNN**, **LSTM**, and **GRU** models, then compare their performance and discuss key architectural differences.

---

## 1) Dataset

**AG News Classification Dataset (Kaggle)**  
- **Size:** 120,000 training samples + 7,600 test samples  
- **Classes (balanced):** World, Sports, Business, Sci/Tech  
- **Format:** CSV with (label, title, description)

**Input construction:** We combine **Title + Description** into one text field for classification.

---

## 2) Models

We compare three recurrent architectures:
- **RNN:** baseline recurrent model (good for short dependencies; struggles with long-term context due to vanishing/exploding gradients).  
- **LSTM:** uses **cell state** + **forget/input/output gates** to preserve longer dependencies.  
- **GRU:** simplified gated variant with **update/reset gates** (often faster with fewer parameters than LSTM).

All models follow the structure:
`Embedding → (RNN/LSTM/GRU) → Dropout → Linear(4 classes)`

---

## 3) Preprocessing

### GRU pipeline (as used in the GRU experiment)
- Combined **Title + Description** into one text column
- Converted class labels to start from **0**
- Text cleaning:
  - lowercasing
  - expanded contractions
  - removed URLs, HTML tags, special characters
  - tokenized using **NLTK WordPunctTokenizer**
- Built vocabulary:
  - counted word frequencies
  - selected top **30,000** most common words
- Converted tokens → indices (word → ID)
- Calculated length distribution and chose **max length using the 95th percentile**
- Padded sequences and converted to PyTorch tensors

### RNN + LSTM pipeline (as used in the RNN/LSTM experiments)
- Loaded CSV and kept (Class Index, Title, Description)
- Combined **Title + Description**
- Converted labels from **1–4 → 0–3**
- Tokenized text by:
  - lowercasing
  - regex tokenizer: **[a-z0-9']+**
- Built vocabulary from training set only:
  - `Counter` word frequencies
  - kept top **30,000** tokens
  - added special tokens: `<pad> = 0`, `<unk> = 1`
- Numericalized tokens → IDs (unknown → `<unk>`)
- Truncated/padded each example to **max_len = 200**
- Converted to PyTorch tensors (`x` token IDs, `y` label)

**Padding handling (key fix):** We track true sequence lengths and use a padding-aware approach so the recurrent model learns from real tokens instead of `<pad>`.

---

## 4) Implementation (PyTorch)

### GRU model implementation
- Custom `GRUClassifier(nn.Module)` with:
  - Embedding → GRU → Dropout → Linear
- Used the **last hidden state** for prediction
- Loss: `CrossEntropyLoss`
- Optimizer: Adam
- DataLoader batching
- Validated after each epoch

### RNN + LSTM implementation
- Custom PyTorch `Dataset` returning:
  - `x`: token IDs after tokenization → numericalization → padding/truncation
  - `y`: label (0–3)
  - `length`: true length (before padding)
- `DataLoader`:
  - shuffle=True (train), shuffle=False (val/test)
- Classifiers as `nn.Module`:
  - `nn.Embedding`
  - recurrent layer (`nn.RNN` or `nn.LSTM`, `batch_first=True`)
  - final hidden state as sequence representation
  - `nn.Dropout`
  - `nn.Linear` to 4 classes
- Training:
  - `CrossEntropyLoss`
  - Adam optimizer
  - evaluate on validation each epoch and keep the best model by validation accuracy
- Final evaluation on test set using **Accuracy** and **Macro-F1**

---

## 5) Hyperparameters

### GRU run
- Vocabulary size: **30,002**
- Embedding dim: **128**
- Hidden dim: **64**
- Batch size: **64**
- Learning rate: **1e-3**
- Weight decay: **1e-4**
- Dropout: **0.5**
- Epochs: **5**
- Validation split: **10%**

### RNN/LSTM run
- Learning rate: **1e-3**
- Optimizer: **Adam**
- Batch size: **64**
- Epochs: **6**
- max_vocab: **30,000**
- max_len: **200**
- embed_dim: **128**
- hidden_dim: **128**
- num_layers: **1**
- dropout: **0.2**
- seed: **42**

---

## 6) Results

### GRU
- **Final test:** Loss **0.2510**, Accuracy **0.9157**

### LSTM
- **Test:** Loss **0.3355**, Accuracy **0.9049**, Macro-F1 **0.9045**

### RNN
- **Test:** Loss **0.6374**, Accuracy **0.7787**, Macro-F1 **0.7775**

---

## 7) Conclusion

- **RNN** performs the weakest due to difficulty learning long dependencies (vanishing/exploding gradients).
- **LSTM** improves performance by controlling memory through gates and stabilizing training.
- **GRU** achieved the best accuracy in our experiments, offering strong results with a simpler gated structure and fewer parameters than LSTM.

---

## 8) How to Run (Typical)

1. Place dataset files under:
   - `data/train.csv`
   - `data/test.csv`
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib torch nltk
   ```
3. Run notebooks (recommended order):
   - Core / preprocessing
   - RNN training
   - LSTM training
   - GRU training

---

## 9) References (Coding + Theory)

**PyTorch Docs**
- DataLoader: https://pytorch.org/docs/stable/data.html
- Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- RNN: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
- LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- pack_padded_sequence: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
- CrossEntropyLoss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Adam: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
- torch.argmax: https://pytorch.org/docs/stable/generated/torch.argmax.html

**pandas**
- DataFrame.iloc: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html

**Dataset**
- AG News (Kaggle): https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

**Theory**
- GRU: https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/
- LSTM vs GRU: https://aicompetence.org/lstm-vs-gru-sequence-processing/
- RNN vs LSTM vs GRU vs Transformers: https://www.geeksforgeeks.org/deep-learning/rnn-vs-lstm-vs-gru-vs-transformers/
