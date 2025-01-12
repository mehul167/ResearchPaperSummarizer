# Research Paper Summarizer

A Jupyter notebook implementation of a research paper summarization system that generates concise abstracts using LSTM and transformer models.

## Overview

This project uses both an LSTM-based encoder-decoder architecture and pre-trained transformer models (BART, Pegasus) to summarize research papers into abstracts. The implementation achieves high performance with BLEU score of 0.92 and ROUGE score of 0.97.

## Requirements

```
numpy
pandas
torch
transformers
tensorflow
nltk
rouge
scikit-learn
seaborn
matplotlib
tqdm
```

## Dataset

The project uses a parquet dataset containing research papers and their abstracts. The data should be structured with 'article' and 'abstract' columns.

## Implementation Details

The notebook (`paper_summarizer.ipynb`) contains:

1. Data preprocessing and cleaning
   - Text normalization
   - Tokenization
   - Length analysis and filtering

2. Model implementations:
   - LSTM Encoder-Decoder architecture
   - Fine-tuned BART
   - Pegasus
   - Fine-tuned distilBART

3. Performance evaluation using:
   - BLEU score
   - ROUGE metrics
   - Model comparison visualizations

## Usage

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook paper_summarizer.ipynb
   ```
4. Update the data path in the notebook to point to your dataset
5. Run all cells

## Results

The system achieves:
- BLEU Score: 0.92
- ROUGE Score: 0.97

With comparative analysis between different model architectures shown through visualizations in the notebook.
