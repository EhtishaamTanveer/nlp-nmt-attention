# Neural Machine Translation with Attention Mechanism (English to Portuguese)

This repository implements a Neural Machine Translation (NMT) model using an encoder-decoder architecture with attention mechanism for translating English sentences to Portuguese. The model is built from scratch using TensorFlow and addresses the vanishing gradient problem commonly encountered in RNNs through the use of LSTMs and attention.

## Key Points

- **Encoder-decoder with Attention:** This project implements an encoder-decoder architecture with attention mechanism. The encoder processes the source language sentence (English) and generates a context vector. The decoder, with the help of the attention mechanism, focuses on relevant parts of the encoder output while generating the target language sentence (Portuguese).
- **Built from scratch with TensorFlow:** The entire NMT model is built from scratch using TensorFlow, allowing for customization and deeper understanding of the underlying architecture.
- **Greedy and MBR Decoding:** The project explores both greedy decoding and Minimum Bayes Risk (MBR) decoding for generating translations. Greedy decoding selects the most probable word at each step, while MBR decoding considers a sequence of words and aims to minimize the overall risk of translation errors.

## Model Architecture
### Encoder-Decoder Architecture with Attention

The model consists of two main parts:

  1. Encoder: Processes the input sequence (English) and encodes it into a fixed-length context vector. The encoder uses an LSTM layer to process the sequence.
  2. Decoder: Takes the context vector and generates the output sequence (Portuguese) step by step. The decoder also uses an LSTM, but it receives the context vector and previous words as input at each step.

The Attention Mechanism is introduced in the decoder, allowing the model to focus on different parts of the input sequence at each decoding step. This helps mitigate the vanishing gradient problem and improves performance, especially for longer sentences.

## Project Structure

The repository contains the following files and folders:

- nlp-nmt-attention.ipynb: This Jupyter notebook is the main script for training, evaluating, and generating translations using the NMT model.
- utils.py: This Python file contains various utility functions used throughout the project, such as data preprocessing, tokenization, and padding functions.
- w1_unittest.py: This file contains unit tests for the functions implemented in utils.py. It's good practice to include unit tests for ensuring the correctness of your code.
- por.txt: the dataset containing English and Portuguese sentences

## Getting Started

### Prerequisites:

  Python 3.0 and above
  TensorFlow (>2.0)
  Jupyter Notebook

### Installation:

1. Clone this repository:
```
git clone https://github.com/<your-username>/Neural-Machine-Translation-with-Attention-Mechanism.git
```
2. Navigate to the project directory:
```
cd Neural-Machine-Translation-with-Attention-Mechanism
```
3. Install required dependencies:
```
pip install tensorflow numpy collections tensorflow_text pathlib math itertools dlai_grader
```

## Running the Project:

1. Open nlp-nmt-attention.ipynb in Jupyter Notebook.
2. Follow the instructions within the notebook to:
    - Download and pre-process the English-Portuguese dataset.
    - Train the NMT model.
    - Evaluate the model's performance on a validation set.
    - Generate translations using greedy and MBR decoding strategies.

## Example Usage

Here's an example of how you can use the trained model to translate an English sentence into Portuguese:

```
english_sentence = "I love languages"

translation, candidates = mbr_decode(trained_translator, english_sentence, n_samples=10, temperature=0.6)

print("Translation candidates:")
for c in candidates:
    print(c)

print(f"\nSelected translation: {translation}")
```

Output looks like this:

```
Selected translation: eu amo idiomas
```
## Conclusion

This project demonstrates how to build a Neural Machine Translation system with an attention mechanism from scratch using TensorFlow. The attention mechanism enhances the model’s performance by allowing it to focus on different parts of the input sequence, addressing challenges faced by traditional RNNs and LSTMs in sequence-to-sequence tasks.

## Acknowledgements

- The attention mechanism used in this project is inspired by the work of Bahdanau et al. in their paper Neural Machine Translation by Jointly Learning to Align and Translate (2014).
- TensorFlow for providing the tools to build and train deep learning models.
- We thank [Coursera](https://www.coursera.org/) for providing the dataset of English and Portuguese Sentences
    
## Contributing

We welcome contributions to this project! Feel free to fork the repository, make changes, and submit pull requests


