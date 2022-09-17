# Title:

Deep Learning Methods for Text and Sequences - Assignment 3 - LSTM Acceptor and BiLSTM Tagger

# Description:

This project has 2 README files, one for each main section, see below.

# Project structure:
    .
    ├── README.acceptor.md    # LSTM Acceptor README file
    ├── README.bilstm.md      # BiLSTM Tagger README file
    ├── README.md             # This README file
    ├── ass3.pdf              # Assignment instructions
    ├── code                  # Code folder
    │   ├── bilstmPredict.py  # The application module for BiLSTM tagger prediction application
    │   ├── bilstmTrain.py    # The application module for BiLSTM tagger training application
    │   ├── experiment.py     # The application module for LSTM Acceptor
    │   ├── gen_examples.py   # The module to generate the data for part 1
    │   └── lib_rnn.py        # The library module for this assignment
    ├── data                  # Dataset folder
    │   ├── ner               # NER tagging dataset folder
    │   │   ├── dev           # NER tagging dev dataset
    │   │   ├── test          # NER tagging test dataset
    │   │   └── train         # NER tagging train dataset
    │   ├── pos               # POS tagging dataset folder
    │   │   ├── dev           # POS tagging dev dataset
    │   │   ├── test          # POS tagging test dataset
    │   │   └── train         # POS tagging train dataset
    │   ├── pos_neg           # Positive/negative dataset folder
    │   │   ├── neg_examples  # Negative examples
    │   │   ├── pos_examples  # Positive examples
    │   │   ├── test          # Positive/Negative tagging test dataset
    │   │   └── train         # Positive/Negative tagging train dataset
    │   ├── vocab.txt         # Words vocabulary
    │   └── wordVectors.txt   # Word vectors embeddings
    ├── predictions           # Predictions folder
    │   ├── test4.ner         # The test file prediction over NER tagging task
    │   └── test4.pos         # The test file prediction over POS tagging task
    └── reports               # Reports folder
        ├── challenge.pdf     # The challenge PDF file
        ├── report1.pdf       # The report for part 1
        ├── report2.pdf       # The report for part 2
        └── report3.pdf       # The report for part 3

## LSTM Acceptor:
[LSTM Acceptor README](./README.acceptor.md/ "LSTM Acceptor README")

## BiLSTM Tagger:
[BiLSTM Tagger README](./README.bilstm.md/ "BiLSTM Tagger README")
