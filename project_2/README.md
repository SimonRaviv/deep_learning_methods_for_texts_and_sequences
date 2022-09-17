# Title:

Deep Learning Methods for Text and Sequences - Assignment 2 - Window-based Tagging

# Installation:

Run the following command to install third-party Python libraries:

```console
pip install numpy pandas sklearn matplotlib torch
```

## Notes:
* For torch GPU support, follow the instructions here: https://pytorch.org/get-started/locally/.
* sklearn is used for the CNN filters visualization. Its optional, the application will work
    without it. In case its not installed, the filters visualization will not be supported.

# Description:

This is generic application for language tagging tasks.
It supports POS and NER tagging.
The application supplies command line interface to control its behavior.
The application supports CPU and GPU. It will run on GPU if GPU is available, otherwise it will run on CPU.

# Project structure:
    .
    ├── README.md            # This README file
    ├── ass2.pdf             # Assignment instructions
    ├── code                 # Code folder
    │   ├── lib_tagger.py    # The library module for this assignment
    │   ├── tagger.py        # The application module for this assignment
    │   └── top_k.py         # The top-k module accuracy calculation
    ├── data                 # Dataset folder
    │   ├── ner              # NER tagging dataset folder
    │   │   ├── dev          # NER tagging dev dataset
    │   │   ├── test         # NER tagging test dataset
    │   │   └── train        # NER tagging train dataset
    │   ├── pos              # POS tagging dataset folder
    │   │   ├── dev          # POS tagging dev dataset
    │   │   ├── test         # POS tagging test dataset
    │   │   └── train        # POS tagging train dataset
    │   ├── vocab.txt        # Words vocabulary
    │   └── wordVectors.txt  # Word vectors embeddings
    ├── predictions          # Predictions folder
    │   ├── test1.ner        # NER test prediction file for part 1 of the assignment
    │   ├── test1.pos        # POS test prediction file for part 1 of the assignment
    │   ├── test3.ner        # NER test prediction file for part 3 of the assignment
    │   ├── test3.pos        # POS test prediction file for part 3 of the assignment
    │   ├── test4.ner        # NER test prediction file for part 4 of the assignment
    │   ├── test4.pos        # POS test prediction file for part 4 of the assignment
    │   ├── test5.ner        # NER test prediction file for part 5 of the assignment
    │   └── test5.pos        # POS test prediction file for part 5 of the assignment
    └── reports              # Reports folder
        ├── part1.pdf        # Report for part 1 of the assignment
        ├── part2.pdf        # Report for part 2 of the assignment
        ├── part3.pdf        # Report for part 3 of the assignment
        ├── part4.pdf        # Report for part 4 of the assignment
        └── part5.pdf        # Report for part 5 of the assignment

# Data:

Data input is a sequence of items (in our case, a sentence of natural-language words),
and an output is a label for each of the item.

## Part-of-speech (POS) tagging:

    Input:  The black fox jumped over the lazy rabbit
    Output: DT   JJ    NN   VBD   IN  DT   JJ    NN

## Named-entity-recognition (NER) tagging:

    Input:  John Doe met with Jane on Tuesday in Jerusalem
    Output: PER  PER  O    O   PER  O  TIME    O   LOC

The data folder contains train, dev (validation) and test datasets for each
language tagging task.
The file format is one training example per line,
where sentence is separated by a blank line in the file, in the format:

    word<SPACE>tag

# Usage:

Follow the command line options to control the application behavior.
All parts use the same tagger.py (or top_k.py for part 2) file for the neural network module
and lib_tagger.py for the library with additional supporting code.
See instructions below.

## Tagger application CLI usage:

 ```console
python ./code/tagger.py --help

usage: tagger.py [-h] [--fit] [--debug] [--dataset-path DATASET_PATH]
                 [--tag-type {ner,pos}] [--plot-path PLOT_PATH] [--evaluate]
                 [--test-path TEST_PATH] [--predict-path PREDICT_PATH]
                 [--epochs EPOCHS] [--hidden-dim HIDDEN_DIM]
                 [--batch-size BATCH_SIZE] [--log-train] [--log-error]
                 [--lr LR] [--dropout DROPOUT] [--weight-decay WEIGHT_DECAY]
                 [--sched-step SCHED_STEP] [--sched-gamma SCHED_GAMMA]
                 [--subword-embedding {cbow,cnn}]
                 [--char-embedding-dim CHAR_EMBEDDING_DIM]
                 [--cnn-char-num-filters CNN_CHAR_NUM_FILTERS]
                 [--cnn-char-kernel-size CNN_CHAR_KERNEL_SIZE]
                 [--cnn-char-stride CNN_CHAR_STRIDE]
                 [--window-size WINDOW_SIZE] [--visualize-cnn-filters]
                 [--vocabulary-file VOCABULARY_FILE]
                 [--embedding-file EMBEDDING_FILE]

Language tagging

optional arguments:
    -h, --help            show this help message and exit
    --fit                 Fit the model.
    --debug               Run in debug mode, set parameters in tagger module.
    --dataset-path DATASET_PATH
                            The path to the folder with datasets named train/dev.
    --tag-type {ner,pos}  The language tagging task.
    --plot-path PLOT_PATH
                            Path to save plots of fitting statistics and CNN
                            filters visualization.
    --evaluate            Evaluate the model.
    --test-path TEST_PATH
                            The path to the folder with dataset named test.
    --predict-path PREDICT_PATH
                            The path to save the predictions.
    --epochs EPOCHS       The number of epochs to fit.
    --hidden-dim HIDDEN_DIM
                            The hidden layer dimension of the model.
    --batch-size BATCH_SIZE
                            The batch size to use.
    --log-train           Print logs of the training progress.
    --log-error           Print logs of the train/dev progress error.
    --lr LR               The learning rate to use.
    --dropout DROPOUT     The dropout to use.
    --weight-decay WEIGHT_DECAY
                            The weight decay to use.
    --sched-step SCHED_STEP
                            The step of the learning rate scheduler to use.
    --sched-gamma SCHED_GAMMA
                            The gamma of the learning rate scheduler to use.
    --subword-embedding {cbow,cnn}
                            Type of sub-word embedding to use.
    --char-embedding-dim CHAR_EMBEDDING_DIM
                            The character embedding dimension to use, used with
                            CNN sub-word embedding.
    --cnn-char-num-filters CNN_CHAR_NUM_FILTERS
                            The number of filters to use for CNN sub-word
                            embedding.
    --cnn-char-kernel-size CNN_CHAR_KERNEL_SIZE
                            The kernel height to use for CNN sub-word embedding,
                            width is --char-embedding-dim.
    --cnn-char-stride CNN_CHAR_STRIDE
                            The stride height size to use for CNN sub-word
                            embedding.
    --window-size WINDOW_SIZE
                            The window size to use for the context.
    --visualize-cnn-filters
                            Visualize CNN filters explanation, note this option
                            takes a while to complete.
    --vocabulary-file VOCABULARY_FILE
                            The path to the vocabulary file.
    --embedding-file EMBEDDING_FILE
                            The path to the embedding vectors file.
```

## Top K application CLI usage:

```console
python ./code/top_k.py --help

Top K application CLI usage:
usage: top_k.py [-h] [--debug] [--top-k TOP_K]
                [--top-k-input TOP_K_INPUT [TOP_K_INPUT ...]]
                [--vocabulary-file VOCABULARY_FILE]
                [--embedding-file EMBEDDING_FILE]

Top K

optional arguments:
    -h, --help            show this help message and exit
    --debug               Run in debug mode, set parameters in tagger module.
    --top-k TOP_K         Get top K similarities.
    --top-k-input TOP_K_INPUT [TOP_K_INPUT ...]
                            Get top K similarities for specific input.
    --vocabulary-file VOCABULARY_FILE
                            The path to the vocabulary file.
    --embedding-file EMBEDDING_FILE
                            The path to the embedding vectors file.
```

## Note:
The following command examples will run on Linux as they appear in the README.<br>
To run on Windows, replace the CLI line separating character and the folder separating character.

* Line separating character: \ => `
* Folder separating character: / => \


# Part 1:

## The relevant files are:
* lib_tagger.py
* tagger.py
* data/ner/train
* data/ner/dev
* data/pos/train
* data/pos/dev
* data/ner/test
* data/pos/test
* output - empty folder to store the output files

In order to run part one, run the following commands.

## NER tagging:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="ner" \
    --dataset-path="data/ner" \
    --plot-path="output" \
    --predict-path="output/test1.ner" \
    --test-path="data/ner" \
    --epochs=22 \
    --hidden-dim=64 \
    --batch-size=128 \
    --lr=0.001 \
    --sched-gamma=1 \
    --sched-step=10 \
    --dropout=0.5 \
    --weight-decay=0.0001
```

## POS tagging:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="pos" \
    --dataset-path="data/pos" \
    --plot-path="output" \
    --predict-path="output/test1.pos" \
    --test-path="data/pos" \
    --epochs=45 \
    --hidden-dim=80 \
    --batch-size=128 \
    --lr=0.009 \
    --sched-gamma=0.8 \
    --sched-step=10 \
    --dropout=0.2 \
    --weight-decay=0.00001
```

# Part 2:


## The relevant files are:
* lib_tagger.py
* top_k.py
* data/vocab.txt
* data/wordVectors.txt

In order to run part two, run the following command.
```console
python ./code/top_k.py \
    --top-k=5 \
    --vocabulary-file="data/vocab.txt" \
    --embedding-file="data/wordVectors.txt"
```

# Part 3:

## The relevant files are:
* lib_tagger.py
* tagger.py
* data/ner/train
* data/ner/dev
* data/pos/train
* data/pos/dev
* data/ner/test
* data/pos/test
* data/vocab.txt
* data/wordVectors.txt
* output - empty folder to store the output files

In order to run part three, run the following commands.

## NER tagging:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="ner" \
    --dataset-path="data/ner" \
    --plot-path="output" \
    --predict-path="output/test1.ner" \
    --test-path="data/ner" \
    --vocabulary-file="data/vocab.txt" \
    --embedding-file="data/wordVectors.txt" \
    --epochs=22 \
    --hidden-dim=64 \
    --batch-size=128 \
    --lr=0.001 \
    --sched-gamma=1 \
    --sched-step=10 \
    --dropout=0.5 \
    --weight-decay=0.0001
```

## POS tagging:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="pos" \
    --dataset-path="data/pos" \
    --plot-path="output" \
    --predict-path="output/test1.pos" \
    --test-path="data/pos" \
    --vocabulary-file="data/vocab.txt" \
    --embedding-file="data/wordVectors.txt" \
    --epochs=45 \
    --hidden-dim=80 \
    --batch-size=128 \
    --lr=0.009 \
    --sched-gamma=0.8 \
    --sched-step=10 \
    --dropout=0.2 \
    --weight-decay=0.00001
```

# Part 4:

## The relevant files are:
* lib_tagger.py
* tagger.py
* data/ner/train
* data/ner/dev
* data/pos/train
* data/pos/dev
* data/ner/test
* data/pos/test
* data/vocab.txt
* data/wordVectors.txt
* output - empty folder to store the output files

In order to run part four, run the following commands.

## NER tagging without pretrained embeddings:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="ner" \
    --dataset-path="data/ner" \
    --plot-path="output" \
    --predict-path="output/without_pretrained_embedding_test4.ner" \
    --test-path="data/ner" \
    --vocabulary-file="" \
    --embedding-file="" \
    --subword-embedding="cbow" \
    --epochs=16 \
    --hidden-dim=64 \
    --batch-size=64 \
    --lr=0.001 \
    --sched-gamma=0.85 \
    --sched-step=7 \
    --dropout=0.2 \
    --weight-decay=0.0001
```

## NER tagging with pretrained embeddings:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="ner" \
    --dataset-path="data/ner" \
    --plot-path="output" \
    --predict-path="output/with_pretrained_embedding_test4.ner" \
    --test-path="data/ner" \
    --vocabulary-file="data/vocab.txt" \
    --embedding-file="data/wordVectors.txt" \
    --subword-embedding="cbow" \
    --epochs=16 \
    --hidden-dim=64 \
    --batch-size=64 \
    --lr=0.001 \
    --sched-gamma=0.85 \
    --sched-step=7 \
    --dropout=0.2 \
    --weight-decay=0.0001
```

## POS tagging without pretrained embeddings:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="pos" \
    --dataset-path="data/pos" \
    --plot-path="output" \
    --predict-path="output/without_pretrained_embedding_test4.pos" \
    --test-path="data/pos" \
    --vocabulary-file="" \
    --embedding-file="" \
    --subword-embedding="cbow" \
    --epochs=36 \
    --hidden-dim=64 \
    --batch-size=64 \
    --lr=0.001 \
    --sched-gamma=0.95 \
    --sched-step=15 \
    --dropout=0.2 \
    --weight-decay=0.00001
```

## POS tagging with pretrained embeddings:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="pos" \
    --dataset-path="data/pos" \
    --plot-path="output" \
    --predict-path="output/with_pretrained_embedding_test4.pos" \
    --test-path="data/pos" \
    --vocabulary-file="data/vocab.txt" \
    --embedding-file="data/wordVectors.txt" \
    --subword-embedding="cbow" \
    --epochs=36 \
    --hidden-dim=64 \
    --batch-size=64 \
    --lr=0.001 \
    --sched-gamma=0.95 \
    --sched-step=15 \
    --dropout=0.2 \
    --weight-decay=0.00001
```

# Part 5:

## The relevant files are:
* lib_tagger.py
* tagger.py
* data/ner/train
* data/ner/dev
* data/pos/train
* data/pos/dev
* data/ner/test
* data/pos/test
* data/vocab.txt
* data/wordVectors.txt
* output - empty folder to store the output files

In order to run part five, run the following commands.

## NER tagging:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --log-error \
    --tag-type="ner" \
    --dataset-path="data/ner" \
    --plot-path="output" \
    --predict-path="output/test5.ner" \
    --test-path="data/ner" \
    --vocabulary-file="data/vocab.txt" \
    --embedding-file="data/wordVectors.txt" \
    --subword-embedding='cnn' \
    --epochs=15 \
    --hidden-dim=64 \
    --batch-size=64 \
    --lr=0.002 \
    --sched-gamma=0.5 \
    --sched-step=2 \
    --dropout=0.5 \
    --weight-decay=0.00004 \
    --char-embedding-dim=30 \
    --cnn-char-num-filters=5 \
    --cnn-char-kernel-size=1 \
    --cnn-char-stride=1
```

## POS tagging:

```console
python ./code/tagger.py \
    --fit \
    --evaluate \
    --tag-type="pos" \
    --dataset-path="data/pos" \
    --plot-path="output/" \
    --predict-path="output/test5.pos" \
    --test-path="data/pos" \
    --vocabulary-file="data/vocab.txt" \
    --embedding-file="data/wordVectors.txt" \
    --log-error \
    --epochs=30 \
    --hidden-dim=64 \
    --batch-size=128 \
    --lr=0.0012 \
    --sched-gamma=0.5 \
    --sched-step=10 \
    --dropout=0.5 \
    --weight-decay=0.000005 \
    --subword-embedding='cnn' \
    --char-embedding-dim=30 \
    --cnn-char-num-filters=30 \
    --cnn-char-kernel-size=3 \
    --cnn-char-stride=1
```
