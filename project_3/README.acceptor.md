# Title:

Deep Learning Methods for Text and Sequences - Assignment 3 - LSTM Acceptor

# Installation:

Run the following command to install third-party Python libraries:
```console
pip install numpy pandas matplotlib torch
```

## Note:
For torch GPU support, follow the instructions here: https://pytorch.org/get-started/locally/.

# Description:

This is generic application for sequence classification tasks.
The application supplies command line interface to control its behavior.
The application supports CPU and GPU. It will run on GPU if GPU is available, otherwise it will run on CPU.

# Data:
Sequence of "good" sequences and "bad" ones.

## Positive examples in the form:
    [1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+

That is, a sequence of random digits, followed by a sequence of a, followed
by another sequence of random digits, followed by a sequence of b followed
by another sequence of digits, then a sequence of c and a another sequence
of random digits, then a sequence of d followed by a final sequence of random
digits.

## Negative examples in the form:
    [1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+

That is, just like before, but this time the sequence of c appear before the
sequence of b.

# Usage:

Follow the command line options to control the application behavior.
There is library module lib_rnn.py to serve as generic library code for this task.
In addition, the application code itself resides in experiment.py with the module code.
Data generation is in gen_examples.py.

See instructions below.

## Data generation application CLI usage:

```console
python ./code/gen_examples.py  --help

usage: gen_examples.py [-h] [--number NUMBER] [--pos-file POS_FILE]
                       [--neg-file NEG_FILE] [--dataset-file DATASET_FILE] [--seed SEED]

Generate examples dataset for the assignment.

    optional arguments:
    -h, --help            show this help message and exit
    --number NUMBER       Number of examples to generate, half positive and half negative.
    --pos-file POS_FILE   Name of the positive examples file.
    --neg-file NEG_FILE   Name of the negative examples file.
    --dataset-file DATASET_FILE
                            Path to the dataset file.
    --seed SEED           Seed for the random number generator.
```

## Sequence Classification application CLI usage:
```console
python ./code/experiment.py --help

usage: experiment.py [-h] [--tag-task {pos_neg,ner,pos}] [--num-workers NUM_WORKERS]
                     [--fit | --load-model-file LOAD_MODEL_FILE]
                     [--save-model-file SAVE_MODEL_FILE] [--predict] [--debug]
                     [--train-file TRAIN_FILE] [--plot-path PLOT_PATH]
                     [--test-file TEST_FILE] [--predict-file PREDICT_FILE]
                     [--dev-ratio DEV_RATIO] [--epochs EPOCHS]
                     [--embedding-dim EMBEDDING_DIM] [--mlp-hidden-dim MLP_HIDDEN_DIM]
                     [--lstm-hidden-dim LSTM_HIDDEN_DIM] [--batch-size BATCH_SIZE]
                     [--log-train] [--log-error] [--lr LR] [--sched-step SCHED_STEP]
                     [--sched-gamma SCHED_GAMMA] [--dropout DROPOUT]
                     [--weight-decay WEIGHT_DECAY]

    Positive and negative tagging sequence classifier

    optional arguments:
    -h, --help            show this help message and exit
    --tag-task {pos_neg,ner,pos}
                            The language tagging task.
    --num-workers NUM_WORKERS
                            The number of workers to use for data loading.
    --fit                 Fit the model.
    --load-model-file LOAD_MODEL_FILE
                            The path to load a trained model from.
    --save-model-file SAVE_MODEL_FILE
                            The path to save the trained model to.
    --predict             Predict for the input file.
    --debug               Run in debug mode, set parameters in the module file.
    --train-file TRAIN_FILE
                            The name of the train file contains the entire train dataset, this
                            will be splitted to train/dev.
    --plot-path PLOT_PATH
                            Path to save plots of fitting statistics.
    --test-file TEST_FILE
                            The path to the test dataset file.
    --predict-file PREDICT_FILE
                            The path to the predictions file.
    --dev-ratio DEV_RATIO
                            The ratio of the dev dataset from the train dataset.
    --epochs EPOCHS       The number of epochs to fit.
    --embedding-dim EMBEDDING_DIM
                            The size of the word embedding vectors.
    --mlp-hidden-dim MLP_HIDDEN_DIM
                            The hidden layer dimension of the MLP layer.
    --lstm-hidden-dim LSTM_HIDDEN_DIM
                            The hidden layer dimension of the LSTM layer.
    --batch-size BATCH_SIZE
                            The batch size to use.
    --log-train           Print logs of the training progress.
    --log-error           Print logs of the train/dev progress error.
    --lr LR               The learning rate to use.
    --sched-step SCHED_STEP
                            The step of the learning rate scheduler to use.
    --sched-gamma SCHED_GAMMA
                            The gamma of the learning rate scheduler to use.
    --dropout DROPOUT     The dropout probability for linear/MLP layers to use.
    --weight-decay WEIGHT_DECAY
                            The weight decay to use.
```

## Note:
The following command examples will run on Linux as they appear in the README.<br>
To run on Windows, replace the CLI line separating character and the folder separating character.

* Line separating character: \ => `
* Folder separating character: / => \

# Commands:

## The relevant files are for gen_examples.py:
* lib_rnn.py
* experiment.py
* gen_examples.py
* data/pos_neg/
* data/pos_neg/train

In order to run part one data generation, run the following command.

```console
python ./code/gen_examples.py \
    --pos-file="data/pos_neg/pos_examples" \
    --neg-file="data/pos_neg/neg_examples" \
    --number=1000 \
    --dataset-file="data/pos_neg/train" \
    --seed=2022
```

## The relevant files are for experiment.py:
* lib_rnn.py
* experiment.py
* data/pos_neg/train

In order to run part one experiment.py, run the following command.

```console
python ./code/experiment.py \
    --fit \
    --log-error \
    --train-file="data/pos_neg/train" \
    --dev-ratio=0.1 \
    --epochs=25 \
    --embedding-dim=30 \
    --mlp-hidden-dim=16 \
    --lstm-hidden-dim=32 \
    --batch-size=16 \
    --lr=0.003 \
    --weight-decay=0 \
    --dropout=0
```