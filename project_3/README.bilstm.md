# Title:

Deep Learning Methods for Text and Sequences - Assignment 3 - BiLSTM Tagger

# Installation:

Run the following command to install third-party Python libraries:
```console
pip install numpy pandas matplotlib torch
```

## Note:
For torch GPU support, follow the instructions here: https://pytorch.org/get-started/locally/.

# Description:

This is generic application for sequence word classification tasks.
The application supplies command line interface to control its behavior.
The application supports CPU and GPU. It will run on GPU if GPU is available, otherwise it will run on CPU.

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
There is library module lib_rnn.py to serve as generic library code for this task.
In addition, the application code itself resides in bilstmTrain.py and bilstmPredict.py with the modules code.

See instructions below.

## Sequence Word Classification training application CLI usage:

```console
python ./code/bilstmTrain.py --help

usage: bilstmTrain.py [-h] [--tag-task {pos_neg,ner,pos}] [--num-workers NUM_WORKERS]
                      [--fit | --load-model-file LOAD_MODEL_FILE]
                      [--save-model-file SAVE_MODEL_FILE] [--debug]
                      [--train-file TRAIN_FILE] [--plot-path PLOT_PATH]
                      [--dev-ratio DEV_RATIO] [--epochs EPOCHS]
                      [--embedding-dim EMBEDDING_DIM] [--word-representation {a,b,c,d}]
                      [--lstm-hidden-dim LSTM_HIDDEN_DIM] [--batch-size BATCH_SIZE]
                      [--log-train] [--log-error] [--lr LR] [--sched-step SCHED_STEP]
                      [--sched-gamma SCHED_GAMMA] [--dropout DROPOUT]
                      [--weight-decay WEIGHT_DECAY]

BiLSTM-Tagger for POS/NEG tagging

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
    --debug               Run in debug mode, set parameters in the module file.
    --train-file TRAIN_FILE
                            The name of the train file contains the entire train dataset, this
                            will be splitted to train/dev.
    --plot-path PLOT_PATH
                            Path to save plots of fitting statistics.
    --dev-ratio DEV_RATIO
                            The ratio of the dev dataset from the train dataset.
    --epochs EPOCHS       The number of epochs to fit.
    --embedding-dim EMBEDDING_DIM
                            The size of the word embedding vectors.
    --word-representation {a,b,c,d}
                            The word representation type, following types available: ['(a)
                            Embedding', '(b) Character level LSTM', '(c) Embedding + Subword',
                            '(d) Embedding + Character level LSTM followed by a linear layer'].
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

## Sequence Word Classification prediction application CLI usage:

```console
python ./code/bilstmPredict.py --help

usage: bilstmPredict.py [-h] [--tag-task {pos_neg,ner,pos}] [--num-workers NUM_WORKERS][5/5499]
                        [--fit | --load-model-file LOAD_MODEL_FILE]
                        [--save-model-file SAVE_MODEL_FILE] [--debug]
                        [--train-file TRAIN_FILE] [--plot-path PLOT_PATH]
                        [--dev-ratio DEV_RATIO] [--epochs EPOCHS]
                        [--embedding-dim EMBEDDING_DIM] [--word-representation {a,b,c,d}]
                        [--lstm-hidden-dim LSTM_HIDDEN_DIM] [--batch-size BATCH_SIZE]
                        [--log-train] [--log-error] [--lr LR] [--sched-step SCHED_STEP]
                        [--sched-gamma SCHED_GAMMA] [--dropout DROPOUT]
                        [--weight-decay WEIGHT_DECAY]

BiLSTM-Tagger for POS/NEG tagging

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
    --debug               Run in debug mode, set parameters in the module file.
    --train-file TRAIN_FILE
                            The name of the train file contains the entire train dataset, this
                            will be splitted to train/dev.
    --plot-path PLOT_PATH
                            Path to save plots of fitting statistics.
    --dev-ratio DEV_RATIO
                            The ratio of the dev dataset from the train dataset.
    --epochs EPOCHS       The number of epochs to fit.
    --embedding-dim EMBEDDING_DIM
                            The size of the word embedding vectors.
    --word-representation {a,b,c,d}
                            The word representation type, following types available: ['(a)
                            Embedding', '(b) Character level LSTM', '(c) Embedding + Subword',
                            '(d) Embedding + Character level LSTM followed by a linear layer'].
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
```

## Note:
The following command examples will run on Linux as they appear in the README.<br>
To run on Windows, replace the CLI line separating character and the folder separating character.

* Line separating character: \ => `
* Folder separating character: / => \

# Commands:

The below is examples command.
See bilstm.pdf for the parameters for the needed model.

## Training application:

### The relevant files are for bilstmTrain.py:
* lib_rnn.py
* bilstmTrain.py
* data/pos/train
* data/ner/train

### Commands:
* Change <&#82;EPRESENTATION> to a / b / c / d
* Change <TRAIN_FILE> to data/ner/train / data/pos/train
* Change <MODEL_FILE> to model_name.pt
* Change <TAG_TASK> to ner / pos

In order to run part three bilstmTrain.py, run the following command with the following arguments:

```console
python ./code/bilstmTrain.py \
    <REPRESENTATION> \
    <TRAIN_FILE> \
    <MODEL_FILE> \
    --tag-task=<TAG_TASK> \
    --fit \
    --log-error \
    --log-train \
    --dev-ratio=0.1 \
    --epochs=5 \
    --batch-size=32 \
    --embedding-dim=50 \
    --lstm-hidden-dim=50 \
    --lr=0.008 \
    --sched-step=1 \
    --sched-gamma=1 \
    --weight-decay=0 \
    --dropout=0
```

## Prediction application:

### The relevant files are for bilstmPredict.py:
* lib_rnn.py
* bilstmPredict.py
* data/pos/test
* data/ner/test

### Commands:
* Change <&#82;EPRESENTATION> to a / b / c / d
* Change <INPUT_FILE> to data/ner/test / data/pos/test
* Change <MODEL_FILE> to model_name.pt
* Change <TAG_TASK> to ner / pos

In order to run part three bilstmPredict.py, run the following command with the following arguments:

```console
python ./code/bilstmPredict.py \
    <REPRESENTATION> \
    <MODEL_FILE> \
    <INPUT_FILE> \
    --tag-task=<TAG_TASK> \
    --predict-file="data/predict_file" \
    --batch-size=32
```