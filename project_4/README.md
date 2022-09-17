# Title:

Deep Learning Methods for Text and Sequences - Assignment 4 - A Decomposable Attention Model for Natural Language Inference Paper

# Installation:

Run the following command to install third-party Python libraries:

```console
pip install numpy matplotlib torch h5py
```

## Note:
For torch GPU support, follow the instructions here: https://pytorch.org/get-started/locally/.

# Description:

This is generic application for results reproduction of the following paper:
* Decomposable Attention in Neural Networks for Natural Language Inference
* https://arxiv.org/pdf/1606.01933v2.pdf

The application supplies command line interface to control its behavior.
The application supports CPU and GPU. It will run on GPU if GPU is available, otherwise it will run on CPU.

# Project structure:
    .
    ├── README.md          # This README file
    ├── code               # Code folder
    │   ├── datten.py      # The application module for this assignment
    │   └── lib_datten.py  # The library module for this assignment
    └── reports            # Reports folder
        ├── datten.log     # Training log file of the submitted model
        └── report.pdf     # Report file

# Data:

## Dataset URLs:
* Dataset file URL: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
* GLOVE embedding file URL: https://nlp.stanford.edu/data/glove.840B.300d.zip

## Follow the steps below to get the data:
1. Download the dataset file from the URL above.
2. Unzip the dataset file to data/snli directory.
3. Download the GLOVE embedding file from the URL above.
4. Unzip the GLOVE embedding file to data/glove directory.

# Usage:

Follow the command line options to control the application model behavior.
There is library module lib_datten.py to serve as generic library code for this task.
In addition, the application code itself resides in datten.py with the models code.
The application has 2 subcommands: One for data preprocessing and one for training.

See instructions below.

## Main CLI:

```console
python ./code/datten.py --help

usage: datten.py [-h] {data,model} ...

A Decomposable Attention Model for Natural Language Inference

optional arguments:
    -h, --help    show this help message and exit

    subcommands:
    {data,model}
        data        Data processing command.
        model       Model command.
```

## Data Preprocessing:

```console
python ./code/datten.py data --help

usage: datten.py data [-h] [--debug] [--log-file LOG_FILE] [--no-log-console]
                      [--snli-path SNLI_PATH] [--glove-path GLOVE_PATH]
                      [--output-path OUTPUT_PATH] [--batch-size BATCH_SIZE] [--shuffle]
                      [--max-sequence-length MAX_SEQUENCE_LENGTH]

optional arguments:
    -h, --help            show this help message and exit
    --debug               Run in debug mode, set parameters in the module file.
    --log-file LOG_FILE   The file path to log the output to.
    --no-log-console      Don't to log to console.
    --snli-path SNLI_PATH
                            The path to the unzipped SNLI snli_1.0 dataset folder.
    --glove-path GLOVE_PATH
                            The path to the unzipped GLOVE 840B.300d word embeddings file.
    --output-path OUTPUT_PATH
                            The path to the output folder.
    --batch-size BATCH_SIZE
                            The batch size to use.
    --shuffle             Shuffle the data.
    --max-sequence-length MAX_SEQUENCE_LENGTH
                            The maximum sequence length to use, other sequences will be dropped.
```

## Model Training:

```console
python ./code/datten.py model --help

usage: datten.py model [-h] [--debug] [--log-file LOG_FILE] [--no-log-console] [--log-train]
                       [--log-train-interval LOG_TRAIN_INTERVAL] [--log-epoch] [--log-norms]
                       [--fit | --load-model-path LOAD_MODEL_PATH] [--save-model-path SAVE_MODEL_PATH]
                       [--evaluate] [--train-file TRAIN_FILE] [--dev-file DEV_FILE]
                       [--test-file TEST_FILE] [--results-break-down] [--word-embeddings-file WORD_EMBEDDINGS_FILE]
                       [--plot-path PLOT_PATH] [--gaussian-std GAUSSIAN_STD]
                       [--initial-accumulator-value INITIAL_ACCUMULATOR_VALUE] [--shuffle] [--epochs EPOCHS]
                       [--embedding-dim EMBEDDING_DIM] [--mlp-hidden-dim MLP_HIDDEN_DIM] [--lr LR] 
                       [--dropout DROPOUT] [--weight-decay WEIGHT_DECAY] [--max-grad-norm MAX_GRAD_NORM]

optional arguments:
    -h, --help            show this help message and exit
    --debug               Run in debug mode, set parameters in the module file.
    --log-file LOG_FILE   The file path to log the output to.
    --no-log-console      Don't to log to console.
    --log-train           Print logs of the fitting progress.
    --log-train-interval LOG_TRAIN_INTERVAL
                            The interval to print the logs of the fitting progress during training.
    --log-epoch           Print logs of the fitting train/dev progress.
    --log-norms           Print logs of the parameters and gradients normals.
    --fit                 Fit the model.
    --load-model-path LOAD_MODEL_PATH
                            The path to the model to load.
    --save-model-path SAVE_MODEL_PATH
                            The path to save the best models to.
    --evaluate            Evaluate on test file.
    --train-file TRAIN_FILE
                            The name of the train file contains the train dataset.
    --dev-file DEV_FILE   The name of the dev file contains the dev dataset.
    --test-file TEST_FILE
                            The name of the test file contains the test dataset.
    --results-break-down  Print accuracy per label during test time.
    --word-embeddings-file WORD_EMBEDDINGS_FILE
                            The path to GLOVE word embeddings file.
    --plot-path PLOT_PATH
                            Path to save plots of fitting statistics.
    --gaussian-std GAUSSIAN_STD
                            The standard deviation of the gaussian initialization, mean is 0.
    --initial-accumulator-value INITIAL_ACCUMULATOR_VALUE
                            The initial accumulator value for the Adagrad optimizer.
    --shuffle             Shuffle the data before each epoch.
    --epochs EPOCHS       The number of epochs to fit.
    --embedding-dim EMBEDDING_DIM
                            The size of the word embedding vectors.
    --mlp-hidden-dim MLP_HIDDEN_DIM
                            The hidden layer dimension of the MLP layer.
    --lr LR               The learning rate to use.
    --dropout DROPOUT     The dropout probability to use.
    --weight-decay WEIGHT_DECAY
                            The weight decay to use.
    --max-grad-norm MAX_GRAD_NORM
                            The maximum gradient norm to use, gradients over this value will be clipped.
```

## Note:
The following command examples will run on Linux as they appear in the README.<br>
To run on Windows, replace the CLI line separating character and the folder separating character.

* Line separating character: \ => `
* Folder separating character: / => \

# Commands:

Use the commands below to reproduce the results in the report.


## Data Preprocessing:


This command will pre-process the SNLI dataset and save it to the output folder.
Note that it will take several minutes to complete the run.

## Relevant files:
* lib_datten.py
* datten.py
* output_dir/
* output_dir/datten.log
* data/processed_dataset/
* data/snli/snli_1.0/snli_1.0_train.txt
* data/snli/snli_1.0/snli_1.0_dev.txt
* data/snli/snli_1.0/snli_1.0_test.txt
* data/glove/glove.840B.300d.txt

## Command:
```console
python ./code/datten.py \
    data \
    --log-file=output_dir/datten.log \
    --snli-path=data/snli/snli_1.0 \
    --glove-path=data/glove/glove.840B.300d.txt \
    --output-path=data/processed_dataset \
    --batch-size=32
```

## Model Training:


This command will train the model and will evaluate the model on the test set.
In addition, it will save the best models during the training and save training history plots.

### Relevant files:
* lib_datten.py
* datten.py
* output_dir/
* output_dir/datten.log
* data/processed_dataset/train.hdf5
* data/processed_dataset/dev.hdf5
* data/processed_dataset/test.hdf5
* data/processed_dataset/glove.hdf5

## Command:
```console
python ./code/datten.py \
    model \
    --log-file=output_dir/datten.log \
    --log-train \
    --log-train-interval=1000 \
    --log-epoch \
    --log-norms \
    --fit \
    --save-model-path=output_dir \
    --evaluate \
    --train-file=data/processed_dataset/train.hdf5 \
    --dev-file=data/processed_dataset/dev.hdf5 \
    --test-file=data/processed_dataset/test.hdf5 \
    --results-break-down \
    --word-embeddings-file=data/processed_dataset/glove.hdf5 \
    --plot-path=output_dir \
    --initial-accumulator-value=0 \
    --gaussian-std=0.01 \
    --shuffle \
    --epochs=250 \
    --embedding-dim=300 \
    --mlp-hidden-dim=300 \
    --lr=0.05 \
    --dropout=0.2 \
    --weight-decay=0.00001 \
    --max-grad-norm=5
```

## Model Evaluation:

This command will evaluate the model on the test set.

### Relevant files:
* lib_datten.py
* datten.py
* output_dir/
* output_dir/datten.log
* best_model_file.pt
* data/processed_dataset/test.hdf5
* data/processed_dataset/glove.hdf5

## Command:
```console
python ./code/datten.py \
    model \
    --log-file=output_dir/datten.log \
    --load-model-path=best_model_file.pt \
    --evaluate \
    --test-file=data/processed_dataset/test.hdf5 \
    --results-break-down \
    --word-embeddings-file=data/processed_dataset/glove.hdf5
```

# Evaluate My Model:

In this section I present a way to download the trained model and evaluate it on the test set/dev.
The training log file attached to the submission, contains the training process of this model.

Model URL: https://drive.google.com/uc?export=download&id=1cIpZjfZwvlgPSefXUCe1RIgKH6N6jiKe

## Steps:
1. Download the model file and place it in <DOWNLOAD_PATH>.
2. Use the same relevant file from the previous section to evaluate the model.
3. Run the command below, replace <DOWNLOAD_PATH> with the path to the model file.
### My Model Evaluation Command:
```console
python ./code/datten.py \
    model \
    --load-model-path=<DOWNLOAD_PATH>/DecomposableAttentionModel_epoch_198_dev_accuracy_86.10.pt \
    --evaluate \
    --test-file=data/processed_dataset/test.hdf5 \
    --results-break-down \
    --word-embeddings-file=data/processed_dataset/glove.hdf5
```