# Title:

Deep Learning Methods for Text and Sequences - Assignment 1 - Gradients Based Learning

# Installation:

Run the following command to install third-party Python libraries:

```console
pip install numpy
```

# Project structure:
    ├── README.md               # This README file
    ├── ass1.pdf                # Assignment instructions
    ├── code                    # Code folder
    │   ├── grad_check.py       # A module for gradients validation checker code
    │   ├── loglinear.py        # A module for log linear classifier
    │   ├── mlp1.py             # A module for MLP neural network with 1 hidden layer classifier
    │   ├── mlpn.py             # A module for MLP neural network with arbitrary n hidden layers classifier
    │   ├── predict_on_test.py  # A module for predict on test dataset module for MLP 1 classifier
    │   ├── train_loglin.py     # The training module for the log linear classifier
    │   ├── train_mlp1.py       # The training module for MLP neural network with 1 hidden layer classifier
    │   ├── train_mlpn.py       # The training module for MLP neural network with arbitrary n hidden layers classifier
    │   └── utils.py            # Utilities module used in other modules
    ├── data                    # Data folder
    │   ├── dev                 # Dev dataset
    │   ├── test                # Test dataset
    │   ├── train               # Train dataset
    │   └── xor_data.py         # XOR dataset
    ├── predictions             # Predictions folder
    │   └── test.pred           # Prediction on test dataset
    └── reports                 # Reports folder
        └── answers.txt         # Answers to the questions in ass1.pdf file

# Data:

The data is for language identification task. The data uses bi-gram features.
The data folder contains train, dev (validation) and test datasets for the
language identification task.
The file format is one training example per line, in the format:

    language<TAB>text

# Usage:

Follow the commands below to run the modules.

## Gradient checks:

### Relevant files:
* grad_check.py

### Command:
```console
python ./code/grad_check.py
```

## Log-linear classifier:

### Relevant files:
* loglinear.py
* train_loglin.py
* utils.py
* data/train
* data/dev

### Command:
```console
python ./code/train_loglin.py
```

## MLP1 classifier:

### Relevant files:
* mlp1.py
* train_mlp1.py
* utils.py
* predict_on_test.py
* data/train
* data/dev

### Command to train:
```console
python ./code/train_mlp1.py
```

### Command to train and predict:
```console
python ./code/predict_on_test.py
```

## MLP-N classifier:

### Relevant files:
* mlpn.py
* train_mlpn.py
* utils.py
* data/train
* data/dev

### Command:
```console
python ./code/train_mlpn.py
```
