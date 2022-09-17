# Deep Learning Methods for Texts and Sequences BIU 2021 Final Projects

This course teaches the foundation of deep neural networks for text and sequences.

In this course we have implemented several projects, covers the topics listed below.
All the projects implemented as a generic CLI application with a library module.

## Project 1 - Gradient-based Learning
The project implements a log-linear classifier and a multi-layer
perceptron classifier (MLP) with arbitrary-depth from scratch using only NumPy.
It includes the training code for them. All the classifiers will use the hard cross-entropy loss.
The classifiers tested on a language identification task, and the XOR task.

## Project 2 -  Window-based Tagging
This project implements a sequence tagger with window-based features using PyTorch.
It includes word pre-trained embeddings support, and with different
word representations (sub-word units, CNN based sub-word units), on two different tagging tasks.

## Project 3 - RNN Acceptors and BiRNN Transducers
This project implements RNN acceptor and train it on a specific language and
explore the capabilities of the RNN acceptor. In addition, it implements a Bi-LSTM tagger
for language related tagging tasks (i.e. POS, NER).
The project implemented using PyTorch LSTM-Cell, where all data batching implemented in
the project natively without the usage of PyTorch LSTM layer.

## Project 4 - Implementing a Model from the Literature
This project implements and recreates a deep learning model results from the literature,
it includes a detailed report on the process of doing so.
The paper is "A Decomposable Attention Model for Natural Language Inference".

## Usage:
Each project has it's own detailed README.md file with all the details and instructions about the project.

## Repository structure:
    .
    ├── README.md  # This README file
    ├── project_1  # Project 1 related implementation and details
    ├── project_2  # Project 2 related implementation and details
    ├── project_3  # Project 3 related implementation and details
    └── project_4  # Project 4 related implementation and details
