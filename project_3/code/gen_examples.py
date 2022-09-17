"""
A module to generate examples dataset for the assignment.

Positive sentences in the form:
[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+

Negative sentences in the form:
[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+

Author:
Simon Raviv.
"""

import sys
import argparse
import random


# Constants:
SEED = 2022
MAX_NUMBER_SEQ_LEN = 20
MAX_LETTER_SEQ_LEN = 20
POS_LETTERS_ORDER = ['a', 'b', 'c', 'd']
NEG_LETTERS_ORDER = ['a', 'c', 'b', 'd']
POS_SEQ = "POS"
NEG_SEQ = "NEG"
DATASET_SEPARATOR = '\t'


def generate_single_example_default(sequence_type: str) -> str:
    """
    @brief: Generate a single example.

    @param sequence_type: Type of the sequence, POS_SEQ or NEG_SEQ.

    @return: Generated example.
    """
    letters = POS_LETTERS_ORDER if sequence_type == POS_SEQ else NEG_LETTERS_ORDER
    sequence = ''
    for letter in letters:
        number_sequence = [str(random.randint(1, 9)) for _ in range(random.randint(1, MAX_NUMBER_SEQ_LEN))]
        letter_sequence = letter * random.randint(1, MAX_LETTER_SEQ_LEN)
        sequence += ''.join(number_sequence) + letter_sequence

    number_sequence = [str(random.randint(1, 9)) for _ in range(random.randint(1, MAX_NUMBER_SEQ_LEN))]
    sequence += ''.join(number_sequence)

    return sequence


def generate_single_example_exp_lang(sequence_type: str) -> str:
    """
    @brief: Generate a single example an experimental language.

    @param sequence_type: Type of the sequence, POS_SEQ or NEG_SEQ.

    @return: Generated example.
    """
    sequence = ''
    if sequence_type == POS_SEQ:
        import math
        n = random.randint(1, 10000)
        n = int(math.sqrt(n))
        sequence += 'a' * n
        sequence += 'b' * n
        sequence += 'c' * n
    elif sequence_type == NEG_SEQ:
        n = random.randint(1, 100)
        sequence += 'a' * n
        sequence += 'b' * n
        sequence += 'c' * n

    return sequence


def generate_single_example_exp_lang2(sequence_type: str) -> str:
    """
    @brief: Generate a single example an experimental language 2.

    @param sequence_type: Type of the sequence, POS_SEQ or NEG_SEQ.

    @return: Generated example.
    """
    sequence = ''
    letter = random.choice(POS_LETTERS_ORDER)
    n = random.randint(1, 100)

    if sequence_type == POS_SEQ:
        n = n if n % 2 == 0 else n + 1
    elif sequence_type == NEG_SEQ:
        n = n + 1 if n % 2 == 0 else n

    letter = random.choice(POS_LETTERS_ORDER)
    sequence += letter * n

    return sequence


def generate_examples(language_type: str, number: int, filename: str,
                      generate_single_example_cb: object = generate_single_example_default) -> None:
    """
    @brief: Generate positive/negative examples.

    @param language_type: Type of the language, POS_SEQ or NEG_SEQ.
    @param number: Number of examples to generate.
    @param filename: Name of the examples file.
    @param generate_single_example_cb: Callback to generate a single example.

    @return: None.
    """
    with open(filename, 'w') as f:
        for _ in range(number):
            sequence = generate_single_example_cb(language_type)
            f.write(sequence + '\n')


def generate_dataset(dataset_file: str, number: int,
                     generate_single_example_cb: object = generate_single_example_default) -> None:
    """
    @brief: Generate dataset.

    @param dataset_file: Path to the dataset file.
    @param number: Number of examples to generate.
    @param generate_single_example_cb: Callback to generate a single example.

    @return: None.
    """
    examples = []
    for _ in range(number//2):
        pos_example = (generate_single_example_cb(POS_SEQ), POS_SEQ)
        neg_example = (generate_single_example_cb(NEG_SEQ), NEG_SEQ)
        examples.append(pos_example)
        examples.append(neg_example)

    with open(dataset_file, 'w') as f:
        for example in examples:
            sentence, label = example
            f.write(sentence + DATASET_SEPARATOR + label + '\n')


def parse_cli() -> argparse.Namespace:
    """
    @brief: Parse the command line arguments.

    @return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Generate examples dataset for the assignment.')
    parser.add_argument('--number', type=int, default=1000,
                        help='Number of examples to generate, half positive and half negative.')
    parser.add_argument('--pos-file', type=str, default='pos_examples',
                        help='Name of the positive examples file.')
    parser.add_argument('--neg-file', type=str, default='neg_examples',
                        help='Name of the negative examples file.')
    parser.add_argument('--dataset-file', type=str, default='train',
                        help='Path to the dataset file.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')

    return parser.parse_args()


def main() -> None:
    """
    @brief: Main function for the module.

    @return: Exit code.
    """
    cli_args = parse_cli()
    random.seed(cli_args.seed)
    generate_examples(language_type=POS_SEQ, number=cli_args.number//2, filename=cli_args.pos_file)
    generate_examples(language_type=NEG_SEQ, number=cli_args.number//2, filename=cli_args.neg_file)
    generate_dataset(
        dataset_file=cli_args.dataset_file, number=cli_args.number,
        generate_single_example_cb=generate_single_example_default)

    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
