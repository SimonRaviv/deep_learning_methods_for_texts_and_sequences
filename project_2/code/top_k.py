"""
A module to compute the top K similar words using the cosine similarity.
"""
import argparse
import os
import sys

import lib_tagger as lib

MODULE_ROOT_DIR = os.path.dirname(__file__)


def main(cli_args: argparse.Namespace) -> int:
    """
    @brief: Main function.

    @param cli_args: The command line arguments.

    @return: Exit code.
    """
    if cli_args.top_k is None:
        return 1

    # Load the data:
    vocabulary, embedding = lib.load_pretrained_embedding(cli_args.vocabulary_file, cli_args.embedding_file)

    # Get top K similar words:
    top_k = []
    for word in cli_args.top_k_input:
        top_k_similar_words, top_k_scores = lib.get_top_k(word, vocabulary, embedding, cli_args.top_k)
        if len(top_k_similar_words) > 0:
            top_k.append((word, top_k_similar_words, top_k_scores))

    # Print the results:
    print("Top K similar words:")
    for word, top_k_similar_words, top_k_scores in top_k:
        print(f"Word: {word}")
        for i in range(len(top_k_similar_words)):
            print(f"{i+1}. \t{top_k_similar_words[i]:<10}\t{top_k_scores[i]:<10}")

    return 0


if __name__ == '__main__':
    try:
        exit_code = 1
        cli_args = lib.parser_cli(top_k_parser=True)

        # Run in production mode:
        if cli_args.debug is False:
            exit_code = main(cli_args)
            sys.exit(exit_code)
    except Exception as error:
        print(error)
        exit_code = 1
        sys.exit(exit_code)

    # Run in debug mode, ignore all other CLI arguments.
    cli_args = argparse.Namespace(
        debug=True,
        top_k=5,
        top_k_input=["dog", "england", "john", "explode", "office"],
        vocabulary_file=os.path.join(MODULE_ROOT_DIR, "..", "data", "vocab.txt"),
        embedding_file=os.path.join(MODULE_ROOT_DIR, "..", "data", "wordVectors.txt"),
    )

    print(cli_args)
    exit_code = main(cli_args)
    sys.exit(exit_code)
