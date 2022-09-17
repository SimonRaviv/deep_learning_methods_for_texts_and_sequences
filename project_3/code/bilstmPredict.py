"""
A module to predict a BiLSTM model.

Author:
Simon Raviv.
"""
import os
import sys
import argparse
import torch

import lib_rnn as lib
from bilstmTrain import BiLSTMTagger

# Global variables:
MODULE_ROOT_DIR = os.path.dirname(__file__)


def main(cli_args) -> None:
    """
    @brief: Main function for the module.

    @return: Exit code.
    """
    # Set the seed for reproducibility:
    lib.seed_everything(lib.SEED)

    # Needed to support CUDA:
    torch.multiprocessing.set_start_method('spawn')

    # Set the device to run on:
    device = lib.get_device()

    # Set num of workers for data parallelism:
    num_workers = cli_args.num_workers

    # Load the model:
    if cli_args.load_model_file != "":
        model, dataset_metadata, fit_statistics = lib.load_model(device, cli_args.load_model_file, BiLSTMTagger)
        print(model)

        # Evaluate the model:
        if cli_args.predict is True and cli_args.test_file != "" and cli_args.predict_file != "":
            test_data_loader = lib.load_test_dataset(
                dataset_type=cli_args.tag_task,
                device=device, test_file=cli_args.test_file, num_workers=num_workers,
                batch_size=cli_args.batch_size, metadata=dataset_metadata,
                collate_fn=lib.collate_batch)
            lib.predict(cli_args.tag_task, test_data_loader, model, cli_args.predict_file)

    return 0


if __name__ == "__main__":
    try:
        exit_code = 0
        cli_args = lib.parse_cli(description="BiLSTM-Tagger for POS/NEG tagging", bilstm_predict=True)

        # Run in production mode:
        if cli_args.debug is False:
            exit_code = main(cli_args)
            sys.exit(exit_code)
    except Exception as error:
        print(error)
        exit_code = 1
        sys.exit(exit_code)

    # Run in debug mode, ignore all other CLI arguments.
    tag = "pos"
    repr = "c"
    cli_args = argparse.Namespace(
        debug=True,
        tag_task=tag,
        predict=True,
        test_file=os.path.join(MODULE_ROOT_DIR, "..", "data", tag, "test_verification"),
        predict_file=os.path.join(MODULE_ROOT_DIR, "..", "output", "part_3", tag, f"{tag}_{repr}.pred.test"),
        load_model_file=os.path.join(MODULE_ROOT_DIR, "..", "output", "part_3", tag, f"{tag}_{repr}_model.pt"),
        word_representation=repr,
        batch_size=128,
        num_workers=2)

    print(cli_args)
    exit_code = main(cli_args)
    sys.exit(exit_code)
