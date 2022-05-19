import argparse
import pandas as pd
import numpy as np
import os


def merge_and_split(filenames, train_output_file, val_output_file, val_frac=0.1):

    df_list = []
    for filename in filenames.split(","):
        df_list.append(pd.read_csv(filename))
    df = pd.concat(df_list)

    train, validate = np.split(
        df.sample(frac=1, random_state=42), [int((1 - val_frac) * len(df))]
    )

    train.to_csv(train_output_file)
    validate.to_csv(val_output_file)
    print("Written train data to {}".format(train_output_file))
    print("Written validation data to {}".format(val_output_file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess WMT 2014 and WMT 2015 data"
    )
    parser.add_argument(
        "--filenames",
        type=str,
        help="List of comma seperated filenames to merge and split",
    )
    parser.add_argument(
        "--train-output-file", type=str, help="Path to save the output train file",
    )
    parser.add_argument(
        "--val-output-file", type=str, help="Path to save the output validation file",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of the examples to be split as the validation set",
    )
    args = parser.parse_args()
    merge_and_split(
        args.filenames, args.train_output_file, args.val_output_file, args.val_frac
    )
