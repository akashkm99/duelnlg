import numpy as np
import pickle
import pandas as pd
import json
import argparse
import os


def compute_stats(args):
    feedback_param_dict = json.load(open(args.feedback_config, "r"))
    for feedback_name in feedback_param_dict:
        feedback_config = feedback_param_dict[feedback_name]
        df = pd.read_csv(feedback_config["test"])
        df["Dataset"] = df["Id"].str.split("--", n=0, expand=True)[0]

        def func(df):
            return df.groupby("label")["Id"].nunique() / len(df)

        df = df.groupby("Dataset").apply(func).round(2)
        print("Df: ", df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute untrained metric scores")
    parser.add_argument(
        "--feedback-config",
        type=str,
        help="path to the feedback config file",
        required=True,
    )
    args = parser.parse_args()
    compute_stats(args)
