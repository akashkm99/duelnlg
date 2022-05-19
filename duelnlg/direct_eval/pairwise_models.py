import numpy as np
import pickle
import pandas as pd
import json
import argparse
import os
from utils import (
    bradley_terry_luce,
    linear,
    bradley_terry_luce_logistic,
    cross_entropy,
    kl_divergence,
    accuracy,
    prediction_fn,
)


def tune_gamma(score1, score2, label, search_space, link_fn):

    label = label.reshape(-1, 1)
    loss_list = []
    for gamma in search_space:
        probs = link_fn(score1, score2, gamma)
        loss = kl_divergence(label, probs)
        loss_list.append(loss)

    losses = np.stack(loss_list, axis=0)
    best_idx = np.argmin(losses, axis=0)
    best_gamma = search_space[best_idx]
    return best_gamma


def get_gamma(score1, score2, label, linkfn_name):

    gamma = None

    if linkfn_name == "Linear":
        diff = score1 - score2
        gamma = 2 * np.amax(np.absolute(diff), axis=0).reshape(1, -1)

    elif linkfn_name == "BTL":
        gamma = np.maximum(np.amin(score1, axis=0), np.amin(score2, axis=0)).reshape(
            1, -1
        )

    elif linkfn_name == "BTL-Logistic":
        search_space = np.linspace(0.005, 1, 1000)
        gamma = tune_gamma(
            score1, score2, label, search_space, bradley_terry_luce_logistic
        ).reshape(1, -1)

    return gamma


def cartesian_product(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def tune_thesholds(probs, label):

    label = label.reshape(-1, 1)
    search_space = cartesian_product(
        np.linspace(0.4, 0.5, 100), np.linspace(0.5, 0.6, 100)
    )
    num_metrics = probs.shape[1]
    best_acc = np.zeros([num_metrics])
    best_idx = np.zeros([num_metrics], dtype=np.int32)

    for idx, (thresh1, thresh2) in enumerate(search_space):
        acc = accuracy(label, probs, lower_threshold=thresh1, upper_threshold=thresh2)
        best_idx = np.where(acc > best_acc, idx * np.ones_like(best_idx), best_idx)
        best_acc = np.where(acc > best_acc, acc, best_acc)

    best_thresh = search_space[best_idx]
    return best_thresh, best_acc


def direct_to_pairwise(data, test_path, best_threshs=None, gammas=None):

    link_functions = [
        ("BTL", bradley_terry_luce),
        ("BTL-Logistic", bradley_terry_luce_logistic),
        ("Linear", linear),
    ]

    input_df = pd.read_csv(test_path)
    metric_names = data["test"]["metric_names"]
    num_metrics = len(metric_names)
    results = []
    overall_predictions = []

    gamma_list = []
    best_thresh_list = []

    for i, (linkfn_name, link_fn) in enumerate(link_functions):

        if gammas is None:
            gamma = get_gamma(
                data["val"]["score1"],
                data["val"]["score2"],
                data["val"]["label"],
                linkfn_name,
            )
        else:
            gamma = gammas[i]

        dev_probs = link_fn(data["val"]["score1"], data["val"]["score2"], gamma=gamma)
        test_probs = link_fn(
            data["test"]["score1"], data["test"]["score2"], gamma=gamma
        )
        test_loss = kl_divergence(data["test"]["label"].reshape(-1, 1), test_probs)

        if best_threshs is None:
            best_thresh, _ = tune_thesholds(dev_probs, data["val"]["label"])
        else:
            best_thresh = best_threshs[i]

        test_acc = []
        test_predictions = []
        for i in range(num_metrics):
            test_acc.append(
                accuracy(
                    data["test"]["label"],
                    test_probs[:, i],
                    lower_threshold=best_thresh[i][0],
                    upper_threshold=best_thresh[i][1],
                )
            )
            test_predictions.append(
                prediction_fn(
                    test_probs[:, i],
                    lower_threshold=best_thresh[i][0],
                    upper_threshold=best_thresh[i][1],
                )
            )

        test_predictions = np.stack(test_predictions, axis=1)
        test_probs_df = pd.DataFrame(
            test_probs,
            columns=[
                col_name + "_{}_prob".format(linkfn_name) for col_name in metric_names
            ],
        )
        test_pred_df = pd.DataFrame(
            test_predictions,
            columns=[
                col_name + "_{}_predictions".format(linkfn_name)
                for col_name in metric_names
            ],
        )
        test_df = pd.concat([test_probs_df, test_pred_df], axis=1)
        test_df["label"] = data["test"]["label"]

        def result_func(df):
            y_true = df["label"].values
            data = {
                "Metric": [],
                "Link Function": [],
                "KL Divergence": [],
                "Accuracy": [],
                "Thresholds": [],
                "Gamma": [],
            }

            for idx, metric_name in enumerate(metric_names):
                column_name_pred = "{}_{}_predictions".format(metric_name, linkfn_name)
                column_name_prob = "{}_{}_prob".format(metric_name, linkfn_name)

                y_pred = df[column_name_pred].values
                acc = accuracy(y_true, y_pred)
                data["Accuracy"].append(acc)

                y_prob = df[column_name_prob].values
                kl_div = kl_divergence(y_true, y_prob)
                data["KL Divergence"].append(kl_div)

            data["Metric"].extend(metric_names)
            data["Link Function"].extend([linkfn_name] * num_metrics)
            data["Thresholds"].extend(best_thresh.tolist())
            data["Gamma"].extend(gamma.reshape(-1).tolist())

            return pd.DataFrame(data)

        result_df = result_func(test_df)
        results.append(result_df)
        test_df = test_df.drop(columns=["label"])
        overall_predictions.append(test_df)

        gamma_list.append(gamma)
        best_thresh_list.append(best_thresh)

    results = pd.concat(results, axis=0)
    overall_predictions = pd.concat(overall_predictions, axis=1)
    final_df = pd.concat([input_df, overall_predictions], axis=1)
    return results, final_df, metric_names
