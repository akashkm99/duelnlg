import numpy as np
import os
import glob
import pickle


def bradley_terry_luce(score1, score2, gamma=0, eps=1e-9):
    score1 = np.maximum(score1 - gamma, np.zeros_like(score1))
    score2 = np.maximum(score2 - gamma, np.zeros_like(score2))
    return score1 / (score1 + score2 + eps)


def linear(score1, score2, gamma=1.0, eps=1e-9):
    delta = np.clip((score1 - score2) / (gamma + eps), -0.5, 0.5)
    return 0.5 + delta


def bradley_terry_luce_logistic(score1, score2, gamma=1.0):
    delta = (score1 - score2) / gamma
    return 1 / (1 + np.exp(-delta))


def cross_entropy(y_true, y_pred, eps=1e-9):

    assert np.all(y_pred <= 1) and np.all(y_pred >= 0)
    loss = -1 * (
        y_true * (np.log(y_pred + eps)) + (1 - y_true) * (np.log((1 - y_pred) + eps))
    )
    return np.mean(loss, axis=0)


def kl_divergence(y_true, y_pred):
    return cross_entropy(y_true, y_pred) - cross_entropy(y_true, y_true)


def accuracy(y_true, probs, lower_threshold=1.0 / 3, upper_threshold=2.0 / 3):

    accuracy_0 = (y_true == 0) & (probs <= lower_threshold)
    accuracy_1 = (y_true == 1) & (probs >= upper_threshold)
    accuracy_half = (y_true == 0.5) & (
        (probs > lower_threshold) & (probs < upper_threshold)
    )
    accuracy = accuracy_0 | accuracy_1 | accuracy_half
    return np.mean(accuracy, axis=0)


def prediction_fn(probs, lower_threshold=1.0 / 3, upper_threshold=2.0 / 3):

    prediction = np.zeros_like(probs)
    prediction[probs > upper_threshold] = 1
    prediction[(probs < upper_threshold) & (probs > lower_threshold)] = 0.5
    return prediction


def update_processed(processed_dir, predictions, metric_names):

    predictions = predictions.set_index("unique_id")
    link_functions = ["Linear", "BTL", "BTL-Logistic"]
    output_types = ["prob", "predictions"]

    keys = []
    for metric_name in metric_names:
        for link_function in link_functions:
            for output_type in output_types:
                keys.append("{}_{}_{}".format(metric_name, link_function, output_type))

    print(
        "Updating the processed files with the following metric predictions/probabilities .."
    )
    print(keys)

    for filename in glob.glob(os.path.join(processed_dir, "*.pkl")):
        fp = open(filename, "rb")
        data = pickle.load(fp)
        sample_dict = data["samples"]
        for (sys1, sys2), samples in sample_dict.items():
            for sample in samples:
                unique_id = sample["unique_id"]
                if unique_id == "":
                    continue
                try:
                    values = predictions.loc[unique_id, keys].values.tolist()
                    sample.update(dict(zip(keys, values)))
                except:
                    continue
        fp.close()
        with open(filename, "wb") as fp:
            pickle.dump(data, fp)
        print("Updated processed file: {}".format(filename))
