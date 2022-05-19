import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

eps = 1e-9


def cross_entropy(y_true, y_pred):
    loss = -1 * (
        y_true * (np.log(y_pred + eps)) + (1 - y_true) * (np.log((1 - y_pred) + eps))
    )
    return loss


def bald(y_pred):
    y_mean = np.mean(y_pred, axis=1)
    entropy_mean = cross_entropy(y_mean, y_mean)
    entropy = cross_entropy(y_pred, y_pred)
    mean_entropy = np.mean(entropy, axis=1)
    return entropy_mean - mean_entropy


def compute_and_Plot(data):
    data = np.sort(data)[100:-100]
    num_samples = len(data)
    values, base = np.histogram(data, bins=100)
    # evaluate the cumulative
    cumulative = np.cumsum(values) / num_samples
    # plot the cumulative function
    # plt.plot(base[:-1], cumulative, c='blue')

    thresholds = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    for threshold in thresholds:
        xval = base[np.where(cumulative >= threshold)[0][0]]
        yvals = np.arange(0, threshold, 0.01)
        xvals = [xval] * len(yvals)
        # plt.plot(xvals, yvals, c="red")
        print(xval, threshold)

    # plt.show()


def main(fileName):
    df = pd.read_csv(fileName)
    ensemble_metric_names = [
        "bleurt_base_ensemble_{}_Linear_prob".format(i) for i in range(20)
    ]
    predictions = df[ensemble_metric_names]
    bald_scores = bald(predictions)
    print("BALD")
    print("=" * 20)
    compute_and_Plot(bald_scores)

    std_scores = np.std(predictions, axis=1)
    print("Std. dev")
    print("=" * 20)
    compute_and_Plot(std_scores)


if __name__ == "__main__":
    main(sys.argv[1])
