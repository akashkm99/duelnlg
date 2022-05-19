import numpy as np
import pandas as pd
import os
import codecs
import pickle
import argparse


def prepare_submissions_reference(reference_file, system_dir):

    data = {"reference": []}
    with codecs.open(reference_file, "r", encoding="utf-8") as fp:

        for line in fp.readlines():
            data["reference"].append(line.strip())

    systems = os.listdir(system_dir)
    for system in systems:
        with codecs.open(os.path.join(system_dir, system), "r", encoding="utf-8") as fp:
            data[system] = []
            for line in fp.readlines():
                data[system].append(line.strip())
    return data


def _create_samples_dict(num_arms):

    samples_dict = {}
    for i in range(num_arms):
        samples_dict[(i, i)] = [
            {
                "unique_id": "",
                "srcIndex": -1,
                "system1Text": "",
                "system2Text": "",
                "referenceText": "",
                "label": 0.5,
            }
        ]
        for j in range(i + 1, num_arms):
            samples_dict[(i, j)] = []
            samples_dict[(j, i)] = []
    return samples_dict


def _create_arm_mappings(arm_names):

    arms = list(range(len(arm_names)))
    arm_to_idx = {}
    idx_to_arm = {}

    for arm in arms:
        idx = arm
        name = arm_names[arm]
        idx_to_arm[idx] = name
        arm_to_idx[name] = idx

    return arms, idx_to_arm, arm_to_idx


def _prepare_data(
    data, reference_file, system_dir, human_judgements_file, processed_dir, year
):

    system_outputs = prepare_submissions_reference(reference_file, system_dir)
    df = pd.read_csv(human_judgements_file)
    total_samples = len(df)

    arm_names = np.unique(
        np.concatenate([df["system1Id"].unique(), df["system2Id"].unique()])
    )
    arm_names = [i.replace(".txt", "") for i in arm_names]
    arms, idx_to_arm, arm_to_idx = _create_arm_mappings(arm_names)
    num_arms = len(arms)

    samples_dict = _create_samples_dict(num_arms)
    count_matrix = np.zeros([num_arms, num_arms], dtype=np.float32)

    for i in range(total_samples):

        src = df.iloc[i, :]["srclang"]
        tgt = df.iloc[i, :]["trglang"]

        idx = df.iloc[i, :]["srcIndex"]
        lang_pair = "{}-{}".format(src, tgt)

        referenceText = system_outputs["reference"][idx - 1]
        system1Id = df.iloc[i, :]["system1Id"].replace(".txt", "")
        system2Id = df.iloc[i, :]["system2Id"].replace(".txt", "")
        system1Idx = arm_to_idx[system1Id]
        system2Idx = arm_to_idx[system2Id]
        system1rank = df.iloc[i, :]["system1rank"]
        system2rank = df.iloc[i, :]["system2rank"]

        system1Text = system_outputs[system1Id][idx - 1]
        system2Text = system_outputs[system2Id][idx - 1]

        unique_id1 = "wmt_{}_{}_{}_1".format(year, lang_pair, i)
        unique_id2 = "wmt_{}_{}_{}_2".format(year, lang_pair, i)

        if system1rank < system2rank:
            label = 1.0
        elif system2rank < system1rank:
            label = 0.0
        else:
            label = 0.5

        data["unique_id"].append(unique_id1)
        data["langPair"].append(lang_pair)
        data["srcIndex"].append(idx)
        data["system1Id"].append(system1Id)
        data["system2Id"].append(system2Id)
        data["system1Text"].append(system1Text)
        data["system2Text"].append(system2Text)
        data["referenceText"].append(referenceText)
        data["label"].append(label)

        sample1 = {
            "unique_id": unique_id1,
            "srcIndex": idx,
            "system1Text": system1Text,
            "system2Text": system2Text,
            "referenceText": referenceText,
            "label": label,
        }
        samples_dict[(system1Idx, system2Idx)].append(sample1)
        count_matrix[system1Idx, system2Idx] += label

        data["unique_id"].append(unique_id2)
        data["langPair"].append(lang_pair)
        data["srcIndex"].append(idx)
        data["system1Id"].append(system2Id)
        data["system2Id"].append(system1Id)
        data["system1Text"].append(system2Text)
        data["system2Text"].append(system1Text)
        data["referenceText"].append(referenceText)
        data["label"].append(1 - label)

        sample2 = {
            "unique_id": unique_id2,
            "srcIndex": idx,
            "system1Text": system2Text,
            "system2Text": system1Text,
            "referenceText": referenceText,
            "label": 1 - label,
        }
        samples_dict[(system2Idx, system1Idx)].append(sample2)
        count_matrix[system2Idx, system1Idx] += 1 - label

    data = pd.DataFrame(data)

    total_counts = count_matrix + count_matrix.T
    total_counts[arms, arms] = 1
    preference_matrix = count_matrix / total_counts
    np.fill_diagonal(preference_matrix, 0.5)

    processed_data = {
        "arms": arms,
        "samples": samples_dict,
        "preference_matrix": preference_matrix,
        "idx_to_arm": idx_to_arm,
        "count_matrix": count_matrix,
        "total_samples": total_samples,
    }

    save_filename = os.path.split(human_judgements_file)[1].replace(
        ".csv", "_processed.pkl"
    )
    save_filename = os.path.join(processed_dir, save_filename)
    with open(save_filename, "wb") as fp:
        pickle.dump(processed_data, fp)

    print("Saved Preprocessed dump at {}".format(save_filename))


def prepare_data(
    reference_file_list,
    system_dir_list,
    human_judgements_file_list,
    output_file,
    processed_dir,
    year,
):

    data = {
        "unique_id": [],
        "langPair": [],
        "srcIndex": [],
        "system1Id": [],
        "system2Id": [],
        "system1Text": [],
        "system2Text": [],
        "referenceText": [],
        "label": [],
    }

    for i in range(len(reference_file_list)):

        reference_file = reference_file_list[i]
        system_dir = system_dir_list[i]
        human_judgements_file = human_judgements_file_list[i]

        _prepare_data(
            data, reference_file, system_dir, human_judgements_file, processed_dir, year
        )

    data = pd.DataFrame(data)
    data.to_csv(output_file)
    print("Saved all human judgments at {}".format(output_file))


def wmt15(raw_dir, processed_dir):

    ####  English Target Dataset #####################

    reference_file_list = [
        "wmt15-submitted-data/txt/references/newstest2015-fien-ref.en",
        "wmt15-submitted-data/txt/references/newstest2015-deen-ref.en",
        "wmt15-submitted-data/txt/references/newstest2015-ruen-ref.en",
    ]

    system_dir_list = [
        "wmt15-submitted-data/txt/system-outputs/newstest2015/fi-en",
        "wmt15-submitted-data/txt/system-outputs/newstest2015/de-en",
        "wmt15-submitted-data/txt/system-outputs/newstest2015/ru-en",
    ]

    human_judgements_file_list = [
        "wmt15-master/data/wmt15.fin-eng.csv",
        "wmt15-master/data/wmt15.deu-eng.csv",
        "wmt15-master/data/wmt15.rus-eng.csv",
    ]

    file_lists = [reference_file_list, system_dir_list, human_judgements_file_list]
    for file_list in file_lists:
        for i in range(len(file_list)):
            file_list[i] = os.path.join(raw_dir, file_list[i])

    output_file = os.path.join(processed_dir, "wmt15-human-judgements.csv")

    prepare_data(
        reference_file_list,
        system_dir_list,
        human_judgements_file_list,
        output_file,
        processed_dir,
        year=2015,
    )


def wmt16(raw_dir, processed_dir):

    reference_file_list = [
        "wmt16-submitted-data/txt/references/newstest2016-tren-ref.en",
        # "wmt16-submitted-data/txt/references/newstest2016-roen-ref.en",
        # "wmt16-submitted-data/txt/references/newstest2016-csen-ref.en",
        # "wmt16-submitted-data/txt/references/newstest2016-deen-ref.en",
    ]

    system_dir_list = [
        "wmt16-submitted-data/txt/system-outputs/newstest2016/tr-en",
        # "wmt16-submitted-data/txt/system-outputs/newstest2016/ro-en",
        # "wmt16-submitted-data/txt/system-outputs/newstest2016/cs-en",
        # "wmt16-submitted-data/txt/system-outputs/newstest2016/de-en",
    ]

    human_judgements_file_list = [
        "wmt16-master/data/wmt16.tur-eng.csv",
        # "wmt16-master/data/wmt16.ron-eng.csv",
        # "wmt16-master/data/wmt16.cze-eng.csv",
        # "wmt16-master/data/wmt16.deu-eng.csv",
    ]

    file_lists = [reference_file_list, system_dir_list, human_judgements_file_list]
    for file_list in file_lists:
        for i in range(len(file_list)):
            file_list[i] = os.path.join(raw_dir, file_list[i])

    output_file = os.path.join(processed_dir, "wmt16-human-judgements.csv")

    prepare_data(
        reference_file_list,
        system_dir_list,
        human_judgements_file_list,
        output_file,
        processed_dir,
        year=2016,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WMT 2015 or WMT 2016 data")

    parser.add_argument(
        "--raw-dir",
        type=str,
        help="Path to the directory containing the downloaded (raw) datasets",
    )

    parser.add_argument(
        "--processed-dir",
        type=str,
        help="Path to the directory to dump the processed files",
    )

    parser.add_argument(
        "--year",
        type=int,
        choices=[2015, 2016],
        help="whether to preprocess wmt 2015 or 2016",
    )

    args = parser.parse_args()

    if args.year == 2015:
        wmt15(args.raw_dir, args.processed_dir)
    else:
        wmt16(args.raw_dir, args.processed_dir)
