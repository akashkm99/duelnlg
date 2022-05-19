import numpy as np
import pandas as pd
import os
import codecs
import argparse

langauge_dict = {
    "Czech": "cs",
    "English": "en",
    "Spanish": "es",
    "Russian": "ru",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
}


def prepare_submissions_reference(reference_file_list, system_dir_list, lang_pair_list):

    assert len(reference_file_list) == len(system_dir_list) and len(
        reference_file_list
    ) == len(lang_pair_list)

    full_data = {}
    for i in range(len(reference_file_list)):

        reference_file = reference_file_list[i]
        system_dir = system_dir_list[i]
        lang_pair = lang_pair_list[i]

        data = {"reference": []}
        with codecs.open(reference_file, "r", encoding="utf-8") as fp:

            for line in fp.readlines():
                data["reference"].append(line.strip())

        systems = os.listdir(system_dir)
        for system in systems:
            with codecs.open(
                os.path.join(system_dir, system), "r", encoding="utf-8"
            ) as fp:
                try:
                    data[system] = []
                    for line in fp.readlines():
                        data[system].append(line.strip())
                except:
                    del data[system]
                    print(
                        "Unable to read from file {}".format(
                            os.path.join(system_dir, system)
                        )
                    )
                    print("=" * 30)

        full_data[lang_pair] = data

    return full_data


def prepare_data(
    reference_file_list,
    system_dir_list,
    lang_pair_list,
    human_judgements_file,
    output_file,
):

    system_outputs = prepare_submissions_reference(
        reference_file_list, system_dir_list, lang_pair_list
    )
    data = {
        "langPair": [],
        "system1Id": [],
        "system2Id": [],
        "system1Text": [],
        "system2Text": [],
        "referenceText": [],
        "label": [],
    }
    df = pd.read_csv(human_judgements_file)
    n = len(df)

    for i in range(n):
        src = langauge_dict[df.iloc[i, :]["srclang"]]
        tgt = langauge_dict[df.iloc[i, :]["trglang"]]

        if tgt != "en":
            continue

        idx = df.iloc[i, :]["srcIndex"]
        lang_pair = "{}-{}".format(src, tgt)
        referenceText = system_outputs[lang_pair]["reference"][idx - 1]

        for j in range(1, 6):
            for k in range(j + 1, 6):
                system1Id = df.iloc[i, :]["system{}Id".format(j)]
                system2Id = df.iloc[i, :]["system{}Id".format(k)]

                if system1Id not in system_outputs[lang_pair]:
                    # print ("Removing {}".format(system1Id))
                    continue

                if system2Id not in system_outputs[lang_pair]:
                    # print ("Removing {}".format(system2Id))
                    continue

                system1rank = df.iloc[i, :]["system{}rank".format(j)]
                system2rank = df.iloc[i, :]["system{}rank".format(k)]

                system1Text = system_outputs[lang_pair][system1Id][idx - 1]
                system2Text = system_outputs[lang_pair][system2Id][idx - 1]

                if system1rank < system2rank:
                    label = 1.0
                elif system2rank < system1rank:
                    label = 0.0
                else:
                    label = 0.5

                data["langPair"].append(lang_pair)
                data["system1Id"].append(system1Id)
                data["system2Id"].append(system2Id)
                data["system1Text"].append(system1Text)
                data["system2Text"].append(system2Text)
                data["referenceText"].append(referenceText)
                data["label"].append(label)

    data = pd.DataFrame(data)
    data.to_csv(output_file)
    print("Saved data to {}".format(output_file))


def wmt13(raw_dir, processed_dir):
    reference_file_list = [
        "wmt13-data/plain/references/newstest2013-ref.en",
        "wmt13-data/plain/references/newstest2013-ref.en",
        "wmt13-data/plain/references/newstest2013-ref.en",
        "wmt13-data/plain/references/newstest2013-ref.en",
        "wmt13-data/plain/references/newstest2013-ref.en",
    ]

    system_dir_list = [
        "wmt13-data/plain/system-outputs/newstest2013/cs-en",
        "wmt13-data/plain/system-outputs/newstest2013/de-en",
        "wmt13-data/plain/system-outputs/newstest2013/fr-en",
        "wmt13-data/plain/system-outputs/newstest2013/ru-en",
        "wmt13-data/plain/system-outputs/newstest2013/es-en",
    ]

    lang_pair_list = ["cs-en", "de-en", "fr-en", "ru-en", "es-en"]
    human_judgements_file = "wmt13-manual-evaluation/wmt13-judgments.csv"

    file_lists = [reference_file_list, system_dir_list]
    for file_list in file_lists:
        for i in range(len(file_list)):
            file_list[i] = os.path.join(raw_dir, file_list[i])

    human_judgements_file = os.path.join(raw_dir, human_judgements_file)
    output_file = os.path.join(processed_dir, "wmt13-human-judgements.csv")
    prepare_data(
        file_lists[0], file_lists[1], lang_pair_list, human_judgements_file, output_file
    )


def wmt14(raw_dir, processed_dir):
    reference_file_list = [
        "wmt14-data/txt/references/newstest2014-csen-ref.en",
        "wmt14-data/txt/references/newstest2014-deen-ref.en",
        "wmt14-data/txt/references/newstest2014-fren-ref.en",
        "wmt14-data/txt/references/newstest2014-ruen-ref.en",
        "wmt14-data/txt/references/newstest2014-hien-ref.en",
    ]

    system_dir_list = [
        "wmt14-data/txt/system-outputs/newstest2014/cs-en",
        "wmt14-data/txt/system-outputs/newstest2014/de-en",
        "wmt14-data/txt/system-outputs/newstest2014/fr-en",
        "wmt14-data/txt/system-outputs/newstest2014/ru-en",
        "wmt14-data/txt/system-outputs/newstest2014/hi-en",
    ]

    lang_pair_list = ["cs-en", "de-en", "fr-en", "ru-en", "hi-en"]
    human_judgements_file = "wmt14-manual-evaluation/wmt14-judgments-anonymized.csv"

    file_lists = [reference_file_list, system_dir_list]
    for file_list in file_lists:
        for i in range(len(file_list)):
            file_list[i] = os.path.join(raw_dir, file_list[i])

    human_judgements_file = os.path.join(raw_dir, human_judgements_file)
    output_file = os.path.join(processed_dir, "wmt14-human-judgements.csv")
    prepare_data(
        file_lists[0], file_lists[1], lang_pair_list, human_judgements_file, output_file
    )


def merge_and_split(filename1, filename2, output_dir):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df = pd.concat([df1, df2])

    train, validate, test = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    train.to_csv(os.path.join(output_dir, "train.csv"))
    test.to_csv(os.path.join(output_dir, "test.csv"))
    validate.to_csv(os.path.join(output_dir, "val.csv"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess WMT 2014 and WMT 2015 data"
    )
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
        choices=[2013, 2014],
        help="whether to preprocess wmt 2015 or 2016",
    )
    args = parser.parse_args()
    if args.year == 2014:
        wmt14(args.raw_dir, args.processed_dir)
    else:
        wmt13(args.raw_dir, args.processed_dir)
