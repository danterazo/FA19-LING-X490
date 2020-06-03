# LING-X 490
# This file builds, imports, and exports data
# Dante Razo, drazo
from kaggle_preprocessing import read_data, boost_data, sample_data
import pandas as pd
import os


# data headers: [y, X]

# only import once
def get_train():
    dataset = "train.target+comments.tsv"  # 'test' for classification problem
    return read_data(dataset)


# Gets 'n' posts, randomly selected, from the dataset. Then save to `.csv`
def build_random(data, sample_size, repeats=3):
    to_export = []
    # sample + export
    for i in range(0, repeats):
        to_export.append(sample_data(data, sample_size))

    export_data("random", to_export)


def build_boosted(data, manual_boost, sample_size, repeats=3):
    data_file = "train.target+comments.tsv"  # name for verbose prints
    to_export = []

    # sample + export, topic
    boosted_topic_data = boost_data(data, data_file, manual_boost)
    for i in range(0, repeats):
        to_export.append(sample_data(boosted_topic_data, sample_size))

    export_data("topic", to_export)

    # boost + sample + export, wordbank
    boosted_wordbank_data = boost_data(data, data_file)

    for i in range(0, repeats):
        to_export.append(sample_data(boosted_wordbank_data, sample_size))

    export_data("wordbank", to_export)


# save data to `.tsv`, `.csv`, etc.
def export_data(source, data, extension=".csv"):
    i = 1

    for d in data:
        filepath = os.path.join("../data/kaggle_data", f"train.{source}{i}{extension}")
        d.to_csv(filepath, index=False, header=False)
        i += 1


# generalized version of the above. `.csv`
def export_df(data, sample="no_sample", i="", path="", prefix="", index=True):
    filepath = os.path.join(path, f"{prefix}.{sample}{i}.csv")
    data.to_csv(filepath, index=index, header=True)


# builds one or both
def build_main(choice, topic, repeats, sample_size, verbose):
    """
    choice: choose which sample types to build. "random", "boosted", or "all"
    topic: topic for manual boosting
    """
    train = get_train()

    build_random(train, sample_size, repeats) if choice is "random" or "all" else None
    build_boosted(train, topic, sample_size, repeats) if choice is "boosted" or "all" else None
    print(f"Datasets built.") if verbose else None


# import Wiegand's lexicons, format them, and export them
# in `kaggle_build.py` because it isn't dynamic, i.e. the output is the same after every run
def build_wiegand_lexicons():
    cwd = os.getcwd()

    data_dir = "../repos/lexicon-of-abusive-words/lexicons"  # common directory for all repos. assumes local sys

    # read lexicons
    base = pd.read_csv(f"{data_dir}/baseLexicon.txt", sep='\t', header=None, names=["word", "class"])  # import base
    exp = pd.read_csv(f"{data_dir}/expandedLexicon.txt", sep='\t', header=None, names=["word", "score"])  # expanded

    # split word and part-of-speech
    base_split = [w.split("_") for w in base["word"]]
    exp_split = [w.split("_") for w in exp["word"]]

    # new dfs
    base["part"] = [s[1] for s in base_split]
    base["word"] = [s[0] for s in base_split]
    exp["part"] = [s[1] for s in exp_split]
    exp["word"] = [s[0] for s in exp_split]

    # base-specific
    base_abusive = base[base["class"]]["word"]

    # expanded-specific. any word with a score >0 is considered abusive in their expanded lexicon
    #   ref: Introducing a Lexicon of Abusive Words (Wiegand, Ruppenhofer, Schmidt, Greenberg)
    exp_abusive = exp[exp["score"] > 0]["word"]

    # export
    os.chdir("../data/kaggle_data/lexicon")
    base.to_csv("lexicon.wiegand.base.csv", header=0, index=False)
    exp.to_csv("lexicon.wiegand.expanded.csv", header=0, index=False)
    base_abusive.to_csv("lexicon.wiegand.base.abusive.csv", header=0, index=False)
    exp_abusive.to_csv("lexicon.wiegand.expanded.abusive.csv", header=0, index=False)
    os.chdir(cwd)  # go back to previous cwd


""" MAIN """
# configuration
topic = ["trump"]  # [str]
to_build = "all"  # "all", "random", or "boosted"

# manually run:
# build_main(to_build, topic)
# build_wiegand_lexicons()
