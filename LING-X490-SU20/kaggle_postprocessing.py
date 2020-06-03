# LING-X 490
# This standalone file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from kaggle_preprocessing import boost_multithreaded
from multiprocessing import Process, Queue
import pandas as pd



# calculate % examples in given data that contains abusive words. returns list
def percent_abusive(data, lex):
    """
    data (df): dataframe to filter
    lex (str): lexicon to filter with. Either "we" (wiegand extended) or "rds" (our manually tagged dataset)
    """
    jobs = []
    lexicon_names = ["Manual", "Wiegand (Base)", "Wiegand (Expanded)"]
    percentages = []
    q = Queue()

    filename = "../data/kaggle_data/lexicon_manual/lexicon.manual.all.csv"
    lexicon_rds = pd.read_csv(filename, sep=",", header=0)
    lexicon_rds = lexicon_rds[lexicon_rds["class"] == 1]  # only use abusive words (class=1)
    boost_list = list(lexicon_rds["word"])
    p1 = Process(target=boost_multithreaded, args=(data, filename, boost_list, q,))
    jobs.append(p1)
    p1.start()

    filename = "../Data/kaggle_data/lexicon_wiegand/lexicon.wiegand.base.abusive.csv"
    lexicon_wiegand_base = pd.read_csv(filename, sep=",", header=0)
    boost_list = list(lexicon_wiegand_base["word"])
    p2 = Process(target=boost_multithreaded, args=(data, filename, boost_list, q,))
    jobs.append(p2)
    p2.start()

    filename = "../Data/kaggle_data/lexicon_wiegand/lexicon.wiegand.expanded.abusive.csv"
    lexicon_wiegand_exp = pd.read_csv(filename, sep=",", header=0)
    boost_list = list(lexicon_wiegand_exp["word"])
    p3 = Process(target=boost_multithreaded, args=(data, filename, boost_list, q,))
    jobs.append(p3)
    p3.start()

    # multithreaded boosting; waits for all jobs to finish
    for process in jobs:
        process.join()

    # get output
    for x in jobs:
        boosted_df = q.get()
        percentages.append(round(len(boosted_df) / len(data) * 100, 2))

    to_return = list(zip(lexicon_names, percentages))
    print(to_return)  # DEBUG

    return to_return


if __name__ == '__main__':
    # build_manual_lexicon()
    pass
