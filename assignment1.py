import argparse

from utils import read_hate_tweets
from evaluation import accuracy, f_1
from model.naivebayes import NaiveBayes
from model.logreg import LogReg
from helper import train_smooth, train_feature_eng, train_logreg


TWEETS_ANNO = './data/NAACL_SRW_2016.csv'
TWEETS_TEXT = './data/NAACL_SRW_2016_tweets.json'
MODEL_DICT = {'naive-bayes': NaiveBayes, 'logreg': LogReg}


def main():
    parser = argparse.ArgumentParser(
        description='Train naive bayes or logistic regression'
    )

    parser.add_argument(
        '--model', dest='model',
        choices=['naive-bayes', 'logreg'],
        help='{naive-bayes, logreg}', type=str,
        required=True
    )

    parser.add_argument(
        '--test_smooth', dest='test_smooth',
        help='Train and test Naive Bayes with varying smoothing parameter k',
        action='store_true'
    )

    parser.add_argument(
        '--feature_eng', dest='feature_eng',
        help='Train and test Naive Bayes with different feature types',
        action='store_true'
    )

    args = parser.parse_args()

    (train_data, test_data) = read_hate_tweets(TWEETS_ANNO, TWEETS_TEXT)

    model = MODEL_DICT[args.model]

    if args.model == 'naive-bayes':
        print("Training naive bayes classifier...")
        nb = model.train(train_data)
        print("Accuracy: ", accuracy(nb, test_data))
        print("F_1: ", f_1(nb, test_data))

        if args.test_smooth:
            train_smooth(train_data, test_data)

        if args.feature_eng:
            train_feature_eng(train_data, test_data)
    else:
        print("Training logistic regression classifier...")
        train_logreg(train_data, test_data)


if __name__ == "__main__":
    main()
