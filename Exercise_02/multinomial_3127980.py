#! /usr/bin/python
# -*- coding: utf-8 -*-


"""MLE for the multinomial distribution."""


import math
import numpy as np
from argparse import ArgumentParser
from collections import Counter, OrderedDict


def get_words(file_path):
    """Return a list of words from a file, converted to lower case."""
    with open(file_path, encoding='utf-8') as hfile:
        return hfile.read().lower().split()


def get_probabilities(words, stopwords, k):
    # filter stopwords
    filtered_words = [word for word in words if word not in stopwords]

    # count all words
    counter = Counter(filtered_words)

    # initialize dict
    return_dict = {}

    # pre-calculate inverse of total count of k-most-common words (for multiplication)
    inverse_size = 1.0/sum([count for _,count in counter.most_common(k)])

    # put all findings in a dict
    for word, frequency in counter.most_common(k):
        return_dict[word] = frequency*inverse_size

    return return_dict


def multinomial_pmf(observation, probabilities):
    """
    The multinomial probability mass function.
    Inputs:
        * observation: dictionary, maps words (X_i) to observed frequencies (x_i)
        * probabilities: dictionary, maps words to their probabilities (p_i)

    Return the probability for the observation, i.e. P(X_1=x_1, ..., X_k=x_k).
    """

    print("observation" + str(observation))

    # the product of all observed frequencys as factorials (x1!*x2*!*x3!...*xn!)
    factorial_product = 1.0;

    # the product of all probabilities risen to the power of the actual observation
    # (p1^x1 * p2^x2 * ... * pn^xn)
    probability_product = 1.0;

    # count the total number of observations
    n=0.0;

    observed_frequency = 0
    for word, probability in probabilities.items():
        observed_frequency = observation.get(word, 0); # in case, the key does not exist, return 0
        n = n + observed_frequency
        print(str(n))
        factorial_product = factorial_product*math.factorial(observed_frequency)
        probability_product = probability_product * probability**observed_frequency

    return math.factorial(n)/factorial_product * probability_product


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUT_FILE', help='A file containing whitespace-delimited words')
    arg_parser.add_argument('SW_FILE', help='A file containing whitespace-delimited stopwords')
    arg_parser.add_argument('-k', type=int, default=10,
                            help='How many of the most frequent words to consider')
    args = arg_parser.parse_args()

    words = get_words(args.INPUT_FILE)
    stopwords = set(get_words(args.SW_FILE))
    probabilities = get_probabilities(words, stopwords, args.k)

    # we should have k probabilities
    assert len(probabilities) == args.k

    # check if all p_i sum to 1 (accounting for some rounding error)
    assert 1 - 1e-12 <= sum(probabilities.values()) <= 1 + 1e-12

    # check if p_i >= 0
    assert not any(p < 0 for p in probabilities.values())

    # print estimated probabilities
    print('estimated probabilities:')
    i = 1
    for word, prob in probabilities.items():
        print('p_{}\t{}\t{:.5f}'.format(i, word, prob))
        i += 1

    # read inputs for x_i
    print('\nenter observation:')
    observation = {}
    i = 1
    for word in probabilities:
        observation[word] = int(input('X_{}='.format(i)))
        i += 1

    # print P(X_1=x_1, ..., X_k=x_k)
    print('\nresult: {}'.format(multinomial_pmf(observation, probabilities)))


if __name__ == '__main__':
    main()
