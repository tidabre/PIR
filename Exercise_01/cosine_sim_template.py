#! /usr/bin/python
# -*- coding: utf-8 -*-


"""Rank sentences based on cosine similarity and a query."""


from collections import defaultdict
from argparse import ArgumentParser
from operator import itemgetter
import numpy as np


def get_sentences(file_path):
    """Return a list of sentences from a file."""
    with open(file_path, encoding='utf-8') as hfile:
        return hfile.read().splitlines()


def get_top_k_words(sentences, k):
    
    counter = defaultdict(int)
    
    """Return the k most frequent words as a list."""
    for sentence in sentences:
        for word in sentence.split():
            counter[word] += 1
                   
    value_key_pairs = [(val,key) for key,val in counter.items()];
    value_key_pairs.sort(reverse=True)
                                           
    """Return top k items, take only the key"""
    return [key for val,key in value_key_pairs[:k]]


def encode(sentence, vocabulary):
    """Return a vector encoding the sentence."""
    vector = [0 for i in range(len(vocabulary))]
    
    for word in sentence.split():
        for i in range(len(vocabulary)):
            if word==vocabulary[i] :
                vector[i]+=1
                continue
    
    return np.asarray(vector)


def get_top_l_sentences(sentences, query, vocabulary, l):
    """
    For every sentence in "sentences", calculate the similarity to the query.
    Sort the sentences by their similarities to the query.

    Return the top-l most similar sentences as a list of tuples of the form
    (similarity, sentence).
    """
    query_vector = encode(query, vocabulary)
        
    """sentences.sort(key=lambda sentence: cosine_sim(encode(sentence, vocabulary), query_vector), reverse=True)"""
    
    list_of_values = [];
    for sentence in sentences:
        list_of_values.append((cosine_sim(encode(sentence, vocabulary), query_vector), sentence))
    
    print("done, appending")
    
    list_of_values.sort(key=itemgetter(0), reverse=True)
    
    return list_of_values[:l]


def cosine_sim(u, v):
    """Return the cosine similarity of u and v."""
    
    return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUT_FILE', help='An input file containing sentences, one per line')
    arg_parser.add_argument('QUERY', help='The query sentence')
    arg_parser.add_argument('-k', type=int, default=1000,
                            help='How many of the most frequent words to consider')
    arg_parser.add_argument('-l', type=int, default=10, help='How many sentences to return')
    args = arg_parser.parse_args()

    sentences = get_sentences(args.INPUT_FILE)
    top_k_words = get_top_k_words(sentences, args.k)
    query = args.QUERY.lower()

    print('using vocabulary: {}\n'.format(top_k_words))
    print('using query: {}\n'.format(query))

    # suppress numpy's "divide by 0" warning.
    # this is fine since we consider a zero-vector to be dissimilar to other vectors
    with np.errstate(invalid='ignore'):
        result = get_top_l_sentences(sentences, query, top_k_words, args.l)

    print('result:')
    for sim, sentence in result:
        print('{:.5f}\t{}'.format(sim, sentence))


if __name__ == '__main__':
    main()
