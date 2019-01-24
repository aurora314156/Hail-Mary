#!/usr/bin/env python3
import logging, time
from gensim.models import KeyedVectors

class PosEncodAndEmbed():
    def __init__(self, words, max_dim_len):
        self.words = words
        self.max_dim_len = max_dim_len
    def PosEncodAndEmbedMain(self):
        data_matrix = []
        data_matrix = self.Embedding(self.words)
        data_matrix = self.PosEncoding(self.words)
        data_matrix = self.Add()
        return data_matrix
        
    def Embedding(self, words):
        print("Starting Embedding words.")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # get vector with weight of words from w2v model
        model_weight_vector = model.wv
        embeddedWords = []
        for word in words:
            if word not in model_weight_vector:
                embeddedWords = 

        return embeddedWords
    
    def PosEncoding(self, words):
        print("Starting Encoding words.")
        return 123
    def Add(self):
        print("Starting Add Embedding words and Encoding words.")
        return 123
    