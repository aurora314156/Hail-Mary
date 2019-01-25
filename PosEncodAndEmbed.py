#!/usr/bin/env python3

class PosEncodAndEmbed():
    def __init__(self, words, word2vec, max_dim_len):
        self.words = words
        self.word2vec = word2vec
        self.max_dim_len = max_dim_len
        
    def PosEncodAndEmbedMain(self):
        data_matrix = []
        data_matrix = self.Embedding(self.words, self.word2vec)
        data_matrix = self.PosEncoding(self.words)
        data_matrix = self.Add()
        return data_matrix
        
    def Embedding(self, words, word2vec):
        print("Starting Embedding words.")
        embeddedWords = []
        # use pre-trained model to transforming each word 
        for word in words:
            if word in word2vec:
                word_with_weight = word2vec[word]
                embeddedWords.append(word_with_weight)
                print(len(embeddedWords))

        return embeddedWords
    
    def PosEncoding(self, words):
        print("Starting Encoding words.")
        return 123
    def Add(self):
        print("Starting Add Embedding words and Encoding words.")
        return 123
    