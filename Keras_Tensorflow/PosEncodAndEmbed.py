#!/usr/bin/env python3
import numpy as np
from keras.layers import Embedding
class PosEncodAndEmbed():
    def __init__(self, words_len, len_limit, d_model):
        self.words_len = words_len
        self.len_limit = len_limit
        self.d_model = d_model
        
    def PosEncodAndEmbedMain(self):
        pos_emb = self.GetPosEncodingMatrix(self.len_limit, self.d_model)
        word_emb = self.GetWordEmbedMatrix(self.words_len, self.d_model)

        return pos_emb, word_emb

    def GetPosEncodingMatrix(self, len_limit, d_model):
        print("Starting process PosEncoding.")
        pos_enc = np.array([
                [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] 
                if pos != 0 else np.zeros(d_model) 
                    for pos in range(len_limit)
                    ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
        
        pos_emb = Embedding(len_limit, d_model, trainable=False, weights=[pos_enc])

        return pos_emb

    def GetWordEmbedMatrix(self, words_len, d_model):
        print("Starting process WordEmbedding.")
        word_emb = Embedding(words_len, d_model)

        return word_emb
    """
    def Embedding(self, words, word2vec):
        print("Starting Embedding words.")
        embeddedWords = []
        # use pre-trained model to transforming each word
        for word in words:
            if word in word2vec:
                word_with_weight = word2vec[word]
                embeddedWords.append(word_with_weight)
        print(len(embeddedWords))
        npArr_embeddedWords = np.array(embeddedWords)
        #print(npArr_embeddedWords.shape)
        return npArr_embeddedWords
    """
