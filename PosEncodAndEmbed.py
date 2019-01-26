#!/usr/bin/env python3
import numpy as np
class PosEncodAndEmbed():
    def __init__(self, words, word2vec, d_model):
        self.words = words
        self.word2vec = word2vec
        self.d_model = d_model
        self.max_data_len = len(words)
        
    def PosEncodAndEmbedMain(self):
        data_matrix = []
        data_matrix = self.Embedding(self.words, self.word2vec)
        data_matrix = self.PosEncoding(self.max_data_len, self.d_model)
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
        npArr_embeddedWords = np.array(embeddedWords)
        #print(npArr_embeddedWords.shape)
        return npArr_embeddedWords
    
    def PosEncoding(self, max_data_len, d_model):
        print("Starting PosEncoding.")
        pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] 
		if pos != 0 else np.zeros(d_model) 
			for pos in range(max_data_len)
			])
        pos_enc = np.array([
		    [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		    if pos != 0 else np.zeros(d_emb) 
			    for pos in range(max_len)
			    ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
        print(max_data_len)
        #print(pos_enc.shape)
        return pos_enc

    def Add(self):
        print("Starting Add Embedding words and Encoding words.")
        return 123
    