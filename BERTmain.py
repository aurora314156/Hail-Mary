import os, time, json
from numpy import array
from keras.optimizers import *
from keras.callbacks import *
from bert_serving.client import BertClient
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Initial import Initial


def main():
    dataset, d_model = Initial().InitialMain()
    tTime = time.time()
    # initial BERT model
    bc = BertClient()
    for single_data in dataset:
        temp_str = ''
        sentence = temp_str.join(single_data['story'])
        print(sentence)
        print("=======================")
        array = bc.encode(sentence)
        print("=======================")
        print(array)
        print("=======================")
        print(array.shape)
        print("=======================")
        
    print("Total cost time %.2fs." % (time.time()-tTime))

if __name__ == "__main__":
    main()
