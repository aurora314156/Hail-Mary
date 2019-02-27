import os, time, json
from numpy import array
from bert_serving.client import BertClient
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
