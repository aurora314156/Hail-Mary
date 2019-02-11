import os, time
from keras.optimizers import *
from keras.callbacks import *
from Transformer import Transformer
from Initial import Initial

def main():
    dataset, d_model = Initial().InitialMain()
    tTime = time.time()
    for single_data in dataset:
        model = Transformer(single_data['story'], d_model, len_limit=70, d_inner_hid=512, \
                                  n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1).TransformerMain()
        model.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
        model.model.summary()

        print("Total cost time %.2fs." % (time.time()-tTime))

if __name__ == "__main__":
    main()