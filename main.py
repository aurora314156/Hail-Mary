import os, time
from numpy import array
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from Transformer import Transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Initial import Initial





def main():
    dataset, d_model = Initial().InitialMain()
    tTime = time.time()
    for single_data in dataset:
        transformer_model = Transformer(single_data['story'], d_model, len_limit=70, d_inner_hid=512, \
                                  n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1)
        transformer_model.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
        transformer_model.model.summary()

        # create the tokenizer
        values = array(single_data['story'])
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        x_train = onehot_encoded.astype('float32') / 255.
        print(onehot_encoded)
        x_test = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        
        #x_test = [one_hot(s, vocab_size) for s in single_data['story']]
        transformer_model.model.fit(x_test,x_test,epochs=20,batch_size=64,validation_data=(x_test, x_test))

        print("Total cost time %.2fs." % (time.time()-tTime))

if __name__ == "__main__":
    main()