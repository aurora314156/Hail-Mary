from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from PosEncodAndEmbed import PosEncodAndEmbed
from Encoder import Encoder
class Transformer():
    def __init__(self, words, d_model, len_limit, d_inner_hid, n_head, d_k, d_v, layers, dropout):
        self.words = words
        self.words_len = len(words)
        self.d_model = d_model
        self.len_limit = len_limit
        self.d_inner_hid = d_inner_hid
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.layers = layers
        self.dropout = dropout

    def TransformerMain(self):
        # get result by embedding with positional encoding
        pos_emb, word_emb = PosEncodAndEmbed(self.words_len, self.len_limit, self.d_model).PosEncodAndEmbedMain()
        encoder_result = Encoder(self.d_model, self.d_inner_hid, self.n_head, self.d_k, self.d_v, self.layers,\
                          self.dropout, word_emb=word_emb, pos_emb=pos_emb)

        return encoder_result

    def compile(self, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(None,), dtype='int32')
        tgt_seq_input = Input(shape=(None,), dtype='int32')

        src_seq = src_seq_input
        tgt_seq  = Lambda(lambda x:x[:,:-1])(tgt_seq_input)
        tgt_true = Lambda(lambda x:x[:,1:])(tgt_seq_input)

        src_pos = Lambda(self.get_pos_seq)(src_seq)
        tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
        if not self.src_loc_info: src_pos = None

        enc_output = self.encoder(src_seq, src_pos, active_layers=active_layers)
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)
        final_output = self.target_layer(dec_output)

        def get_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_accu(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        loss = Lambda(get_loss)([final_output, tgt_true])
        self.ppl = Lambda(K.exp)(loss)
        self.accu = Lambda(get_accu)([final_output, tgt_true])

        self.model = Model([src_seq_input, tgt_seq_input], loss)
        self.model.add_loss([loss])
        self.output_model = Model([src_seq_input, tgt_seq_input], final_output)

        self.model.compile(optimizer, None)
        self.model.metrics_names.append('ppl')
        self.model.metrics_tensors.append(self.ppl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(self.accu)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask