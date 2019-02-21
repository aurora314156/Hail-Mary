import os, time
from Initial import Initial
from tensor2tensor.models import transformer
import tensorflow as tf

def main():
    dataset, d_model = Initial().InitialMain()
    tTime = time.time()
    for single_data in dataset:
        hparams = transformer.transformer_base()
        encoder = transformer.TransformerEncoder(hparams, mode=tf.estimator.ModeKeys.PREDICT)
        #x = <shape [batch_size, timesteps, 1, hparams.hidden_dim]>
        #enc = encoder({"inputs": x, "targets": x})

        print("Total cost time %.2fs." % (time.time()-tTime))

if __name__ == "__main__":
    main()