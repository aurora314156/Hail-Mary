import os, time
from Initial import Initial
from tensor2tensor.models import transformer
import tensorflow as tf

def main():
    INPUT_TEXT_TO_TRANSLATE = 'Translate this sentence into French'
    dataset, d_model = Initial().InitialMain()
    tTime = time.time()
    for single_data in dataset:
        hparams = transformer.transformer_base()
        encoder = transformer.TransformerEncoder(hparams, mode=tf.estimator.ModeKeys.PREDICT)
        #x = <your inputs, which should be of shape [batch_size, timesteps, 1, hparams.hidden_dim]>
        #input_x = [30, timesteps, 1, hparams.hidden_dim]
        #enc = encoder({"inputs": input_x, "targets": 0, "target_space_id": 0})
        encoder.infer(INPUT_TEXT_TO_TRANSLATE, )

        print("Total cost time %.2fs." % (time.time()-tTime))

if __name__ == "__main__":
    main()