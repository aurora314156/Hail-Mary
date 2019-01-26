
from PosEncodAndEmbed import PosEncodAndEmbed

class Transformer():
    def __init__(self, words, word2vec, d_model):
        self.words = words
        self.word2vec = word2vec
        self.d_model = d_model
        self.data_matrix = ""
    def TransformerMain(self):
        # get result by embedding with positional encoding
        self.data_matrix = PosEncodAndEmbed(self.words, self.word2vec, self.d_model).PosEncodAndEmbedMain()