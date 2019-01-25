
from PosEncodAndEmbed import PosEncodAndEmbed

class Transformer():
    def __init__(self, words, word2vec, max_dim_len):
        self.words = words
        self.word2vec = word2vec
        self.max_dim_len = max_dim_len
        self.data_matrix = ""
    def TransformerMain(self):
        # get result by embedding with positional encoding
        self.data_matrix = PosEncodAndEmbed(self.words, self.word2vec, self.max_dim_len).PosEncodAndEmbedMain()