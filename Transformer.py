
from PosEncodAndEmbed import PosEncodAndEmbed

class Transformer():
    def __init__(self, words, max_dim_len):
        self.words = words
        self.max_dim_len = max_dim_len
        self.data_matrix = ""
    def TransformerMain(self):
        # get result by embedding with positional encoding
        return
        self.data_matrix = PosEncodAndEmbed(self.words, self.max_dim_len).PosEncodAndEmbedMain()