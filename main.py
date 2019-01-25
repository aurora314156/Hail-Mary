import os
from Transformer import Transformer
from Initial import Initial



def main():
    dataset, word2vec, max_dim_len = Initial().InitialMain()
    for single_data in dataset:
        StoryMatirx = Transformer(single_data['story'], word2vec, max_dim_len).TransformerMain()
        


if __name__ == "__main__":
    main()