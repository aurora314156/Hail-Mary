import os
import argparse
from LoadData import LoadData
from Transformer import Transformer

def main():
    # args setting
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--optional-dataType", help="optional dataType", dest="dataType", default="with")
    parser.add_argument("-o", "--optional-dataSet", help="optional dataSet", dest="dataSet", default="train")
    args = parser.parse_args()
    # get dataset
    dataset = LoadData(args.dataType, args.dataSet).getDataSet()
    max_dim_len = 300
    for single_data in dataset:
        StoryMatirx = Transformer(single_data['story'], max_dim_len).TransformerMain()
        

if __name__ == "__main__":
    main()