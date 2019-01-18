import os
import argparse
from loadData import loadData


def main():
    # args setting
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--optional-dataType", help="optional dataType", dest="dataType", default="with")
    parser.add_argument("-o", "--optional-dataSet", help="optional dataSet", dest="dataSet", default="train")
    args = parser.parse_args()
    data = loadData(args.dataType, args.dataSet).getDataSet()
    
if __name__ == "__main__":
    main()