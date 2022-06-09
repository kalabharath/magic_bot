import sys
import os
import glob
import pandas as pd


def run_analytics(data_file):
    df = pd.read_pickle(data_file)
    print(df)
    # split df by dates

    return True


if __name__ == '__main__':

    run_analytics(data_file="magic_formula_daily.pkl")
