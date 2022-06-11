#!/usr/bin/python3
import glob
import os
from datetime import date

import pandas as pd
import yfinance as yf
from tqdm import tqdm


class UsaData():
    def __init__(self, ticker):
        self.ticker = ticker

    def dump_data(self):
        """
        :return:
        """
        today = date.today()
        date_ext = today.strftime("_%d_%b_%Y")
        # Save data so as to avoid online data fetch
        print("# Saving historical data to disk")
        out_file = './' + self.ticker + date_ext + '.pkl'
        # check if the file exists
        if os.path.isfile(out_file):
            return False
        else:
            t = yf.Ticker(self.ticker)
            hist_data = t.history(period="2y", interval="1d")
            print (hist_data.head())
            dates = hist_data['Open'].to_list()
            hist_data.to_pickle(out_file)
        return True


if __name__ == '__main__':

    data = UsaData(ticker='PFE')
    data.dump_data()

