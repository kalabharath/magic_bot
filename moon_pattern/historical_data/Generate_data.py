#!/usr/bin/python3
import logging
import os
from datetime import date
import yfinance as yf
import pandas as pd
from finvizfinance.screener.overview import Overview
# use looger to log the data
from logging import getLogger, DEBUG, INFO, WARNING, ERROR, CRITICAL

logger = logging.getLogger('historical_data')
logger.setLevel(INFO)
logger.info('Start')

# logger to file
logging.basicConfig(filename='historical_data.log', level=INFO, format='%(asctime)s %(message)s')



def get_tickers():

    filter = {'Average Volume': 'Over 2M', 'InsiderOwnership': 'Any', 'P/E': 'Any', 'Forward P/E':'Profitable (>0)',
                      'InstitutionalOwnership': 'Over 20%', 'Price': 'Over $90'}

    foverview = Overview()
    foverview.set_filter(filters_dict=filter)
    df = foverview.screener_view()
    # drop if the value matches a certain value
    df = df.drop(df[df['Sector'] == 'Financial'].index)
    df = df.drop(df[df['Sector'] == 'Energy'].index)
    df = df.drop(df[df['Industry'] == 'Oil & Gas Integrated'].index)
    df = df.drop(df[df['Industry'] == 'Oil & Gas Refining & Marketing'].index)
    df = df.drop(df[df['Industry'] == 'Restaurants'].index)
    df = df.drop(df[df['Industry'] == 'Restaurants'].index)


    return df


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
        logger.info("Saving data to file")
        out_file = './' + self.ticker + date_ext + '.pkl'
        # check if the file exists
        if os.path.isfile(out_file):
            logger.error("File exists")
            return False
        else:
            t = yf.Ticker(self.ticker)
            hist_data = t.history(period="2y", interval="1d")
            # log dataframe header
            logger.info(hist_data.head())
            dates = hist_data['Open'].to_list()
            hist_data.to_pickle(out_file)
            import time
            time.sleep(1)
            return True


if __name__ == '__main__':

    df = get_tickers()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    logger.info(df)
    logger.info("# Number of tickers: %d", len(df))
    # log dataframe to file
    logger.info("# Logging dataframe to file")
    tickers = df['Ticker'].to_list()
    logger.info("# Number of tickers: %d", len(tickers))
    logger.info("# Tickers: %s", tickers)

    logger.info("# Fetching data from yfinance")
    logger.info("# Number of tickers: %d", len(tickers))
    logger.info("# Tickers: %s", tickers)
    tickers = tickers+ ['PFE', 'MRNA', 'BNTX']
    for ticker in tickers:
        data = UsaData(ticker=ticker)
        data.dump_data()

