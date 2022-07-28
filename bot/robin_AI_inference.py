from logger_setup import *
import tensorflow as tf
import yfinance as yf
from keras_preprocessing import image
from tqdm import tqdm
import glob
import os
import argparse
import cufflinks as cf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minimal_radar import ComplexRadar
from sklearn.preprocessing import minmax_scale
from stockstats import StockDataFrame
import datetime
import plotly.io as pio
from PIL import Image


matplotlib.use('TkAgg')


def numba_interweave(arr1, arr2):
    """
    interweaves one array with another, eg [1,2,3] and [A, B, C] = [1, A, 2, B, 3, C]
    :param arr1:
    :param arr2:
    :return:
    """
    res = np.empty(arr1.size + arr2.size, dtype=arr1.dtype)
    for idx, (item1, item2) in enumerate(zip(arr1, arr2)):
        res[idx * 2] = item1
        res[idx * 2 + 1] = item2
    return res


def reduce_array(arr, time_frame=1):
    """
    Dimensonality or data reduction by averaging over a specified window of three.
    :param arr:
    :return:
    """
    if len(arr) % time_frame == 0:
        t = np.mean(arr.reshape(-1, time_frame), axis=1)
    else:
        t = np.nanmean(
            np.pad(arr.astype(float), (0, time_frame - arr.size % time_frame), mode='constant',
                   constant_values=np.NaN).reshape(-1, time_frame),
            axis=1)

    return t


def generate_daily_summary(ticker='ticker', output_file_preix='output_pdf_test'):
    """
    Integrate and merge two images into one and generate a daily summary of the data.
    :param output_file_preix:
    :return: boolean
    """

    out_pdf_string = ''
    for (radar_img, candle_img) in zip(sorted(glob.glob(ticker + "*radar*.png"), key=os.path.getmtime),
                                       sorted(glob.glob(ticker + "*candle*.png"), key=os.path.getmtime)):
        radar = Image.open(radar_img)
        background = Image.open(candle_img)
        background.paste(radar, box=(1900, 10))
        background.save(candle_img)
        out_pdf_string = out_pdf_string + ' ' + candle_img
    os.system('convert ' + out_pdf_string + ' ' + output_file_preix + '.pdf')
    # os.system('open ' + output_file_preix + '.pdf')

    return True


class ManualPlotNsave():
    """
    Supervised selection of 3-month period stock with stock reaching the lowest before a 10% intra-day bump
    and continues exponential climbing.
    1. Displays the gradual drop of stock prices in a three month period where the the normalized 'RSI' and
    'ATR' or 'VWAP" reaches zero.
    2. Draw the support and resistance lines.
    3. Draw the entry and exit prices at 1:3 risk to reward profile.
    4. Save the profile if steps 1-3 above were true.
    """

    def __init__(self, check_point_file, time_series, trailing_time_period, forward_time_period, uptick_threshold,
                 ticker, rsi_threshold):
        """

        :param check_point_file:
        :param time_series:
        :param trailing_time_period:
        :param forward_time_period:
        :param uptick_threshold:
        :param ticker:
        :param rsi_threshold:
        """
        self.check_point_file = check_point_file
        self.time_series = time_series
        self.uptick_threhold = uptick_threshold
        self.ticker = ticker
        self.rsi_threshold = rsi_threshold
        self.total_days = int(trailing_time_period[0]) * 22
        self.forward_days = int(forward_time_period[0]) * 22

    def get_unusual_volume_days(self, unusual_volume_sigma=5.0):
        """

        :return:
        """
        unusual_volume_days = []
        copy_of_time_series = self.time_series.copy(deep=True)

        start = copy_of_time_series.index[-23]  # this has to be a specific index but what is it ?
        stop = copy_of_time_series.index[-3] # last 3 days
        evaluation_date = copy_of_time_series.index[-1]  # today
        day_before_evaluation_date = self.time_series.index[-2]  # yesterday

        # compute the average volume for the period
        avg_volume = copy_of_time_series.loc[start:stop, 'Volume'].mean()
        # compute standard deviation of the volume for the period
        std_volume = copy_of_time_series.loc[start:stop, 'Volume'].std()

        # slice dataframe with the evaluation date
        tdata = copy_of_time_series[start:stop]
        # convert timestamps to seconds
        tdata['timestamp'] = tdata.index.astype('int64') / 1e9
        # convert timestamps to days
        tdata['day'] = tdata['timestamp'].apply(lambda x: int(x / 86400))

        # check if the evaluation date volume is unusually higher than average volume
        last_3_days_volume = copy_of_time_series.loc[stop:day_before_evaluation_date, 'Volume'].mean()

        if last_3_days_volume > (
                avg_volume + (unusual_volume_sigma * std_volume)):

            logger.info("Volume is unusually high: %s "% self.ticker)
            logger.info("Unusual volume ratio is %s" % str(int(last_3_days_volume/avg_volume)))

            # check if the 'close' price is higher than 'open' price for the last 3 days
            trend = 0
            for i in range(1, 4):
                if tdata.iloc[-i]['Open'] < tdata.iloc[-i]['Close']:
                    trend += 1
            if trend > 1:
                logger.info("The 'Close' price is higher than 'Open' price for the last 3 days for %s" % self.ticker)

            """
            f = np.polyfit(tdata['day'], tdata['Close'], deg=1)
            degrees = np.degrees(np.tan(f[0]))
            # check if the degrees of the straight line is horizontal
            if -10.0 <= degrees <= 20.0:
                unusual_volume_days.append(evaluation_date)
                # print(evaluation_date, f, degrees)
            else:
                logger.info("X %s" % self.ticker)
                # logger.info(self.ticker, str(evaluation_date), str(f), degrees)
                return False
            """
        else:
            return False

        return True

    def cufflinks_display(self):
        """
        Display the data in a cufflinks plot.
        :return: boolean
        """
        if not self.get_unusual_volume_days(unusual_volume_sigma=2.0):
            return False

        start = self.time_series.index[-23]  # find exact index of the first day relative to today
        stop = self.time_series.index[-1]
        evaluation_date = self.time_series.index[-1]  # today

        # simple_quant_title
        quant_title = self.ticker + "_candle_" + evaluation_date.strftime('%d%b%y') + ".png"
        radar_quant_title = self.ticker + "_radar_" + evaluation_date.strftime('%d%b%y') + ".png"

        data = self.time_series.loc[start:stop]
        # print('length of data: ', len(data.index)) # this has to be equal to 23
        data['VWAP'] = (data['Volume'] * (data['High'] + data['Low']) / 2).cumsum() / data['Volume'].cumsum()
        daily_percent_change = (data['Close'] / data['Close'].loc[evaluation_date] - 1) * 100
        daily_percent_change = daily_percent_change[evaluation_date:stop]
        stock = StockDataFrame.retype(data)

        data['rsi'] = stock['rsi_3']

        # sort by descending order
        daily_percent_change.sort_values(inplace=True, ascending=False)

        # plot candle chart
        quotes = data[['open', 'high', 'low', 'close', 'volume']]

        qf = cf.QuantFig(quotes, title=quant_title, name='1D Historical', up_color='green',
                         down_color='red')
        qf.add_rsi(periods=3, showbands=True)
        qf.add_support(date=evaluation_date.strftime('%d%b%y'))
        qf.add_volume(colorchange=True)

        # make a radar plot
        data = data.loc[start:evaluation_date]
        vwap = data['vwap'].to_numpy()
        rsi = data['rsi_3'].to_numpy()
        s_rsi = minmax_scale(rsi, feature_range=(0, 1))
        s_rsi[np.isnan(s_rsi)] = 0
        s_vwap = minmax_scale(vwap, feature_range=(0, 1))
        variables = numba_interweave(s_vwap, s_rsi)  # interweave vwap and rsi
        data = np.array(variables)
        # Plot radar plot
        ranges = []
        for k in range(0, len(variables)):
            ranges.append([0, 1])
        fig1 = plt.figure(figsize=(3, 3), dpi=150)
        radar = ComplexRadar(fig1, variables, ranges)
        radar.plot(data)
        radar.fill(data, alpha=0.2)

        # plt.show(block=False)
        plt.draw()
        plt.savefig(radar_quant_title)
        pio.write_image(qf.iplot(asFigure=True), file=quant_title, width=2560, height=1707)
        dump_image = True

        # inference the image!
        img = image.load_img(radar_quant_title, target_size=(450, 450))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)

        if int(classes[0]) == 1:
            print('Target: ', int(classes[0]), self.ticker)
        else:
            print('Decoy: ', int(classes[0]), self.ticker)

        if dump_image:
            return True
        else:
            return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Robin AI inference bot')
    parser.add_argument('--time_period', default=3, type=int,
                        help='Supply either 3 Months or 6 Months timeframe')
    args = parser.parse_args()

    # load the tensor flow model for inference
    tf_file = '../working_tf_model/tf_moon_pattern_v1.h5'
    model = tf.keras.models.load_model(tf_file)

    # load the tickers to invest in
    pandas_fii_investments_file = './robin_screener_data2022-07-19.pkl'

    fii_data = pd.read_pickle(pandas_fii_investments_file)
    # print all rows of a dataframe
    logger.info("##############################################################################")
    logger.info("####################*****Robin AI Inference Bot*****##########################")
    logger.info("##############################################################################")
    pd.set_option('display.max_rows', None)
    logger.info('Loaded the FII data')
    logger.info(fii_data)

    ticker_symbols = fii_data['Ticker'].to_list()
    ticker_symbols = ticker_symbols

    current_month = str(datetime.datetime.now().month)
    current_year = str(datetime.datetime.now().year)

    ticker_symbols = sorted(ticker_symbols)  # sort alphabetically!
    data_return = []
    for i in tqdm(range(len(ticker_symbols))):
        ticker = ticker_symbols[i]

        start_date = today_1 - datetime.timedelta(days=((args.time_period) + 1) * 60)
        t = yf.Ticker(ticker)
        time_series_data = t.history(start=start_date, interval="1d")
        r_utick = 28
        plot = ManualPlotNsave(check_point_file="point_save.pkl", time_series=time_series_data,
                               trailing_time_period='3Mo', forward_time_period='2Mo',
                               uptick_threshold=r_utick, ticker=ticker, rsi_threshold=100)

        if plot.cufflinks_display():
            generate_daily_summary(ticker=ticker, output_file_preix= ticker+'_summary')
