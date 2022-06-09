"""
Project_Name: magic_bot, File_name: signal_generator.py
Author: kalabharath, Email: kalabharath@gmail.com
"""
from sklearn.preprocessing import minmax_scale
import datetime
import os
import time
import argparse
import cufflinks as cf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.offline as plyo
import tensorflow as tf
import yfinance as yf
from keras_preprocessing import image
from prettytable import PrettyTable

from tqdm import tqdm

from algo.minimal_radar import ComplexRadar

matplotlib.use('TkAgg')


class AutoPlotNsave():
    """
    Supervised selection of 3-month period stock with stock reaching the lowest before a 10% intra-day bump
    and continues exponential climbing.
    1. Displays the gradual drop of stock prices in a three month period where the the normalized 'RSI' and
    'ATR' or 'VWAP" reaches zero.
    2. Draw the support and resistance lines.
    3. Draw the entry and exit prices at 1:3 risk to reward profile.
    4. Save the profile if steps 1-3 above were true.
    """

    def __init__(self, time_series, ticker, rsi_threshold, downside_threshold, tf_model, time_period):
        """[summary]

        Args:
            time_series (dataframe):
            ticker ([type]): [description]
            rsi_threshold ([type]): [description]
            downside_threshold ([type]): [description]
            tf_model ([type]): [description]
        """
        self.time_series = time_series
        self.ticker = ticker
        self.rsi_threshold = rsi_threshold
        self.down_threshold = downside_threshold
        self.tf_model = tf_model
        self.time_period = time_period

    @staticmethod
    def compute_rsi(df, periods):
        """[Computes Relative strength index for a given ]

        Args:
            df ([type]): [description]
            periods ([type]): [description]

        Returns:
            [type]: [description]
        """

        df['Up'] = df['Close'].diff().apply(lambda x: x if x > 0 else 0)
        df['Down'] = df['Close'].diff().apply(lambda x: -x if x < 0 else 0)
        df['UpAvg'] = df['Up'].rolling(window=periods).mean()
        df['DownAvg'] = df['Down'].rolling(window=periods).mean()
        df['RSI'] = 100 - (100 / (1 + df['UpAvg'] / df['DownAvg']))
        return df

    @staticmethod
    def numba_interweave(arr1, arr2):
        """
        interweaves one array with another, eg [1,2,3] and [A, B, C] = [1, A, 2, B, 3, C]
        :param arr1:
        :param arr2:
        :return: interweave array
        """
        res = np.empty(arr1.size + arr2.size, dtype=arr1.dtype)
        for idx, (item1, item2) in enumerate(zip(arr1, arr2)):
            res[idx * 2] = item1
            res[idx * 2 + 1] = item2
        return res

    @staticmethod
    def reduce_array(arr, time_period):
        """
        Dimensonality or data reduction by averaging over a specified window of three.
        :param arr:
        :return:
        """
        factor = int(time_period / 3.0)
        if len(arr) % (3 * factor) == 0:
            t = np.mean(arr.reshape(-1, (3 * factor)), axis=1)
        else:
            t = np.nanmean(
                np.pad(arr.astype(float), (0, (3 * factor) - arr.size % (3 * factor)), mode='constant',
                       constant_values=np.NaN).reshape(-1, 3 * factor), axis=1)

        return t

    def cufflinks_display(self):
        """[summary]

        Returns:
            [type]: [description]
        """

        data = self.compute_rsi(self.time_series, periods=14)
        try:
            start = self.time_series.index[len(self.time_series.index) - (67 * int(self.time_period / 3.0))]
            stop = self.time_series.index[-1]
            evaluation_date = self.time_series.index[- 1]
        except Exception as e:
            print(e)
            return ['Error', self.ticker, 'Could not get data']
        time_format = '%d%b%y'
        quant_title = self.ticker + "_" + start.strftime(time_format) + "_" + evaluation_date.strftime(time_format)

        data = data.loc[start:stop]
        data['VWAP'] = (data['Volume'] * (data['High'] +
                                          data['Close']) / 2).cumsum() / data['Volume'].cumsum()

        # is today's close price less than vwap ?
        vwap_percent = (data['Close'].loc[stop] / data['VWAP'].loc[stop] - 1) * 100

        todays_rsi = data['RSI'].loc[stop]

        rsi_within_threshold = False

        for t_rsi in range(5, self.rsi_threshold):

            if todays_rsi <= t_rsi:
                # print("Rsi is within threshold of ", t_rsi, "/", todays_rsi)
                rsi_within_threshold = True
                break

        if not rsi_within_threshold:
            # print("Rsi is Not within threshold of 5-", self.rsi_threshold, ":", todays_rsi)
            return ['Skip', self.ticker, vwap_percent, todays_rsi]

        # if the downtrend is not within a specified limit, continue

        daily_percent_change = (data['Close'].loc[evaluation_date] / data['Close'] - 1) * 100
        downside_yield_date = daily_percent_change.idxmin()
        downside = round(daily_percent_change.min(), 2)

        if not downside <= self.down_threshold:
            return ['Skip', self.ticker, downside, vwap_percent, todays_rsi]
        # plot candle chart
        quotes = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        qf = cf.QuantFig(quotes, title=quant_title, name='1D Historical', up_color='green', down_color='red')
        qf.add_rsi(periods=14, showbands=True)
        qf.add_bollinger_bands()
        qf.add_support(date=evaluation_date.strftime(time_format))
        qf.add_volume(colorchange=True)
        qf.add_trendline(date0=downside_yield_date.strftime(time_format), date1=evaluation_date.strftime(time_format),
                         text=str(downside))  # add trend line
        # plyo.iplot(qf.iplot(asFigure=True))

        # make a radar plot
        vwap = data['VWAP'].to_numpy()
        rsi = data['RSI'].to_numpy()
        s_rsi = minmax_scale(rsi, feature_range=(0, 1))
        s_vwap = minmax_scale(vwap, feature_range=(0, 1))

        # reduce the array dimensionality/complexity
        s_vwap = self.reduce_array(arr=s_vwap, time_period=self.time_period)
        s_rsi = self.reduce_array(arr=s_rsi, time_period=self.time_period)
        variables = self.numba_interweave(s_vwap, s_rsi)  # interweave vwap and rsi
        data = np.array(variables)
        ranges = []
        for _ in range(0, len(variables)):
            ranges.append([0, 1])
        fig1 = plt.figure(figsize=(3, 3), dpi=150)
        radar = ComplexRadar(fig1, variables, ranges)
        radar.plot(data)
        radar.fill(data, alpha=0.2)

        plt.draw()
        plt.savefig(quant_title + "_radar.png")
        # plt.show(block=False)
        time.sleep(0.10)
        fig1.clear()
        plt.close(fig1)

        pio.write_image(qf.iplot(asFigure=True),
                        file=quant_title + ".png", width=2560, height=1707)
        qf.add_shapes()

        img = image.load_img(quant_title + "_radar.png",
                             target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = self.tf_model.predict(images, batch_size=10)

        if int(classes[0]) == 0:  # decoy
            # mv_file = "mv " + quant_title + "* decoy/"
            # os.system(mv_file)
            return ['Decoy', self.ticker, downside, vwap_percent, todays_rsi]

        elif int(classes[0]) == 1:  # target
            # mv_file = "mv " + quant_title + "* target/"
            # os.system(mv_file)
            return ['Target', self.ticker, downside, vwap_percent, todays_rsi]

        else:
            print("*******: Something is seriously wrong with the logic")

        return ['Skip', 'Error']


if __name__ == '__main__':
    """
    Summary: 

        Returns:
            Table/DataFrame : specifying targets and decoys with the following datapoints
+-------------------+----------+----------+---------------------+--------------------+
| AI Classification |  Ticker  | Downside |     vwap_percent    |     todays_rsi     |
+-------------------+----------+----------+---------------------+--------------------+
|       Target      |   JSL    |  -44.37  | -30.503006540229382 | 18.550959195978393 |
|       Decoy       |  AMBER   |  -40.65  | -24.915387442309257 | 23.092182587967315 |
+-------------------+----------+----------+---------------------+--------------------+

    """

    parser = argparse.ArgumentParser(description='Magic formula for NASDAQ and NYSE')

    parser.add_argument('--time_period', default=6, type=int,
                        help='Supply either 3 Months or 6 Months timeframe')
    parser.add_argument('--downside_threshold', default=-50, type=int, help='Minimum downside threshold')
    parser.add_argument('--rsi_threshold', default=40, type=int, help='Minimum rsi threshold')
    args = parser.parse_args()

    current_month = str(datetime.datetime.now().month)
    current_year = str(datetime.datetime.now().year)

    pandas_fii_investments_file = "./data/" + current_month + "_" + \
                                  current_year + "_data/" + current_month + '_fiis_' + current_year + '.pkl'
    nse_listed_mnc = ['LINDEINDIA', 'PFIZER', 'ABBOTINDIA', 'ASTRAZEN', 'RAJESHEXPO', 'GLAXO', 'MPHASIS',
                      'SANOFI', 'COLPAL', 'CASTROLIND', 'SIEMENS', 'OFSS', 'NESTLEIND', 'BRITANNIA', 'HONAUT',
                      'ABB', 'BOSCHLTD', '3MINDIA', 'AKZOINDIA', 'ASAHIINDIA',
                      'BERGEPAINT', 'CIGNITITEC', 'CLNINDIA', 'DENORA', 'CUMMINSIND', 'DICIND',
                      'DLINKINDIA', 'ESABINDIA', 'FMGOETZE', 'FOSECOIND',
                      'GABRIEL', 'GILLETTE', 'GOODYEAR', 'GPPL', 'HEIDELBERG', 'HONDAPOWER',
                      'INDNIPPON', 'INDOTECH', 'INGERRAND', 'ITDCEM', 'KENNAMET', 'KOKUYOCMLN',
                      'KSB', 'LUMAXIND', 'MOTHERSUMI', 'MPHASIS', 'NOVARTIND',
                      'PAGEIND', 'SHREYAS', 'SKFINDIA',
                      'SMLISUZU', 'TIMKEN', 'VESUVIUS', 'HDFCAMC', 'VEDL']

    tf_file = './tf_model/target_pattern2022GC.h5'
    fii_data = pd.read_pickle(pandas_fii_investments_file)
    print(fii_data)
    model = tf.keras.models.load_model(tf_file)
    today_1 = datetime.date.today()
    exclude_sectors = ['Banks', 'Bank', 'Finance', 'Insurance']  # these sectors have no tangible assets

    for exclude_sector in exclude_sectors:
        fii_data.drop(fii_data.loc[fii_data['sectors'] == exclude_sector].index, inplace=True)

    ticker_symbols = fii_data['ticker'].to_list()
    ticker_symbols = ticker_symbols + nse_listed_mnc
    ticker_symbols = sorted(ticker_symbols)  # sort alphabetically!
    print(ticker_symbols, len(ticker_symbols))
    data_return = []

    for i in tqdm(range(len(ticker_symbols))):
        company = ticker_symbols[i]
        if company.isdigit():
            continue
        start_date = today_1 - datetime.timedelta(days=((args.time_period) + 1) * 30)
        t = yf.Ticker(company + '.NS')
        time_series_data = t.history(start=start_date, interval="1d")
        plot = AutoPlotNsave(time_series=time_series_data, ticker=company, rsi_threshold=args.rsi_threshold,
                             downside_threshold=args.downside_threshold, tf_model=model, time_period=args.time_period)
        data_return.append(plot.cufflinks_display())

    # plot_summary & display targets
    targets, decoys = [], []
    for hit in data_return:
        if len(hit) > 0:
            if hit[0] == 'Target':
                targets.append(hit)
            elif hit[0] == 'Decoy':
                decoys.append(hit)
            elif hit[0] == 'Error':
                print ("Coundn't get data for this ticker: ", hit[1])
    x = PrettyTable()
    c_columns = ['AI Classification', "Ticker", "Downside", "vwap_percent", "todays_rsi"]
    x.field_names = c_columns
    x.add_rows(targets)
    x.add_rows(decoys)
    print(x)  # save and dump every day dataframes for further analytic analysis # load daily hits as a pandas df

    df_path = 'magic_formula_daily.pkl'
    # check if the last modifed date is today, if so skip.
    from summary import generate_daily_summary

    generate_daily_summary(output_file_preix='summary_' + str(args.time_period) + str(today_1))

    if os.path.isfile(df_path):
        stat_result = os.stat(df_path)
        modified = datetime.datetime.fromtimestamp(stat_result.st_mtime)
        last_modified = modified.date()

        if last_modified == today_1:
            print("Looks like this is the second run of today")
        else:
            if os.path.isfile(df_path):
                backup = "cp " + df_path + " " + df_path + str(today_1)
                os.system(backup)
                df_file = pd.read_pickle(df_path)
                tdf_file = pd.DataFrame(decoys + targets, columns=c_columns)
                tdf_file['dates'] = pd.Timestamp(today_1)
                df_file = df_file.append(tdf_file)
                print(df_file)
                df_file.to_pickle(df_path)
    else:
        df_file = pd.DataFrame(decoys + targets, columns=c_columns)
        df_file['dates'] = pd.Timestamp(today_1)
        df_file.to_pickle(df_path)
