import glob
import os
import pickle
import random

import cufflinks as cf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.offline as plyo
from minimal_radar import ComplexRadar
from sklearn.preprocessing import minmax_scale
from stockstats import StockDataFrame

import plotly.io as pio


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


def reduce_array(arr):
    """
    Dimensonality or data reduction by averaging over a specified window of three.
    :param arr:
    :return:
    """
    if len(arr) % 3 == 0:
        t = np.mean(arr.reshape(-1, 3), axis=1)
    else:
        t = np.nanmean(
            np.pad(arr.astype(float), (0, 3 - arr.size % 3), mode='constant', constant_values=np.NaN).reshape(-1, 3),
            axis=1)

    return t


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

    def __init__(self, check_point_file, time_series, trailing_time_period, forward_time_period, uptick_threshold, ticker, rsi_threshold):
        self.check_point_file = check_point_file
        self.time_series = time_series
        self.uptick_threhold = uptick_threshold
        self.ticker = ticker
        self.rsi_threshold = rsi_threshold
        self.total_days = int(trailing_time_period[0]) * 22
        self.forward_days = int(forward_time_period[0]) * 22

    def yes_or_no(self, question):
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False
        else:
            return self.yes_or_no("Uhhhh... please enter ")

    def cufflinks_display(self):
        if os.path.isfile(self.check_point_file):
            with open(self.check_point_file, 'rb') as f:
                point_save = pickle.load(f)
        else: # create a check_point_file
            point_save  = [[],[]]
            with open(self.check_point_file, 'wb') as f:
                pickle.dump(point_save, f)

        for i in range(0, len(self.time_series.index) - self.total_days):

            if random.randint(0,1):
                continue
            # for i in range(0, 2):
            start = self.time_series.index[i]
            stop = self.time_series.index[i + self.total_days]
            # get precise index after 3 month period
            evaluation_date = self.time_series.index[i + (self.total_days - self.forward_days)]
            quant_title = self.ticker + "_" + start.strftime('%d%b%y') + "_" + evaluation_date.strftime('%d%b%y')
            if quant_title in point_save[0] or quant_title in point_save[1]:
                continue
            else:
                pass

            # print (start, stop)
            data = self.time_series.loc[start:stop]

            data['VWAP'] = (data['Volume'] * (data['High'] + data['Low']) / 2).cumsum() / data['Volume'].cumsum()
            # daily_percent_change = (data['Close'] / data['Close'].shift(1) - 1) * 100

            daily_percent_change = (data['Close'] / data['Close'].loc[evaluation_date] - 1) * 100
            daily_percent_change = daily_percent_change[evaluation_date:stop]
            stock = StockDataFrame.retype(data)
            data['rsi'] = stock['rsi_14']
            # Get total number of up_tick events:
            bool_evaluation_day = False
            upside = "Yet to be computed"
            upside_yield_date = stop
            for j in range(0, 5):
                threshold = self.uptick_threhold - j
                total_events = daily_percent_change[daily_percent_change.gt(threshold)]
                if len(total_events) > 0:
                    if (data['rsi'].loc[evaluation_date] < self.rsi_threshold) and (
                            data['close'].loc[total_events.index[0]] > data['close'].loc[evaluation_date]):
                        upside = str(total_events.to_list()[0])
                        upside_yield_date = total_events.index[0]
                        print(upside, total_events, "rsi=", data['rsi'].loc[evaluation_date])
                        print(j, threshold, len(total_events))
                        bool_evaluation_day = True
                        break
                    else:
                        continue
                else:
                    pass
            if not bool_evaluation_day:
                continue
            # plot candle chart
            # quotes = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            quotes = data[['open', 'high', 'low', 'close', 'volume']]
            qf = cf.QuantFig(quotes, title=quant_title, name='1D Historical', up_color='green',
                             down_color='red')
            qf.add_rsi(periods=14, showbands=True)
            qf.add_support(date=evaluation_date.strftime('%d%b%y'))
            qf.add_trendline(date0=start.strftime('%d%b%y'), date1=evaluation_date.strftime('%d%b%y'))
            qf.add_trendline(date0=evaluation_date.strftime('%d%b%y'), date1=upside_yield_date.strftime('%d%b%y'),
                             text=upside)
            qf.add_volume(colorchange=True)
            plyo.iplot(qf.iplot(asFigure=True), image='png', filename=quant_title + ".png")  # add options to save and

            # init_notebook_mode(connected=True)
            # cf.go_online()


            # make a radar plot
            data = data.loc[start:evaluation_date]
            vwap = data['vwap'].to_numpy()
            rsi = data['rsi_14'].to_numpy()
            s_rsi = minmax_scale(rsi, feature_range=(0, 1))
            s_vwap = minmax_scale(vwap, feature_range=(0, 1))

            # reduce the array dimensonality or complexity
            s_vwap = reduce_array(s_vwap)
            s_rsi = reduce_array(s_rsi)
            variables = numba_interweave(s_vwap, s_rsi)  # interweave vwap and rsi
            data = np.array(variables)

            ranges = []
            for k in range(0, len(variables)):
                ranges.append([0, 1])
            fig1 = plt.figure(figsize=(6, 6))
            radar = ComplexRadar(fig1, variables, ranges)
            radar.plot(data)
            radar.fill(data, alpha=0.2)
            point_save[0].append(quant_title)
            plt.draw()
            plt.show(block=False)
            # if self.yes_or_no("Do you want to save the radar plot? :"):
            if True:
                plt.draw()
                plt.savefig(quant_title + "_radar.png")
                pio.write_image(qf.iplot(asFigure=True), file=quant_title + ".svg", width=2000, height=1000)
                point_save[1].append(quant_title)
            with open(self.check_point_file, 'wb') as f: # save the check_point
                pickle.dump(point_save, f)

                if random.randint(0,1):
                    return True
        return True


if __name__ == '__main__':

    ticker_symbols= ['PFE']
    for ticker in ticker_symbols:
        time_series_data = pd.read_pickle('./PFE_11_Jun_2022.pkl')
        # compute a rolling window time series
        r_utick =  random.randint(0, 15)
        r_rsi = random.randint(30, 70)
        plot = ManualPlotNsave(check_point_file="point_save.pkl", time_series=time_series_data, trailing_time_period='5Mo', forward_time_period='2Mo',
                               uptick_threshold=r_utick, ticker=ticker, rsi_threshold=r_rsi)

        plot.cufflinks_display()
