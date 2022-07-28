import glob
import os

import cufflinks as cf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minimal_radar import ComplexRadar
from sklearn.preprocessing import minmax_scale
from stockstats import StockDataFrame
import datetime
import plotly.io as pio
from PIL import Image


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
    print("Combining pngs to pdfs")
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

    def get_unusual_volume_days(self):
        """

        :return:
        """
        unusual_volume_days = []
        copy_of_time_series = self.time_series.copy(deep=True)

        for i in range(0, len(copy_of_time_series.index) - self.total_days):
            start = copy_of_time_series.index[i]
            stop = copy_of_time_series.index[i + self.total_days]
            evaluation_date = copy_of_time_series.index[i + (self.total_days - self.forward_days)]
            day_before_evaluation_date = self.time_series.index[i + (self.total_days - self.forward_days) - 1]

            # compute the average volume for the period
            avg_volume = copy_of_time_series.loc[start:day_before_evaluation_date, 'Volume'].mean()
            # compute standard deviation of the volume for the period
            std_volume = copy_of_time_series.loc[start:day_before_evaluation_date, 'Volume'].std()

            # slice dataframe with the evaluation date
            tdata = copy_of_time_series[start:day_before_evaluation_date]
            # convert timestamps to seconds
            tdata['timestamp'] = tdata.index.astype('int64') / 1e9
            # convert timestamps to days
            tdata['day'] = tdata['timestamp'].apply(lambda x: int(x / 86400))

            # check if the evaluation date volume is unusually higher than average volume
            if copy_of_time_series.loc[evaluation_date, 'Volume'] > (avg_volume + (5 * std_volume)):
                print("Volume is unusually high")
                print(copy_of_time_series.loc[evaluation_date, 'Volume'] / avg_volume,
                      copy_of_time_series.loc[evaluation_date, 'Volume'] / (avg_volume + (5 * std_volume)))
                f = np.polyfit(tdata['day'], tdata['Close'], deg=1)
                degrees = np.degrees(np.tan(f[0]))

                # check if the degrees of the straight line is horizontal
                if -10.0 <= degrees <= 30.0:
                    unusual_volume_days.append(evaluation_date)
                    print(evaluation_date, f, degrees)
                else:
                    continue

            else:
                continue

        threshold_days = []
        for date in unusual_volume_days:
            threshold_days.append(date)
            for i in range(1, 6): # look for the next 5 days
                threshold_days.append(date + datetime.timedelta(days=i))
        return threshold_days

    def cufflinks_display(self):
        """
        Display the data in a cufflinks plot.
        :return: boolean
        """
        threshold_days = self.get_unusual_volume_days()
        print (threshold_days)
        dump_image = False
        for i in range(0, len(self.time_series.index) - self.total_days):
            start = self.time_series.index[i]
            stop = self.time_series.index[i + self.total_days]
            evaluation_date = self.time_series.index[i + (self.total_days - self.forward_days)]
            if evaluation_date in threshold_days:
                pass
            else:
                continue

            # simple_quant_title
            quant_title = self.ticker + "_candle_" + evaluation_date.strftime('%d%b%y') + ".png"
            radar_quant_title = self.ticker + "_radar_" + evaluation_date.strftime('%d%b%y') + ".png"

            data = self.time_series.loc[start:stop]
            data['VWAP'] = (data['Volume'] * (data['High'] + data['Low']) / 2).cumsum() / data['Volume'].cumsum()
            daily_percent_change = (data['Close'] / data['Close'].loc[evaluation_date] - 1) * 100
            daily_percent_change = daily_percent_change[evaluation_date:stop]
            stock = StockDataFrame.retype(data)

            data['rsi'] = stock['rsi_3']

            # sort by descending order
            daily_percent_change.sort_values(inplace=True, ascending=False)
            # extract the first value of the sorted array
            upside = daily_percent_change.iloc[0]

            # find the index of the dataframe
            upside_yield_date = daily_percent_change.index[0]

            # plot candle chart
            quotes = data[['open', 'high', 'low', 'close', 'volume']]

            qf = cf.QuantFig(quotes, title=quant_title, name='1D Historical', up_color='green',
                             down_color='red')
            qf.add_rsi(periods=3, showbands=True)
            qf.add_support(date=evaluation_date.strftime('%d%b%y'))
            # qf.add_trendline(date0=start.strftime('%d%b%y'), date1=evaluation_date.strftime('%d%b%y'))
            qf.add_trendline(date0=evaluation_date.strftime('%d%b%y'), date1=upside_yield_date.strftime('%d%b%y'),
                             text=upside)
            qf.add_volume(colorchange=True)

            # make a radar plot
            data = data.loc[start:evaluation_date]
            print ('Min data points :- ',len(data.index))
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

        if dump_image:
            return True
        else:
            return False


if __name__ == '__main__':

    ticker_symbols = glob.glob("./historical_data/*.pkl")
    # ticker_symbols = ['./historical_data/ZTS_14_Jun_2022.pkl']
    print(ticker_symbols)

    for ticker in ticker_symbols:
        print(ticker)
        try:
            time_series_data = pd.read_pickle(ticker)
        except:
            print("Error reading pickle file")
            continue

        ticker = ticker.split("/")[-1].split("_")[0]
        print(ticker)
        if os.path.isfile(ticker + '_summary.pdf'):
            print("File already exists")
            continue
        # compute a rolling window time series
        r_utick = 28
        plot = ManualPlotNsave(check_point_file="point_save.pkl", time_series=time_series_data,
                               trailing_time_period='3Mo', forward_time_period='2Mo',
                               uptick_threshold=r_utick, ticker=ticker, rsi_threshold=100)

        if plot.cufflinks_display():
            generate_daily_summary(ticker=ticker, output_file_preix=ticker + '_summary')
