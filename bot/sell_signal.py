#!/usr/bin/python3
import os

import cufflinks as cf
import matplotlib
import numpy as np
import pandas as pd
import plotly.io as pio

matplotlib.use('TkAgg')
import plotly.offline as plyo
import datetime
import yfinance as yf
import tensorflow as tf
from dateutil.relativedelta import relativedelta


def model_forecast(model, series, window_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


def get_future_price(dataset, tf_model, window_size):
    df = dataset.resample('M').mean()
    # series = df['Close'].to_list()
    all_series = [df['Open'].to_list(), df['High'].to_list(), df['Low'].to_list(), df['Close'].to_list()]
    # make four different series
    predicted = []
    for series in all_series:
        minimum = np.min(series)
        maximum = np.max(series)
        series -= minimum
        series /= maximum
        forecast = model_forecast(tf_model, series[-window_size:], window_size=24)
        scaled_forecast = forecast * maximum
        scaled_forecast = scaled_forecast + minimum
        predicted.append(scaled_forecast[0][0])
    predicted.append(df.iloc[-1]["Volume"])
    return predicted


if __name__ == '__main__':

    tickers = ['EPL', 'IOLCP', 'JUBLPHARMA', 'SUNTECK' ]
    #tickers = ['LAOPALA']
    for ticker in tickers:
        # 1. load the time_series_model
        time_series_file = "../time_series_data/" + ticker + "_timeseries.h5"
        model = "Empty model"
        time_series_model = False
        if os.path.isfile(time_series_file):
            model = tf.keras.models.load_model(time_series_file)
            time_series_model = True
        else:
            print("Time series file not found: ", ticker)
            time_series_model = False
            

        # 2. load the ticker symbol historical data

        # get fourteen month date
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=120)
        two_years_ago = today - relativedelta(months=-25)
        two_years_ago = today.replace(day=1) + relativedelta(months=-25)

        t = yf.Ticker(ticker + '.NS')
        time_series_data = t.history(start=two_years_ago, interval="1d")
        """
        print(time_series_data.tail(5))
        
        exclude_dates = [datetime.datetime.strptime('Feb 03 2021', '%b %d %Y'),
                         datetime.datetime.strptime('Feb 02 2021', '%b %d %Y')]
        
        for date in exclude_dates:
            time_series_data = time_series_data.drop(date)
        print(time_series_data.tail(5))
        """
        if time_series_model:
            prediction = get_future_price(dataset=time_series_data, tf_model=model, window_size=24)
            print(prediction)
            today = time_series_data.index[-1]
            future_date = today + relativedelta(weeks=+1)
            three_months_ago = today + relativedelta(months=-3)

            new_row = pd.DataFrame([prediction], columns=["Open", "High", "Low", "Close", "Volume"], index=[future_date])
            df1 = pd.concat([time_series_data, pd.DataFrame(new_row)], ignore_index=False)
            upside = str((df1.iloc[-1]["Close"] / df1.iloc[-2]["Close"] - 1) * 100)
        
            qf = cf.QuantFig(df1[three_months_ago:future_date], title=ticker, name='1D Historical', up_color='green',
                         down_color='red')
            qf.add_rsi(periods=14, showbands=True)
            #qf.add_ema()        
            qf.add_bollinger_bands()
            qf.add_support(date=today.strftime('%d%b%y'))
            qf.add_volume(colorchange=True)
            qf.add_trendline(date0=today.strftime('%d%b%y'), date1=future_date.strftime('%d%b%y'),
            text=upside)

        else:
            today = time_series_data.index[-1]
            future_date = today 
            three_months_ago = today + relativedelta(months=-3)            
            df1 = time_series_data
            qf = cf.QuantFig(df1[three_months_ago:future_date], title=ticker, name='1D Historical', up_color='green',
                         down_color='red')
            qf.add_rsi(periods=14, showbands=True)            
            qf.add_bollinger_bands()
            qf.add_support(date=today.strftime('%d%b%y'))
            qf.add_volume(colorchange=True)            
        plyo.iplot(qf.iplot(asFigure=True), image='png', filename=ticker + "_timeseries_prediction.png")
        pio.write_image(qf.iplot(asFigure=True), file=ticker +"_"+today.strftime('%d%b%y') +"_timeseries_prediction.svg", width=2000, height=1000)
