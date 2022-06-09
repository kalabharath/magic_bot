Module bot.signal_generator
===========================
Project_Name: magic_bot, File_name: signal_generator.py
Author: kalabharath, Email: kalabharath@gmail.com

Classes
-------

`AutoPlotNsave(time_series, ticker, rsi_threshold, downside_threshold, tf_model, time_period)`
:   Supervised selection of 3-month period stock with stock reaching the lowest before a 10% intra-day bump
    and continues exponential climbing.
    1. Displays the gradual drop of stock prices in a three month period where the the normalized 'RSI' and
    'ATR' or 'VWAP" reaches zero.
    2. Draw the support and resistance lines.
    3. Draw the entry and exit prices at 1:3 risk to reward profile.
    4. Save the profile if steps 1-3 above were true.
    
    [summary]
    
    Args:
        time_series (dataframe):
        ticker ([type]): [description]
        rsi_threshold ([type]): [description]
        downside_threshold ([type]): [description]
        tf_model ([type]): [description]

    ### Static methods

    `compute_rsi(df, periods)`
    :   Computes Relative strength index for a given
        
        Args:
            df (data frame): contains time series
            periods (int): rolling window period, typically 14 days
        
        Returns:
            df (data frame): the returned dataframe now contains computed 'RSI' column

    `numba_interweave(arr1, arr2)`
    :   interweaves one array with another, eg [1,2,3] and [A, B, C] = [1, A, 2, B, 3, C]
        :param arr1:
        :param arr2:
        :return: interweave array

    `reduce_array(arr, time_period)`
    :   Dimensonality or data reduction by averaging over a specified window of three.
        :param arr:
        :return:

    ### Methods

    `cufflinks_display(self)`
    :   [summary]
        
        Returns:
            [type]: [description]