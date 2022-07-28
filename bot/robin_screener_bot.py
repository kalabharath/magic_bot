from logger_setup import *
from finvizfinance.quote import finvizfinance
from finvizfinance.screener.overview import Overview
import pandas as pd
import pickle


def screener(filter):
    foverview = Overview()
    foverview.set_filter(filters_dict=filter)
    df = foverview.screener_view()
    return df


if __name__ == '__main__':
    """
     
    """
    possible_filter = ['Exchange', 'Index', 'Sector', 'Industry', 'Country', 'Market Cap.', 'P/E', 'Forward P/E', 'PEG',
                       'P/S'
                       'P/B', 'Price/Cash', 'Price/Free Cash Flow', 'EPS growththis year', 'EPS growthnext year',
                       'EPS growthpast 5 years', 'EPS growthnext 5 years', 'Sales growthpast 5 years',
                       'EPS growthqtr over qtr',
                       'Sales growthqtr over qtr', 'Dividend Yield', 'Return on Assets', 'Return on Equity',
                       'Return on Investment', 'Current Ratio', 'Quick Ratio', 'LT Debt/Equity', 'Debt/Equity',
                       'Gross Margin',
                       'Operating Margin', 'Net Profit Margin', 'Payout Ratio', 'InsiderOwnership',
                       'InsiderTransactions',
                       'InstitutionalOwnership', 'InstitutionalTransactions', 'Float Short', 'Analyst Recom.',
                       'Option/Short',
                       'Earnings Date', 'Performance', 'Performance 2', 'Volatility', 'RSI (14)', 'Gap',
                       '20-Day Simple Moving Average', '50-Day Simple Moving Average', '200-Day Simple Moving Average',
                       'Change',
                       'Change from Open', '20-Day High/Low', '50-Day High/Low', '52-Week High/Low', 'Pattern',
                       'Candlestick',
                       'Beta', 'Average True Range', 'Average Volume', 'Relative Volume', 'Current Volume', 'Price',
                       'Target Price', 'IPO Date', 'Shares Outstanding', 'Float']

    screen_filter1 = {'Average Volume': 'Over 500K', 'InsiderOwnership': 'Over 30%',
                      'Option/Short': 'Optionable and shortable', '52-Week High/Low': '10% or more above Low',
                      'Float Short': 'Under 5%', 'InstitutionalTransactions': 'Positive (>0%)'}
    screen_filter2 = {'Average Volume': 'Over 1M', 'InsiderOwnership': 'Over 20%', 'InstitutionalOwnership': 'Over 20%',
                      'Option/Short': 'Optionable and shortable', '52-Week High/Low': '10% or more above Low',
                      'Float Short': 'Under 5%', 'InstitutionalTransactions': 'Positive (>0%)'}
    screen_filter3 = {'Average Volume': 'Over 500K', 'InsiderOwnership': 'Any', 'Net Profit Margin': 'Over 30%',
                      'InstitutionalOwnership': 'Over 10%', 'Market Cap.': '+Small (over $300mln)',
                      'Option/Short': 'Optionable and shortable', 'P/E': 'Under 20',
                      '52-Week High/Low': '10% or more above Low', 'Float Short': 'Under 5%',
                      'InstitutionalTransactions': 'Positive (>0%)'}
    screen_filter4 = {'Average Volume': 'Over 500K', 'InstitutionalOwnership': 'Over 30%', 'Operating Margin': 'Over 25%',
                      'Option/Short': 'Optionable and shortable', '52-Week High/Low': '10% or more above Low',
                      'Float Short': 'Under 5%', 'InstitutionalTransactions': 'Positive (>0%)'}

    pd_list = []
    logger.info("##############################################################################")
    logger.info("######################*****Robin Screener Bot*****############################")
    logger.info("##############################################################################")

    for screen in [screen_filter1, screen_filter2, screen_filter3, screen_filter4]:
        tf = screener(filter=screen)
        # drop all rows where market cap is NaN
        tf = tf.dropna(subset=['Market Cap'])
        # tf = tf.dropna(subset=['P/E'])
        # convert NaN values to 0
        tf = tf.fillna(0)
        # drop all rows where Sector is 'Financial'
        tf = tf[tf['Sector'] != 'Financial']
        # drop all rows where Sector is 'Utilities'
        tf = tf[tf['Sector'] != 'Utilities']
        # drop all rows where Industry is Oil & Gas E&P
        tf = tf[tf['Industry'] != 'Oil & Gas E&P']
        tf = tf[tf['Industry'] != 'Oil & Gas Midstream']
        pd_list.append(tf)
    data_df = pd.concat(pd_list)

    # print(data_df)
    # convert price column to float
    data_df['Price'] = data_df['Price'].astype(float)
    # covert P/E to float
    data_df['P/E'] = data_df['P/E'].astype(float)
    # filter out all rows where P/E is more than 100
    data_df = data_df[data_df['P/E'] < 100]

    # filter out all rows where price is less than 10.0
    data_df = data_df[data_df['Price'] > 10.0]
    # convert Market Cap to int
    data_df['Market Cap'] = data_df['Market Cap'].astype(int)
    # filter out all rows where Market Cap is less than $750M
    data_df = data_df[data_df['Market Cap'] > 1000000000]

    # remove duplicates by ticker
    data_df = data_df.drop_duplicates(subset=['Ticker'])
    # reindex
    data_df = data_df.reset_index(drop=True)
    # save dataframe to pickle file
    today = str(pd.Timestamp.today().date())
    data_df.to_pickle('robin_screener_data'+today+'.pkl')
    tickers = data_df['Ticker'].tolist()
    logger.info("Tickers to be analyzed: {}".format(tickers))
    # save tickers to pickle file
    with open('tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)


    """
    for ticker in tf['Ticker'].to_list():
        stock = finvizfinance(ticker)
        stock.ticker_charts()
        fundamentals = stock.ticker_fundament()
        print(fundamentals)
    """
    """
        fun = {'Company': 'BioNTech SE', 'Sector': 'Healthcare', 'Industry': 'Biotechnology', 'Country': 'Germany',
               'Index': '-', 'P/E': '3.14', 'EPS (ttm)': '51.23', 'Insider Own': '62.87%',
               'Shs Outstand': '243.12M', 'Perf Week': '-3.27%', 'Market Cap': '37.44B', 'Forward P/E': '8.95',
               'EPS next Y': '-48.21%', 'Insider Trans': '0.00%', 'Shs Float': '215.45M', 'Perf Month': '16.70%',
               'Income': '12.88B', 'PEG': '-', 'EPS next Q': '6.55', 'Inst Own': '15.90%', 'Short Float': '0.89%',
               'Perf Quarter': '-6.91%', 'Sales': '23.34B', 'P/S': '1.60', 'EPS this Y': '58757.10%',
               'Inst Trans': '0.34%', 'Short Ratio': '1.55', 'Perf Half Y': '-19.61%', 'Book/sh': '65.55',
               'P/B': '2.46', 'ROA': '86.50%', 'Target Price': '242.98', 'Perf Year': '-29.50%', 'Cash/sh': '26.56',
               'P/C': '6.06', 'EPS next 5Y': '-', 'ROE': '122.50%', '52W Range From': '117.08',
               '52W Range To': '457.87', 'Perf YTD': '-36.70%', 'Dividend': '2.11', 'P/FCF': '7.33',
               'EPS past 5Y': '-', 'ROI': '86.30%', '52W High': '-64.83%', 'Beta': '-', 'Dividend %': '1.31%',
               'Quick Ratio': '4.90', 'Sales past 5Y': '-', 'Gross Margin': '83.00%', '52W Low': '37.54%',
               'ATR': '8.25', 'Employees': '3082', 'Current Ratio': '5.10', 'Sales Q/Q': '211.20%',
               'Oper. Margin': '78.80%', 'RSI (14)': '55.52', 'Volatility W': '4.95%', 'Volatility M': '5.60%',
               'Optionable': 'Yes', 'Debt/Eq': '0.01', 'EPS Q/Q': '227.90%', 'Profit Margin': '55.20%',
               'Rel Volume': '0.54', 'Prev Close': '161.44', 'Shortable': 'Yes', 'LT Debt/Eq': '0.01',
               'Earnings': 'Aug 08 BMO', 'Payout': '0.00%', 'Avg Volume': '1.23M', 'Price': '161.03',
               'Recom': '2.50', 'SMA20': '8.46%', 'SMA50': '7.04%', 'SMA200': '-17.70%', 'Volume': '678,393',
               'Change': '-0.25%'}
    """
