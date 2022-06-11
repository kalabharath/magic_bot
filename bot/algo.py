from finvizfinance.quote import finvizfinance
import pandas as pd


def get_tickers():
    tickers = ['tsla']
    return tickers


def screener(filter):
    from finvizfinance.screener.overview import Overview
    foverview = Overview()
    foverview.set_filter(filters_dict=filter)
    df = foverview.screener_view()
    pd.set_option('display.max_columns', None)
    df.head()
    return df


if __name__ == '__main__':

    screen_filter1 = {'Average Volume': 'Over 1M', 'InsiderOwnership': 'Over 40%', 'P/E': 'Under 20',
                      'InstitutionalOwnership': 'Over 10%'}
    screen_filter2 = {'Average Volume': 'Over 1M', 'InsiderOwnership': 'Over 20%', 'P/E': 'Under 30',
                      'InstitutionalOwnership': 'Over 10%'}
    for screen in [screen_filter1, screen_filter2]:
        t = screener(filter= screen)
        print (t)

        for ticker in t['Ticker'].to_list():
            stock = finvizfinance(ticker)
            stock.ticker_charts()
            fundamentals = stock.ticker_fundament()

