import sys, os, glob

def get_moon_tickers():
    """
        Get the tickers of the moon stocks
        """
    tickers = []
    for file in glob.glob('./mooned/*.png'):
        ticker = file.split('/')[2].split('_')[0]
        tickers.append(ticker)
    return tickers


def get_not_moon_tickers(tickers):
    """
        Get the tickers of the not moon stocks
        """
    for ticker in tickers:
        plots = glob.glob(ticker+'*radar*.png')
        for plot in plots:
            mv = 'mv '+plot+' ./not_mooned/'
            os.system(mv)
    return True


if __name__ == '__main__':
    tickers = get_moon_tickers()
    print(tickers)
    get_not_moon_tickers(tickers)
