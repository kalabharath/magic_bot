from logger_setup import *
import robin_stocks
import yfinance as yf
from robin_buy_bot import *


if __name__ == '__main__':
    logger.info("##############################################################################")
    logger.info("######################*******Robin Sell Bot*******############################")
    logger.info("##############################################################################")
    # login to robinhood
    # save username and passowrd in an encrypted file and exclude from git
    username, password, otp_secret = get_credentials()
    totp = pyotp.TOTP(otp_secret).now()
    logger.info("TOTP: {}".format(totp))
    login = robin_stocks.robinhood.login(username=username, password=password, mfa_code=totp)
    logger.info("Login status: {}".format(login))

    # get account positions
    positions = robin_stocks.robinhood.options.get_open_option_positions()
    for position in positions:
        logger.info("Position: {}".format(position))
        """
        position = {'account': 'https://api.robinhood.com/accounts/883363160/', 'average_price': '190.2500',
                    'chain_id': '32d41003-5e1a-4e6f-96ae-b74b733809f1', 'chain_symbol': 'BNTX',
                    'id': 'e40e0cba-9cb6-4801-b152-4c7d029fb00d',
                    'option': 'https://api.robinhood.com/options/instruments/2370919c-5e90-490a-86cf-8ff589b312bb/',
                    'type': 'long', 'pending_buy_quantity': '0.0000', 'pending_expired_quantity': '0.0000',
                    'pending_expiration_quantity': '0.0000', 'pending_exercise_quantity': '0.0000',
                    'pending_assignment_quantity': '0.0000', 'pending_sell_quantity': '0.0000', 'quantity': '4.0000',
                    'intraday_quantity': '0.0000', 'intraday_average_open_price': '0.0000',
                    'created_at': '2022-07-05T14:14:09.288276Z', 'trade_value_multiplier': '100.0000',
                    'updated_at': '2022-07-05T19:58:36.454151Z',
                    'url': 'https://api.robinhood.com/options/positions/e40e0cba-9cb6-4801-b152-4c7d029fb00d/',
                    'option_id': '2370919c-5e90-490a-86cf-8ff589b312bb'}
        """

        market_data_for_the_option = robin_stocks.robinhood.get_option_market_data_by_id(id=position['option_id'])
        quantity = float(position['quantity'])
        quantity = int(quantity)
        # print(market_data_for_the_option)
        logger.info("Market data for the option: {}".format(market_data_for_the_option))
        """
        market_data_for_the_option = [
            {'adjusted_mark_price': '1.350000', 'adjusted_mark_price_round_down': '1.350000', 'ask_price': '1.500000',
             'ask_size': 2, 'bid_price': '1.200000', 'bid_size': 14, 'break_even_price': '241.350000',
             'high_price': None,
             'instrument': 'https://api.robinhood.com/options/instruments/2370919c-5e90-490a-86cf-8ff589b312bb/',
             'instrument_id': '2370919c-5e90-490a-86cf-8ff589b312bb', 'last_trade_price': '1.550000',
             'last_trade_size': 1, 'low_price': None, 'mark_price': '1.350000', 'open_interest': 24,
             'previous_close_date': '2022-07-14', 'previous_close_price': '1.830000',
             'updated_at': '2022-07-15T17:03:14.813383936Z', 'volume': 0, 'symbol': 'BNTX',
             'occ_symbol': 'BNTX  220916C00240000', 'state': 'active', 'chance_of_profit_long': '0.045554',
             'chance_of_profit_short': '0.954446', 'delta': '0.079316', 'gamma': '0.003556',
             'implied_volatility': '0.620803', 'rho': '0.019732', 'theta': '-0.049094', 'vega': '0.098813',
             'high_fill_rate_buy_price': '1.426000', 'high_fill_rate_sell_price': '1.273000',
             'low_fill_rate_buy_price': '1.278000', 'low_fill_rate_sell_price': '1.421000'}]
        """

        # calculate profit/loss
        purchase_price = float(position['average_price']) * quantity
        current_price = float(market_data_for_the_option[0]['adjusted_mark_price_round_down']) * 100 * quantity
        percentage_pl = ((current_price - purchase_price) / purchase_price) * 100
        # print("percentage_pl", percentage_pl)
        logger.info("percentage_pl: {}".format(percentage_pl))
        # decide to sell or not
        # 1. check if the underlying stock has 15% correction from the peak price
        tdate = position['created_at']
        purchase_date = tdate.split('T')[0]
        ticker = market_data_for_the_option[0]['symbol']
        t = yf.Ticker(ticker)
        df = t.history(start=purchase_date, interval='1d')
        today = datetime.today()
        evaluation_date = df.index[-1]
        daily_percent_change = (df['Close'].loc[evaluation_date] / df['Close'] - 1) * 100
        downside_yield_date = daily_percent_change.idxmin()
        downside = round(daily_percent_change.min(), 2)
        upside = round(daily_percent_change.max(), 2)

        # 2. check if it is halfway to expiry date
        option_data = robin_stocks.robinhood.get_option_instrument_data_by_id(id=position['option_id'])
        expiry_date = option_data['expiration_date']
        strike_price = option_data['strike_price']
        tick_multiple = randint(1, 3)
        sell_price = float(market_data_for_the_option[0]['ask_price']) - (
                    tick_multiple * float(option_data['min_ticks']['below_tick']))

        days_to_expiry = (datetime.strptime(expiry_date, '%Y-%m-%d') - datetime.strptime(purchase_date,
                                                                                         '%Y-%m-%d')).days
        days_lapsed_so_far = (evaluation_date - datetime.strptime(purchase_date, '%Y-%m-%d')).days

        # 3. check for option profits threshold
        sell_option = False
        if downside <= -15.0:  # if the underlying stock dropped 15% or more, stop_loss to prevent a loss
            sell_option = True
            # print ("sell_option: -", sell_option, 'Hit the underlying stock drop 15% or more')
            logger.info("Hit the underlying stock drop 15% or more")
            logger.info("sell_option: - {}".format(sell_option))
        elif days_to_expiry / days_lapsed_so_far <= 2:
            # print ("sell_option: -", sell_option, 'Hit the halfway to expiry date')
            logger.info("sell_option: - {}".format(sell_option))
            logger.info("Hit the halfway to expiry date")
            sell_option = True
        elif percentage_pl >= 50:  # if the option has made 50% profit, sell it
            # print ("sell_option: -", sell_option, 'Hit the option profit threshold')
            logger.info("Hit the option profit threshold")
            logger.info("sell_option: - {}".format(sell_option))
            sell_option = True
        else:
            # print ("sell_option: -", sell_option, 'No sell criteria met')
            logger.info("No sell criteria met")
            logger.info("sell_option: - {}".format(sell_option))
            sell_option = False

        if sell_option:
            sell = robin_stocks.robinhood.order_sell_option_limit(positionEffect="close", creditOrDebit="credit",
                                                                  price=sell_price, symbol=ticker, quantity=quantity,
                                                                  expiry_date=expiry_date, strike=strike_price,
                                                                  optionType="call", timeInForce="gfd")
        pause_execution()
