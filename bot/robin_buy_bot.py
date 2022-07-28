from logger_setup import *
from datetime import datetime
from random import gauss, random, randint
from time import sleep
import numpy as np
import pandas as pd
import robin_stocks
from dateutil.relativedelta import FR, relativedelta
import pyotp


def get_credentials():
    """
    This function will get the credentials from the credentials file.
    """
    with open('credentials.txt') as f:
        lines = f.readlines()
        otp_secret = lines[2].strip()
        usrname = lines[1].strip()
        passwd = lines[0].strip()
        return usrname, passwd, otp_secret


def pause_execution():
    # wait for a fair bit of human mimicking time to place order
    s = random()
    if s < 0.3:
        t = gauss(10, 2)
    else:
        t = gauss(30, 5)
    t = min(max(t, 120), 5)
    print('Sleeping for ', t, ' seconds, to spoof algos')
    sleep(t)
    return True


# main
if __name__ == '__main__':
    logger.info("##############################################################################")
    logger.info("######################*****Robin Buy Bot*****############################")
    logger.info("##############################################################################")
    """
    # options trading strategy
    1. Select an option that's 3 months away.
    2. make an option spread of atleast 3 different strikes whose strike price is atleast 20-30% farther than the current stock price.    
    3. Sell the option that's half way to expiry.
    """
    # login to robinhood
    # save username and passowrd in an encrypted file and exclude from git
    username, password, otp_secret = get_credentials()
    totp = pyotp.TOTP(otp_secret).now()
    logger.info("6 digit TOTP: {}".format(totp))
    # print('6digit_code', totp)
    login = robin_stocks.robinhood.login(username=username, password=password, mfa_code=totp)
    ticker = 'ISRG'



    """
    # options trading strategy
    1. Select an option that's 3 months away
    """
    # current date
    today = datetime.now()
    last_week = -1
    optionData = {}
    while True:
        # get the date of that is 3 months away from current date and that is a friday or the latest possible friday
        last_friday = datetime.date(today) + relativedelta(day=31, weekday=FR(last_week), months=3)
        last_friday = last_friday.strftime("%Y-%m-%d")
        # get options data
        optionData = robin_stocks.robinhood.find_options_by_expiration([ticker], expirationDate=last_friday,
                                                                       optionType='call')
        if len(optionData) > 1:
            break
        else:
            last_week = last_week - 1  # if no option is found, try the previous friday

    # change the dictionary to a dataframe
    optionData = pd.DataFrame(optionData)
    optionData['strike_price'] = optionData['strike_price'].astype(float)
    optionData['bid_price'] = optionData['bid_price'].astype(float)

    # get current market price
    current_market_price = robin_stocks.robinhood.markets.get_latest_price(ticker)
    current_market_price = float(current_market_price[0])
    # print('current market price - ', current_market_price)
    logger.info("current market price - {}".format(current_market_price))

    """
    # options trading strategy
    # 2. select the option whose strike price is atleast 20-30% farther than the current stock price.
    # randomly add 20%-30% to the current market price    
    """
    # get the option whose strike price is atleast 20-30% farther than the current stock price.
    otm_strike_price = current_market_price * (1 + np.random.uniform(0.2, 0.3))
    otm_strike_price = int(otm_strike_price)

    # get the option data for the otm strike price from the dataframe
    df2 = (optionData.loc[optionData['strike_price'] >= otm_strike_price])
    df2 = df2.sort_values(by=['strike_price'])

    # to define quantity, get account balance

    profile_info = robin_stocks.robinhood.profiles.load_account_profile(info=None)
    # print (profile_info)
    logger.info("profile_info - {}".format(profile_info))
    buying_power = profile_info['buying_power']
    # print ('buying power - ', buying_power)
    logger.info("buying power - {}".format(buying_power))
    # get the 20% of buying power
    buying_power = round(float(buying_power) * 0.2, 2)
    # print (' 20% of buying power - ', buying_power)
    logger.info("20% of buying power - {}".format(buying_power))
    # create a vertical spread
    for i in range(0, 4):
        # extract value that is 20-30 percent out of strike price
        otm_data_at_strike_price = df2.iloc[i]
        # print(otm_data_at_strike_price)
        logger.info("otm_data_at_strike_price - {}".format(otm_data_at_strike_price))
        # print('otm stricke price -', otm_strike_price)
        logger.info("otm stricke price - {}".format(otm_strike_price))
        tick_multiple = randint(1, 3)
        ask_price = otm_data_at_strike_price['bid_price'] + tick_multiple * (
            float(otm_data_at_strike_price['min_ticks']['above_tick']))
        ask_price = round(ask_price, 2)

        # buying power for the current trade that defines the quantity
        t_buying_power = buying_power/ float(len(range(0, 4)))
        quantity = t_buying_power / ask_price
        quantity = int(quantity)
        if quantity <=0:
            # print ('Not enough buying power to buy')
            logger.info("Not enough buying power to buy")
            continue


        # print('ask price - ', ask_price, 'expiration_date -', last_friday, ' Strike_price -',  otm_data_at_strike_price['strike_price'])

        order_conformation = robin_stocks.robinhood.orders.order_buy_option_limit(positionEffect='open',
                                                                                  creditOrDebit='debit',
                                                                                  price=ask_price, symbol=ticker,
                                                                                  quantity=quantity,
                                                                                  expirationDate=last_friday,
                                                                                  strike=otm_data_at_strike_price[
                                                                                      'strike_price'],
                                                                                  optionType='call',
                                                                                  timeInForce='gfd', jsonify=True)
        # print(order_conformation)
        logger.info("order_conformation - {}".format(order_conformation))
        pause_execution() # wait for a fair bit of human mimicking time to place next order

        """
        chain_id                                       32d41003-5e1a-4e6f-96ae-b74b733809f1
        chain_symbol                                                                   BNTX
        created_at                                              2022-06-03T01:05:25.342206Z
        expiration_date                                                          2022-09-16
        id                                             4c4dc5db-2f88-4856-a3b3-dfa030f2a33d
        issue_date                                                               2022-06-03
        min_ticks                         {'above_tick': '0.10', 'below_tick': '0.05', '...
        rhs_tradability                                                          untradable
        state                                                                        active
        strike_price                                                                    125
        tradability                                                                tradable
        type                                                                           call
        updated_at                                           2022-07-01T19:59:49.900643584Z
        url                               https://api.robinhood.com/options/instruments/...
        sellout_datetime                                          2022-09-16T19:00:00+00:00
        long_strategy_code                          4c4dc5db-2f88-4856-a3b3-dfa030f2a33d_L1
        short_strategy_code                         4c4dc5db-2f88-4856-a3b3-dfa030f2a33d_S1
        adjusted_mark_price                                                       39.300000
        adjusted_mark_price_round_down                                            39.300000
        ask_price                                                                 40.300000
        ask_size                                                                         37
        bid_price                                                                 38.300000
        bid_size                                                                         22
        break_even_price                                                         164.300000
        high_price                                                                     None
        instrument                        https://api.robinhood.com/options/instruments/...
        instrument_id                                  4c4dc5db-2f88-4856-a3b3-dfa030f2a33d
        last_trade_price                                                          28.660000
        last_trade_size                                                                   1
        low_price                                                                      None
        mark_price                                                                39.300000
        open_interest                                                                    13
        previous_close_date                                                      2022-06-30
        previous_close_price                                                      32.950000
        volume                                                                            0
        symbol                                                                         BNTX
        occ_symbol                                                    BNTX  220916C00125000
        chance_of_profit_long                                                      0.388581
        chance_of_profit_short                                                     0.611419
        delta                                                                      0.809797
        gamma                                                                      0.005243
        implied_volatility                                                         0.729834
        rho                                                                        0.179158
        theta                                                                     -0.098787
        vega                                                                       0.192707
        high_fill_rate_buy_price                                                  39.831000
        high_fill_rate_sell_price                                                 38.768000
        low_fill_rate_buy_price                                                   38.849000
        low_fill_rate_sell_price                                                  39.750000
        
        :return
        Name: 30, dtype: object
        otm stricke price - 208
        ask price -  2.3 expiration_date - 2022-09-16  Strike_price - 240.0
        
        t = {'cancel_url': 'https://api.robinhood.com/options/orders/62c44c08-6b81-483a-ac56-9160b56650a4/cancel/',
             'canceled_quantity': '0.00000', 'created_at': '2022-07-05T14:34:48.220922Z', 'direction': 'debit',
             'id': '62c44c08-6b81-483a-ac56-9160b56650a4', 'legs': [
                {'executions': [], 'id': '62c44c08-5422-4c3f-8795-b5537cdbda8b',
                 'option': 'https://api.robinhood.com/options/instruments/2370919c-5e90-490a-86cf-8ff589b312bb/',
                 'position_effect': 'open', 'ratio_quantity': 1, 'side': 'buy', 'expiration_date': '2022-09-16',
                 'strike_price': '240.0000', 'option_type': 'call',
                 'long_strategy_code': '2370919c-5e90-490a-86cf-8ff589b312bb_L1',
                 'short_strategy_code': '2370919c-5e90-490a-86cf-8ff589b312bb_S1'}], 'pending_quantity': '1.00000',
             'premium': '230.00000000', 'processed_premium': '0', 'price': '2.30000000',
             'processed_quantity': '0.00000',
             'quantity': '1.00000', 'ref_id': 'c0d563ea-dd6a-4a44-94a5-a7ee99bdf2f8', 'state': 'unconfirmed',
             'time_in_force': 'gfd', 'trigger': 'immediate', 'type': 'limit',
             'updated_at': '2022-07-05T14:34:48.230696Z',
             'chain_id': '32d41003-5e1a-4e6f-96ae-b74b733809f1', 'chain_symbol': 'BNTX', 'response_category': None,
             'opening_strategy': 'long_call', 'closing_strategy': None, 'stop_price': None, 'form_source': None,
             'client_bid_at_submission': None, 'client_ask_at_submission': None, 'client_time_at_submission': None}
        """

