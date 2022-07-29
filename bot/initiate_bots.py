import sys, os, time
import random


def pause_execution():
    """
    Pause execution for a few seconds to avoid API rate limit
    :return: boolean
    """
    rand_min = random.randint(1, 10)
    rand_sec = random.randint(1, 60)
    random_milliseconds = random.randint(1, 1000)
    time.sleep(rand_min * 60 + rand_sec+random_milliseconds/1000)
    return True


def run_screener():
    """
    Run the robin screener bot
    :return:
    """

    os.system('python robin_screener_bot.py')
    return True


def run_robin_inference_bot():
    """
    Run the robin inference bot
    :return:
    """

    os.system('python robin_AI_inference_bot.py')
    return True


def run_robin_buy_bot():
    """
    Run the robin buy bot
    :return:
    """
    os.system('python robin_buy_bot.py')
    pause_execution()
    return True


def run_robin_sell_bot():
    """
    Run the robin sell bot
    :return:
    """
    os.system('python robin_sell_bot.py')
    return True


if __name__ == '__main__':
    run_screener()
    run_robin_inference_bot()
    run_robin_buy_bot()
    run_robin_sell_bot()
    sys.exit(0)


