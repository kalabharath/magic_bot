import datetime
import logging


format = "%(levelname)s %(asctime)s - %(message)s"
today_1 = datetime.date.today()
# convert datetime to string
today_2 = today_1.strftime("%d%b%y")
logging.basicConfig(filename="screener_"+today_2+".log",
                        filemode="a",
                        format=format)

logger = logging.getLogger()

logger.setLevel(logging.INFO)