import numpy as np
import pandas as pd
import tushare as ts
import time

COMMON_ROOT_PATH = "../../data/Common/"
QUOTATION_ROOT_PATH = "../../data/Quotation_side/"
START_DATE = "20140101"
END_DATE = "20181231"

pro = ts.pro_api("4b354a4846eb10e1001d4cc575ac51187aac178861d8fe1759c3c33d")


# 股票日行情
def quotation_data_daily_processor(stock_code):
    try:
        daily_df = pro.daily(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
        daily_df = daily_df.drop_duplicates(["trade_date"], keep="first")
        csv_path = QUOTATION_ROOT_PATH + stock_code[:6] + "_" + stock_code[7:9] + "_" + "quotation.csv"
        daily_df.to_csv(csv_path, index=False, index_label=False)

        # save data record to file
        now_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        quotation_record_file = open(QUOTATION_ROOT_PATH + "Quotation_data_record", "w")
        quotation_record_file.write(stock_code + " " + now_time + " Done" + "\n")
        quotation_record_file.close()
        print(stock_code + " daily quotation data saved")
        time.sleep(2)
    except Exception as e:
        print(e)
        now_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        quotation_record_file = open(QUOTATION_ROOT_PATH + "Quotation_data_record", "w")
        quotation_record_file.write(stock_code + " " + now_time + " Error : " + e + "\n")
        quotation_record_file.close()
        time.sleep(2)


def selected_stock_traverse():
    selected_stock_df = pd.read_csv(COMMON_ROOT_PATH + "selected_stock.csv")
    selected_stock_list = selected_stock_df.loc[:, "ts_code"].tolist()
    for stock_code in selected_stock_list:
        quotation_data_daily_processor(stock_code)


selected_stock_traverse()
print("all data saved.")
