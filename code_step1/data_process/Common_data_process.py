import numpy as np
import pandas as pd
import tushare as ts


COMMON_ROOT_PATH = "../../data/Common/"

pro = ts.pro_api("4b354a4846eb10e1001d4cc575ac51187aac178861d8fe1759c3c33d")


# 财报-利润表
def common_stock_basic_processor():
    stock_basic_list_df = pro.stock_basic(list_status="L", fields="ts_code,symbol,name,area,industry,fullname,enname,"
                                                                  "market,exchange,curr_type,list_status,list_date,"
                                                                  "delist_date,is_hs")
    stock_basic_delist_df = pro.stock_basic(list_status="D", fields="ts_code,symbol,name,area,industry,fullname,enname,"
                                                                    "market,exchange,curr_type,list_status,list_date,"
                                                                    "delist_date,is_hs")
    stock_basic_pause_df = pro.stock_basic(list_status="P", fields="ts_code,symbol,name,area,industry,fullname,enname,"
                                                                   "market,exchange,curr_type,list_status,list_date,"
                                                                   "delist_date,is_hs")
    csv_path_list = COMMON_ROOT_PATH + "stock_basic_list.csv"
    csv_path_delist = COMMON_ROOT_PATH + "stock_basic_delist.csv"
    csv_path_pause = COMMON_ROOT_PATH + "stock_basic_pause.csv"
    stock_basic_list_df.to_csv(csv_path_list, index=False, index_label=False)
    stock_basic_delist_df.to_csv(csv_path_delist, index=False, index_label=False)
    stock_basic_pause_df.to_csv(csv_path_pause, index=False, index_label=False)


common_stock_basic_processor()
print("all data saved")

