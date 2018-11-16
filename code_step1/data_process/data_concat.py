# TODO:读取所有股票，读取该股票的n种财报数据，连接到一起，然后根据对应日期的收盘价算出是涨是跌
# TODO:按照时间窗口滑动来构造训练集，包括将股票代码进行onehot编码。

import numpy as np
import pandas as pd
import tushare as ts
import time

COMMON_ROOT_PATH = "../../data/Common/"
QUOTATION_ROOT_PATH = "../../data/Quotation_side/"
FINANCIAL_REPORT_ROOT_PATH = "../../data/Financial_side/"


def training_data_creator(stock_code):
    fina_indicator_csv_name = stock_code[:6] + "_" + stock_code[7:9] + "_" + "fina_indicator.csv"
    quotation_csv_name = stock_code[:6] + "_" + stock_code[7:9] + "_" + "quotation.csv"
    fina_indicator_df = pd.read_csv(FINANCIAL_REPORT_ROOT_PATH + "fina_indicator/" + fina_indicator_csv_name)
    daily_df = pd.read_csv(QUOTATION_ROOT_PATH + quotation_csv_name)
    # 获取财报日期列表
    # 根据日期列表获取对应日期的行情
    # 将对应位置拼接形成新的Dataframe
    # 滑动窗口得到新的样本集Dataframe并返回
    return None


def selected_stock_traverse():
    selected_stock_df = pd.read_csv(COMMON_ROOT_PATH + "selected_stock.csv")
    selected_stock_list = selected_stock_df.loc[:, "ts_code"].tolist()
    for stock_code in selected_stock_list:
        # 针对每支股票读取财报数据和对应的日期的行情数据，拼接后，根据滑动窗口得到一系列样本
        xxxxxdf = training_data_creator(stock_code)
        # 将Dataframe拼接到大样本中


selected_stock_traverse()
