import datetime
import numpy as np
import pandas as pd
import tushare as ts
import time


COMMON_ROOT_PATH = "../../data/Common/"
QUOTATION_ROOT_PATH = "../../data/Quotation_side/"
FINANCIAL_REPORT_ROOT_PATH = "../../data/Financial_side/"
TRAIN_TEST_ROOT_PATH = "../../data/Train&Test/"
TRAIN_USE_SEASON_NUM = 4


def training_data_creator(stock_code):
    fina_indicator_csv_name = stock_code[:6] + "_" + stock_code[7:9] + "_" + "fina_indicator.csv"
    quotation_csv_name = stock_code[:6] + "_" + stock_code[7:9] + "_" + "quotation.csv"
    fina_indicator_df = pd.read_csv(FINANCIAL_REPORT_ROOT_PATH + "fina_indicator/" + fina_indicator_csv_name)
    daily_df = pd.read_csv(QUOTATION_ROOT_PATH + quotation_csv_name)
    # 获取财报日期列表
    fina_indicator_date_list = fina_indicator_df.loc[:, "ann_date"].tolist()
    selected_daily_df = pd.DataFrame()
    for fina_indicator_date in fina_indicator_date_list:
        flag = 0
        back_days = 0
        # 字符串类型日期转datetime类型
        fina_indicator_date_origin = datetime.datetime.strptime(str(fina_indicator_date), "%Y%m%d")
        now_date = fina_indicator_date
        while flag == 0:
            if back_days > 30:
                # 如果向前回溯一个月都找不到行情，则认为这段时间股票处于停盘状态，删除该研报，并继续。
                fina_indicator_df = fina_indicator_df.loc[~(fina_indicator_df["ann_date"] == fina_indicator_date)]
                break

            one_day_daily_df = daily_df.loc[daily_df["trade_date"] == int(now_date)]
            if one_day_daily_df.empty:
                # 如果财报发布当日找不到行情数据，则向前不断查找，直到找到
                fina_indicator_date_origin = fina_indicator_date_origin - datetime.timedelta(days=1)
                back_days = back_days + 1
                now_date = fina_indicator_date_origin.strftime("%Y%m%d %H:%M:%S")[:8]
            else:
                flag = 1
                # 将新找到的行情数据添加到dataframe里
                if selected_daily_df.empty:
                    selected_daily_df = one_day_daily_df
                else:
                    selected_daily_df = selected_daily_df.append(one_day_daily_df)

    # 将selected_daily_df的收盘价列取出来转型成dataframe，然后重置index并不保留index，然后成为fina_indicator_df的新列。
    if fina_indicator_df.empty or selected_daily_df.empty:
        return None
    else:
        try:
            fina_indicator_df = fina_indicator_df.reset_index(drop=True)
            selected_daily_df = selected_daily_df["close"].to_frame().reset_index(drop=True)
            fina_indicator_df = pd.concat([fina_indicator_df, selected_daily_df], axis=1)
        except Exception as e:
            print(e)
            return None

    # 滑动窗口得到新的样本集Dataframe并返回
    index_count = fina_indicator_df.shape[0]
    start_index = 4
    train_df = pd.DataFrame()
    while start_index < index_count:
        # 每次循环得到一条样本，每次先判断N个季度是否连续，若不连续则continue
        end_date_list = []
        for season in range(TRAIN_USE_SEASON_NUM):
            end_date_list.append(fina_indicator_df.ix[start_index - season, "end_date"])

        # 检查季度是否连续，continue_flag为0表示连续，1为不连续
        continue_flag = 0
        for index, end_date in enumerate(end_date_list):
            if index + 1 < TRAIN_USE_SEASON_NUM:
                end_date_pre = datetime.datetime.strptime(str(end_date), "%Y%m%d")
                end_date_next = datetime.datetime.strptime(str(end_date_list[index + 1]), "%Y%m%d")
                days_delta = (end_date_next - end_date_pre).days
                if days_delta >= 95:
                    continue_flag = 1
                    break

        # 当前窗口中N个季度若连续，则执行拼接，否则跳过该窗口
        if continue_flag == 0:
            season = 0
            one_line_df = pd.DataFrame()
            # 先计算好label
            next_close = fina_indicator_df.loc[start_index - TRAIN_USE_SEASON_NUM, "close"]
            pre_close = fina_indicator_df.loc[start_index - TRAIN_USE_SEASON_NUM + 1, "close"]
            if next_close >= pre_close:
                label = 1
            else:
                label = 0
            while season < TRAIN_USE_SEASON_NUM:
                if one_line_df.empty:
                    one_line_df = fina_indicator_df.ix[start_index - season, :].to_frame().T
                    one_line_df.rename(columns=lambda x: x + "_" + str(season + 1), inplace=True)
                    # ts_code列名字改回来
                    one_line_df.rename(columns={'ts_code_1': 'ts_code'}, inplace=True)
                    one_line_df = one_line_df.reset_index(drop=True)
                else:
                    temp_df = fina_indicator_df.ix[start_index - season, :].to_frame().T
                    del temp_df['ts_code']
                    temp_df.rename(columns=lambda x: x + "_" + str(season + 1), inplace=True)
                    temp_df = temp_df.reset_index(drop=True)
                    one_line_df = pd.concat([one_line_df, temp_df], axis=1)
                season = season + 1
            # 把label拼上
            one_line_df["label"] = label
            if train_df.empty:
                train_df = one_line_df
            else:
                train_df = train_df.append(one_line_df)
        start_index = start_index + 1
    train_df.reset_index(drop=True)
    return train_df


def selected_stock_traverse():
    selected_stock_df = pd.read_csv(COMMON_ROOT_PATH + "selected_stock.csv")
    selected_stock_list = selected_stock_df.loc[:, "ts_code"].tolist()
    full_train_df = pd.DataFrame()

    # 读取断点续传记录文件
    train_test_record_file = open(TRAIN_TEST_ROOT_PATH + "Train&Test_data_record", "r")
    lines = train_test_record_file.readlines()
    finished_stock_list = []
    for line in lines:
        finished_stock_list.append(str(line)[:9])
    train_test_record_file.close()

    # print(selected_stock_list)
    for stock_code in selected_stock_list:
        if stock_code not in finished_stock_list:
            print("now " + str(stock_code) + " ......")
            # 针对每支股票读取财报数据和对应的日期的行情数据，拼接后，根据滑动窗口得到一系列样本
            one_train_df = training_data_creator(stock_code)
            if one_train_df is None:
                print(stock_code + "error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue
            else:
                # 将Dataframe拼接到大样本中
                if full_train_df.empty:
                    full_train_df = one_train_df
                else:
                    full_train_df = full_train_df.append(one_train_df)
                one_train_df.to_csv(TRAIN_TEST_ROOT_PATH + str(stock_code) + ".csv", index=False, index_label=False)
                print(str(stock_code) + " data process OK!")
                train_test_record_file = open(TRAIN_TEST_ROOT_PATH + "Train&Test_data_record", "a")
                train_test_record_file.write(str(stock_code) + " data process OK!" + "\n")
                train_test_record_file.close()
        else:
            print(stock_code + " has finished,continue...")


def train_data_concator():
    selected_stock_df = pd.read_csv(COMMON_ROOT_PATH + "selected_stock.csv")
    selected_stock_list = selected_stock_df.loc[:, "ts_code"].tolist()
    full_train_df = pd.DataFrame()
    for stock_code in selected_stock_list:
        try:
            one_stock_df = pd.read_csv(TRAIN_TEST_ROOT_PATH + str(stock_code) + ".csv")
            if one_stock_df.empty:
                print(stock_code + ".csv is empty!")
                continue
            else:
                if full_train_df.empty:
                    full_train_df = one_stock_df
                else:
                    full_train_df = full_train_df.append(one_stock_df)
                print(stock_code + ".csv concat ok!")
        except Exception as e:
            print(stock_code + " error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
            continue
    full_train_df.to_csv(TRAIN_TEST_ROOT_PATH + "full_train_set.csv", index=False, index_label=False)


selected_stock_traverse()
train_data_concator()
print("data all ok!")
