import numpy as np
import pandas as pd
import tushare as ts
import time

COMMON_ROOT_PATH = "../../data/Common/"
FINANCIAL_REPORT_ROOT_PATH = "../../data/Financial_side/"
START_DATE = "20140101"
END_DATE = "20181231"


pro = ts.pro_api("4b354a4846eb10e1001d4cc575ac51187aac178861d8fe1759c3c33d")


# 财报-利润表
def financial_data_income_processor(stock_code):
    income_df = pro.income(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
    income_df = income_df.drop_duplicates(["ann_date", "f_ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "income/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "income.csv"
    income_df.to_csv(csv_path, index=False, index_label=False)


# 财报-资产负债表
def financial_data_balancesheet_df_processor(stock_code):
    balancesheet_df = pro.balancesheet(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
    balancesheet_df = balancesheet_df.drop_duplicates(["ann_date", "f_ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "balancesheet/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "balancesheet.csv"
    balancesheet_df.to_csv(csv_path, index=False, index_label=False)


# 财报-现金流量表
def financial_data_cashflow_processor(stock_code):
    cashflow_df = pro.cashflow(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
    cashflow_df = cashflow_df.drop_duplicates(["ann_date", "f_ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "cashflow/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "cashflow.csv"
    cashflow_df.to_csv(csv_path, index=False, index_label=False)


# 财报-业绩预告
def financial_data_forecast_processor(stock_code):
    forecast_df = pro.forecast(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
    forecast_df = forecast_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "forecast/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "forecast.csv"
    forecast_df.to_csv(csv_path, index=False, index_label=False)


# 财报-业绩快报
def financial_data_express_processor(stock_code):
    express_df = pro.express(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
    express_df = express_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "express/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "express.csv"
    express_df.to_csv(csv_path, index=False, index_label=False)


# 财报-分红送股数据
def financial_data_dividend_processor(stock_code):
    dividend_df = pro.dividend(ts_code=stock_code)
    dividend_df = dividend_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "dividend/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "dividend.csv"
    dividend_df.to_csv(csv_path, index=False, index_label=False)


# 财报-财务指标数据
def financial_data_fina_indicator_processor(stock_code):
    fina_indicator_df = pro.fina_indicator(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
    fina_indicator_df = fina_indicator_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "fina_indicator/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "fina_indicator.csv"
    fina_indicator_df.to_csv(csv_path, index=False, index_label=False)


# 财报-财务审计意见
def financial_data_fina_audit_processor(stock_code):
    fina_audit_df = pro.fina_audit(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
    fina_audit_df = fina_audit_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "fina_audit/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "fina_audit.csv"
    fina_audit_df.to_csv(csv_path, index=False, index_label=False)


# 财报-主营业务构成
def financial_data_fina_mainbz_processor(stock_code):
    fina_mainbz_df = pro.fina_mainbz(ts_code=stock_code, start_date=START_DATE, end_date=END_DATE)
    fina_mainbz_df = fina_mainbz_df.drop_duplicates(["end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + "fina_mainbz/" + stock_code[:6] + "_" + stock_code[7:9] + "_" + "fina_mainbz.csv"
    fina_mainbz_df.to_csv(csv_path, index=False, index_label=False)


def financial_data_processor(stock_code):
    try:
        # financial_data_income_processor(stock_code)
        # financial_data_balancesheet_df_processor(stock_code)
        # financial_data_cashflow_processor(stock_code)
        # financial_data_forecast_processor(stock_code)
        # financial_data_express_processor(stock_code)
        # financial_data_dividend_processor(stock_code)
        financial_data_fina_indicator_processor(stock_code)
        # financial_data_fina_audit_processor(stock_code)
        # financial_data_fina_mainbz_processor(stock_code)

        # save data record to file
        now_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        financial_record_file = open(FINANCIAL_REPORT_ROOT_PATH + "fina_indicator/" + "Financial_data_record", "a")
        financial_record_file.write(stock_code + " " + now_time + " Done" + "\n")
        financial_record_file.close()
        print(stock_code + " financial_data saved")
        time.sleep(1.5)
    except Exception as e:
        print(e)
        now_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        financial_record_file = open(FINANCIAL_REPORT_ROOT_PATH + "fina_indicator/" + "Financial_data_record", "a")
        financial_record_file.write(stock_code + " " + now_time + " Error : " + e + "\n")
        financial_record_file.close()
        time.sleep(1.5)


def financial_data_traverse():
    selected_stock_df = pd.read_csv(COMMON_ROOT_PATH + "selected_stock.csv")
    selected_stock_list = selected_stock_df.loc[:, "ts_code"].tolist()
    # 读取记录文件，实现断点续传
    finished_df = pd.read_csv(FINANCIAL_REPORT_ROOT_PATH + "fina_indicator/" + "Financial_data_record",
                              sep=" ", header=None)
    finished_list = finished_df.loc[(finished_df[2] == "Done")].loc[:, 0].tolist()
    difference_list = list(set(selected_stock_list).difference(set(finished_list)))
    for stock_code in difference_list:
        financial_data_processor(stock_code)


financial_data_traverse()
print("all data saved")
