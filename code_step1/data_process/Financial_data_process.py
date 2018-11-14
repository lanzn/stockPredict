import numpy as np
import pandas as pd
import tushare as ts


FINANCIAL_REPORT_ROOT_PATH = "../../data/Financial_side/sample/"
START_DATE = "20140101"
END_DATE = "20181231"
STOCK_CODE = "600000.SH"


pro = ts.pro_api("4b354a4846eb10e1001d4cc575ac51187aac178861d8fe1759c3c33d")


# 财报-利润表
def financial_data_income_processor():
    income_df = pro.income(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    income_df = income_df.drop_duplicates(["ann_date", "f_ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "income.csv"
    income_df.to_csv(csv_path, index=False, index_label=False)


# 财报-资产负债表
def financial_data_balancesheet_df_processor():
    balancesheet_df = pro.balancesheet(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    balancesheet_df = balancesheet_df.drop_duplicates(["ann_date", "f_ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "balancesheet.csv"
    balancesheet_df.to_csv(csv_path, index=False, index_label=False)


# 财报-现金流量表
def financial_data_cashflow_processor():
    cashflow_df = pro.cashflow(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    cashflow_df = cashflow_df.drop_duplicates(["ann_date", "f_ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "cashflow.csv"
    cashflow_df.to_csv(csv_path, index=False, index_label=False)


# 财报-业绩预告
def financial_data_forecast_processor():
    forecast_df = pro.forecast(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    forecast_df = forecast_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "forecast.csv"
    forecast_df.to_csv(csv_path, index=False, index_label=False)


# 财报-业绩快报
def financial_data_express_processor():
    express_df = pro.express(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    express_df = express_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "express.csv"
    express_df.to_csv(csv_path, index=False, index_label=False)


# 财报-分红送股数据
def financial_data_dividend_processor():
    dividend_df = pro.dividend(ts_code=STOCK_CODE)
    dividend_df = dividend_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "dividend.csv"
    dividend_df.to_csv(csv_path, index=False, index_label=False)


# 财报-财务指标数据
def financial_data_fina_indicator_processor():
    fina_indicator_df = pro.fina_indicator(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    fina_indicator_df = fina_indicator_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "fina_indicator.csv"
    fina_indicator_df.to_csv(csv_path, index=False, index_label=False)


# 财报-财务审计意见
def financial_data_fina_audit_processor():
    fina_audit_df = pro.fina_audit(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    fina_audit_df = fina_audit_df.drop_duplicates(["ann_date", "end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "fina_audit.csv"
    fina_audit_df.to_csv(csv_path, index=False, index_label=False)


# 财报-主营业务构成
def financial_data_fina_mainbz_processor():
    fina_mainbz_df = pro.fina_mainbz(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    fina_mainbz_df = fina_mainbz_df.drop_duplicates(["end_date"], keep="first")
    csv_path = FINANCIAL_REPORT_ROOT_PATH + STOCK_CODE[:6] + "_" + STOCK_CODE[7:9] + "_" + "fina_mainbz.csv"
    fina_mainbz_df.to_csv(csv_path, index=False, index_label=False)


financial_data_income_processor()
financial_data_balancesheet_df_processor()
financial_data_cashflow_processor()
financial_data_forecast_processor()
financial_data_express_processor()
financial_data_dividend_processor()
financial_data_fina_indicator_processor()
financial_data_fina_audit_processor()
financial_data_fina_mainbz_processor()
print("all data saved")

