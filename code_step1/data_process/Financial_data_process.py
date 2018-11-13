import numpy
import pandas
import tushare as ts


FINANCIAL_REPORT_ROOT_PATH = "../../data/600000SH.csv"

pro = ts.pro_api("4b354a4846eb10e1001d4cc575ac51187aac178861d8fe1759c3c33d")


def financial_report_income_processor():
    financial_report_df = pro.income(ts_code='600000.SH', start_date='20180101', end_date='2018100')
    print(financial_report_df)
    financial_report_df.to_csv(FINANCIAL_REPORT_ROOT_PATH)


financial_report_income_processor()
print("data saved")

