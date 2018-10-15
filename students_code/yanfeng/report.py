# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:19:53 2018

@author: yanfeng
"""

import tushare as ts
import os

filename='D:\新桌面/data22.csv'
years=[2016,2017]
for year in years:
    jidu=[1,2,3,4]
    for ji in jidu:
        df=ts.get_report_data(int(year),int(ji))
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=None, encoding='utf_8_sig')
        else:
            df.to_csv(filename, encoding='utf_8_sig')

for j in [1,2,3]:
    df=ts.get_report_data(2018,j)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=None, encoding='utf_8_sig')
    else:
        df.to_csv(filename, encoding='utf_8_sig')