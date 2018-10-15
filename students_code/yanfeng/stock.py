# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 12:36:52 2018

@author: yanfeng
"""

import tushare as ts
df=ts.get_stock_basics()
df.to_csv('D:\新桌面/data1.csv', encoding='utf_8_sig')