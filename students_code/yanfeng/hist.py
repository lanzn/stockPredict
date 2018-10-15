# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 00:13:59 2018

@author: yanfeng
"""

import tushare as ts

df=ts.get_hist_data('sh')
df.to_csv('D:\新桌面/data8.csv', encoding='utf_8_sig')     