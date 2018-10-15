def CoData( ):
#Collecting Data
    filename='d:/Predicting Stock/StockData.csv'

#沪深300的股票历史行情
    Data1=ts.get_hist_data('hs300')
    Data1.to_csv(filename,columns=['open','high','low','close'])
