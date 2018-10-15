def CoData1( ):

#Collecting Data
    filename='d:/Predicting Stock/StockData1.csv'

#沪深上市公司基本情况
    Data2=ts.get_stock_basics()
    Data2.to_csv(filename)

