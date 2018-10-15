def CoData3( ):

#Collecting Data
    filename='d:/Predicting Stock/StockData3.csv'

#沪深上市公司盈利能力
    for i3 in [2016,2017]:
        for j3 in [1,2,3,4]:  
            Data4=ts.get_profit_data(i3,j3)
            if os.path.exists(filename):
                Data4.to_csv(filename,mode='a',header=None)
            else:
                Data4.to_csv(filename)
    for m3 in [1,2,3]:
        Data4=ts.get_profit_data(2018,m3)
        if os.path.exists(filename):
            Data4.to_csv(filename,mode='a',header=None)
        else:
            Data4.to_csv(filename)




