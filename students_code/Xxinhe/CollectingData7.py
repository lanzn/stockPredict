def CoData7( ):

#Collecting Data
    filename='d:/Predicting Stock/StockData7.csv'

#沪深上市公司现金流量
    for i8 in [2016,2017]:
        for j8 in [1,2,3,4]:
            Data8=ts.get_cashflow_data(i8,j8)
            if os.path.exists(filename):
                Data8.to_csv(filename,mode='a',header=None)
            else:
                Data8.to_csv(filename)
    for m8 in[1,2,3]:
        Data8=ts.get_cashflow_data(2018,m8)
        if os.path.exists(filename):
            Data8.to_csv(filename,mode='a',header=None)
        else:
            Data8.to_csv(filename)
