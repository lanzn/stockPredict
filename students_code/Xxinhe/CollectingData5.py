def CoData5( ):

#Collecting Data
    filename='d:/Predicting Stock/StockData5.csv'

#沪深上市公司成长能力
    for i6 in [2016,2017]:
        for j6 in [1,2,3,4]:
            Data6=ts.get_growth_data(i6,j6)
            if os.path.exists(filename):
                Data6.to_csv(filename,mode='a',header=None)
            else:
                Data6.to_csv(filename)
    for m6 in [1,2,3]:
        Data6=ts.get_growth_data(2018,m6)
        if os.path.exists(filename):
            Data6.to_csv(filename,mode='a',header=None)
        else:
            Data6.to_csv(filename)
