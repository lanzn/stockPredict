def CoData6( ):

#Collecting Data
    filename='d:/Predicting Stock/StockData6.csv'


#沪深上市公司偿债能力
    for i7 in [2016,2017]:
        for j7 in [1,2,3,4]:
            Data7=ts.get_debtpaying_data(i7,j7)
            if os.path.exists(filename):
                Data7.to_csv(filename,mode='a',header=None)
            else:
                Data7.to_csv(filename)
    for m7 in [3]:
        Data7=ts.get_debtpaying_data(2018,m7)
        if os.path.exists(filename):
            Data7.to_csv(filename,mode='a',header=None)
        else:
            Data7.to_csv(filename)
