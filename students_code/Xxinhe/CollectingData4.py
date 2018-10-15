def CoData4( ):

#Collecting Data
    filename='d:/Predicting Stock/StockData4.csv'

#沪深上市公司营运能力
    for i4 in [2016,2017]:
        for j4 in [1,2,3,4]:  
            Data5=ts.get_operation_data(i4,j4)
            if os.path.exists(filename):
                Data5.to_csv(filename,mode='a',header=None)
            else:
                Data5.to_csv(filename)
    for m4 in [1,2,3]:
        Data5=ts.get_operation_data(2018,m4)
        if os.path.exists(filename):
            Data5.to_csv(filename,mode='a',header=None)
        else:
            Data5.to_csv(filename)









