def CoData2( ):

#Collecting Data
    filename='d:/Predicting Stock/StockData2.csv'

#沪深上市公司业绩报告（主表）
    for i2 in [2016,2017]:
        for j2 in [1,2,3,4]:
            Data3=ts.get_report_data(i2,j2)
            if os.path.exists(filename):
                Data3.to_csv(filename,mode='a',header=None)
            else:
                Data3.to_csv(filename)
    for m2 in [1,2,3]:
        Data3=ts.get_report_data(2018,m2)
        if os.path.exists(filename):
            Data3.to_csv(filename,mode='a',header=None)
        else:
            Data3.to_csv(filename)
