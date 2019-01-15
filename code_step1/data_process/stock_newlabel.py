#coding:utf-8
#label重定义，不再是单纯的涨跌，而是超额收益的正负，超额收益为正，则为正样本，说明是好股票，超额收益为负，则为负样本
#超额收益=策略收益-基准收益
#基准收益选择沪深300每个季度的收益百分比
#策略收益是买入持有策略每个季度的收益，买入持有就是某天买入后，不做任何操作，到某个时间再全部卖出

import pandas as pd
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

COMMON_PATH="../../data/Common/selected_stock.csv"
BASE_STOCK_PATH="../../data/Quotation_side/399300_SZ_quotation.csv"
STOCK_PATH="../../data/Quotation_side/"
NEW_PATH="../../data/Common/New_Label/"
NEW_PATH2="../../data/Common/New_Label2/"
DATE_LIST = ["20140331", "20140630", "20140930", "20141231",
             "20150331", "20150630", "20150930", "20151231",
             "20160331", "20160630", "20160930", "20161231",
             "20170331", "20170630", "20170930", "20171231",
             "20180331", "20180630", "20180930"]
print(DATE_LIST[0][:-2])
base_stock_df=pd.read_csv(BASE_STOCK_PATH)#沪深300的所有行情
def select_date(base_stock_df):
    #选出季度日期，只算开盘时间，没开盘的就往前推一天直到找到
    #找出201806开头的第一个
    df1={}
    for index,row in base_stock_df.iterrows():
        for d in DATE_LIST[::-1]:
            if d[:-2] in str(row["trade_date"]):
                if d[:-2] not in df1.keys():
                    df1[d[:-2]]=row["trade_date"]
    return sorted(df1.values(),reverse=True)
Select_Date_list=select_date(base_stock_df)

def select(df,li):
    #根据季度日期选出df
    re=[]
    for each in li:
        for index,row in df.iterrows():
            if row["trade_date"]==each:
                re.append(row)
    return pd.DataFrame(re).reset_index(drop=True)

base_jidu_df=select(base_stock_df,Select_Date_list)#沪深300的季度df
print(base_jidu_df[base_jidu_df["trade_date"]==20180928]["close"])
common_stock=list(pd.read_csv(COMMON_PATH)["ts_code"])
# print(base_jidu_df.ix[0])

def jidudate(a,A):#按照交易日期得到相应的季度日期
    for i in A:
        if a[:-2]==i[:-2]:
            return i

def xianglin(a,b,D):#判断两个季度是否相邻，不相邻则删除这条数据
    s1=0
    s2=0
    for d in D:
        if d[:-2]==a[:-2]:
            s1=D.index(d)
        if d[:-2]==b[:-2]:
            s2=D.index(d)
    if s1-s2>1:
        return False
    return True

# #处理，根据沪深300每个季度实际开盘日期去筛选每只股票
# for stock in common_stock:
#     path=STOCK_PATH+stock[:-3]+"_"+stock[-2:]+"_quotation.csv"
#     stock_df=pd.read_csv(path)
#     stock_df=select(stock_df,Select_Date_list)
#     df=[]
#     for i in range(len(stock_df)-1):
#         each_l=[]
#         each_l.append(stock_df.ix[i]["ts_code"])#ts_code
#         each_l.append(jidudate(str(stock_df.ix[i]["trade_date"]),DATE_LIST))#季度日期
#         each_l.append(stock_df.ix[i]["trade_date"])#股票交易日期1
#         each_l.append(stock_df.ix[i+1]["trade_date"])#股票交易日期2
#         each_l.append(stock_df.ix[i]["trade_date"])#基准股交易日期1
#         each_l.append(stock_df.ix[i+1]["trade_date"])#基准股交易日期2
#         each_l.append((stock_df.ix[i]["close"]-stock_df.ix[i+1]["close"])/stock_df.ix[i+1]["close"])#股票季度收益百分比
#         each_l.append((list(base_jidu_df[base_jidu_df["trade_date"]==each_l[2]]["close"])[0]-list(base_jidu_df[base_jidu_df["trade_date"]==each_l[3]]["close"])[0])/list(base_jidu_df[base_jidu_df["trade_date"]==each_l[3]]["close"])[0])#基准季度收益百分比
#         if each_l[6]>each_l[7]:
#             each_l.append(1)
#         else:
#             each_l.append(0)
#         if xianglin(str(each_l[2]),str(each_l[3]),DATE_LIST):
#             df.append(each_l)
#         #判断四个交易日期是否一样。不一样的直接删除
#         # if each_l[2] == each_l[4] and each_l[3] == each_l[5]:
#         #
#     df=pd.DataFrame(df,columns=["ts_code","jidu_date","s_date_1","s_date_2","b_date_1","b_date_2","s_pct_change","b_pct_change","label"])
#     df.to_csv(NEW_PATH+stock[:-3]+"_"+stock[-2:]+".csv",index=False)
#     #df["label"]=df.applymap(lambda x:1 if x["s_pct_change"]>x["b_pct_change"] else 0)
#     print(df)
#
#
#处理，算每只股票每个季度真正开盘日期，base的交易日期要和个股一致
for stock in common_stock:
    path=STOCK_PATH+stock[:-3]+"_"+stock[-2:]+"_quotation.csv"
    stock_df=pd.read_csv(path)#个股行情
    stock_real_date_list=select_date(stock_df)#选出个股每个季度真正开盘的日期
    stock_df=select(stock_df,stock_real_date_list)#选出每个季度个股的df
    df=[]
    for i in range(len(stock_df)-1):
        each_l=[]
        each_l.append(stock_df.ix[i]["ts_code"])#ts_code
        each_l.append(jidudate(str(stock_df.ix[i]["trade_date"]),DATE_LIST))#季度日期
        each_l.append(stock_df.ix[i]["trade_date"])#股票交易日期1
        each_l.append(stock_df.ix[i+1]["trade_date"])#股票交易日期2
        each_l.append(stock_df.ix[i]["trade_date"])#基准股交易日期1
        each_l.append(stock_df.ix[i+1]["trade_date"])#基准股交易日期2
        each_l.append((stock_df.ix[i]["close"]-stock_df.ix[i+1]["close"])/stock_df.ix[i+1]["close"])#股票季度收益百分比
        each_l.append((list(base_stock_df[base_stock_df["trade_date"]==each_l[2]]["close"])[0]-list(base_stock_df[base_stock_df["trade_date"]==each_l[3]]["close"])[0])/list(base_stock_df[base_stock_df["trade_date"]==each_l[3]]["close"])[0])#基准季度收益百分比
        #####################################################
        #新加4列收益值,为了计算预测的收益率
        each_l.append(stock_df.ix[i]["close"])#s_date_1的close
        each_l.append(stock_df.ix[i+1]["close"])  # s_date_2的close
        each_l.append(stock_df.ix[i]["close"] - stock_df.ix[i + 1]["close"])#股票季度收益值
        each_l.append(list(base_stock_df[base_stock_df["trade_date"] == each_l[2]]["close"])[0] -list(base_stock_df[base_stock_df["trade_date"] == each_l[3]]["close"])[0])#基准季度收益值
        if each_l[6]>each_l[7]:
            each_l.append(1)
        else:
            each_l.append(0)
        if xianglin(str(each_l[2]),str(each_l[3]),DATE_LIST):
            df.append(each_l)
        #判断四个交易日期是否一样。不一样的直接删除
        # if each_l[2] == each_l[4] and each_l[3] == each_l[5]:
        #
    df=pd.DataFrame(df,columns=["ts_code","jidu_date","s_date_1","s_date_2","b_date_1","b_date_2","s_pct_change","b_pct_change","s_date_1_close","s_date_2_close","s_change","b_change","label"])
    df.to_csv(NEW_PATH2+stock[:-3]+"_"+stock[-2:]+".csv",index=False)
    #df["label"]=df.applymap(lambda x:1 if x["s_pct_change"]>x["b_pct_change"] else 0)
    print(df)

#测试部分
# stock="603085_SH"
# path=STOCK_PATH+stock[:-3]+"_"+stock[-2:]+"_quotation.csv"
# stock_df=pd.read_csv(path)#个股行情
# stock_real_date_list=select_date(stock_df)#选出个股每个季度真正开盘的日期
# stock_df=select(stock_df,stock_real_date_list)#选出每个季度个股的df
# df=[]
# for i in range(len(stock_df)-1):
#     each_l=[]
#     each_l.append(stock_df.ix[i]["ts_code"])#ts_code
#     each_l.append(jidudate(str(stock_df.ix[i]["trade_date"]),DATE_LIST))#季度日期
#     each_l.append(stock_df.ix[i]["trade_date"])#股票交易日期1
#     each_l.append(stock_df.ix[i+1]["trade_date"])#股票交易日期2
#     each_l.append(stock_df.ix[i]["trade_date"])#基准股交易日期1
#     each_l.append(stock_df.ix[i+1]["trade_date"])#基准股交易日期2
#     each_l.append((stock_df.ix[i]["close"]-stock_df.ix[i+1]["close"])/stock_df.ix[i+1]["close"])#股票季度收益百分比
#     each_l.append((list(base_stock_df[base_stock_df["trade_date"]==each_l[2]]["close"])[0]-list(base_stock_df[base_stock_df["trade_date"]==each_l[3]]["close"])[0])/list(base_stock_df[base_stock_df["trade_date"]==each_l[3]]["close"])[0])#基准季度收益百分比
#     if each_l[6]>each_l[7]:
#         each_l.append(1)
#     else:
#         each_l.append(0)
#     if xianglin(str(each_l[2]),str(each_l[3]),DATE_LIST):
#         df.append(each_l)
#         #判断四个交易日期是否一样。不一样的直接删除
#         # if each_l[2] == each_l[4] and each_l[3] == each_l[5]:
#         #
# df=pd.DataFrame(df,columns=["ts_code","jidu_date","s_date_1","s_date_2","b_date_1","b_date_2","s_pct_change","b_pct_change","label"])
# df.to_csv(NEW_PATH2+stock[:-3]+"_"+stock[-2:]+".csv",index=False)
#     #df["label"]=df.applymap(lambda x:1 if x["s_pct_change"]>x["b_pct_change"] else 0)
# print(df)




