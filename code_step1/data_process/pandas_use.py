# 将某一列转型，改变该列的dtype值。
# all_stock_df["market"] = all_stock_df["market"].astype("int64")

# Dataframe.loc[] 是按标签选取行列。
# 选取所有market列值为"主板"且list_status列值为"L"的行
# selected_stock_df = all_stock_df.loc[(all_stock_df["market"] == "主板") & (all_stock_df["list_status"] == "L")]

# fina_indicator_date_list是一个list，该方法能把列值为列表里的值的行都筛选出来
# selected_daily_df = daily_df.loc[daily_df["trade_date"].isin(fina_indicator_date_list)]

# 删除某列值等于某值的行，~是取反的意思
# fina_indicator_df = fina_indicator_df.loc[~(fina_indicator_df["ann_date"] == fina_indicator_date)]
