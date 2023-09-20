# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:15:02 2023

@author: crazy
"""
import os
import pandas as pd

os.chdir(r"C:\Users\crazy\Desktop\QAQ")
os.chdir(r"C:\Users\crazy\Downloads\TXFF")
dff = pd.read_csv("2023_all_RAW.csv")
dff.head()

dff.Date = dff.Date.astype(str)
dff.Time = dff.Time.astype(str)

data_df = pd.DataFrame({"Date":dff.Date, "Time":dff.Time, "Open":dff.Open, "High":dff.High, "Low":dff.Low, "Close":dff.Close, "Volume":dff.Volume})

# to daily candle
all_days = data_df["Date"].unique()

# loop
big_df = pd.DataFrame({"Open":[], "High":[], "Low":[], "Close":[], "Volume":[]})

for i in range(len(all_days)):
    day01 = data_df[data_df["Date"] == all_days[i]]

    D_day01 = day01.set_index(pd.to_datetime(day01["Date"] + " " + day01["Time"], format = "%Y-%m-%d %H:%M:%S")).drop(["Date", "Time"], axis = 1)
    df_60 = D_day01.resample("15min").agg({"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"})
    big_df = pd.concat([big_df, df_60], axis=0)
    
big_df = big_df.dropna()
print(big_df.index[0])
print(big_df.index[-1])
# done
big_df.to_csv("df15_2023_raw.csv", index = True)


# check nan
df3["Open"].isnull().values.any()




# combine all 4
df1 = pd.read_excel("01030113.xlsx", index_col = 0)
df2 = pd.read_excel("01110118.xlsx", index_col = 0)
df3 = pd.read_excel("01170131.xlsx", index_col = 0)
df4 = pd.read_excel("01300220.xlsx", index_col = 0)
df5 = pd.read_excel("02160301.xlsx", index_col = 0)
df6 = pd.read_excel("02240315.xlsx", index_col = 0)
df7 = pd.read_excel("03140331.xlsx", index_col = 0)
df8 = pd.read_excel("03300417.xlsx", index_col = 0)
df9 = pd.read_excel("04140502.xlsx", index_col = 0)
df10 = pd.read_excel("04280517.xlsx", index_col = 0)
df11 = pd.read_excel("05160531.xlsx", index_col = 0)
df12 = pd.read_excel("05290620.xlsx", index_col = 0)
df13 = pd.read_excel("06160703.xlsx", index_col = 0)
df14 = pd.read_excel("06300718.xlsx", index_col = 0)
df15 = pd.read_excel("07140803.xlsx", index_col = 0)
df16 = pd.read_excel("08020811.xlsx", index_col = 0)

df_all = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16]).drop_duplicates()
df_all.sort_index()
df_all.to_csv("2023_all_RAW.csv", index=True)

#
df01 = pd.read_csv("15_all_RAW.csv")
df02 = pd.read_csv("df15_2023_raw.csv")
df = pd.concat([df01, df02])


# talib
import talib
bb_u = talib.BBANDS(df["Close"], 20, 2)[0].rename("bb_u")
bb_l = talib.BBANDS(df["Close"], 20, 2)[2].rename("bb_l")

K = talib.STOCH(df["High"], df["Low"], df["Close"])[0].rename("K")
D = talib.STOCH(df["High"], df["Low"], df["Close"])[1].rename("D")

macdf = talib.MACD(df["Close"], 12, 26, 9)[0].rename("macdf")
macds = talib.MACD(df["Close"], 12, 26, 9)[1].rename("macds")
macdh = talib.MACD(df["Close"], 12, 26, 9)[2].rename("macdh")

ma1 = talib.SMA(df["Close"], 10).rename("MA10")
ma2 = talib.SMA(df["Close"], 20).rename("MA20")
ma3 = talib.SMA(df["Close"], 60).rename("MA60")

Vma1 = talib.SMA(df["Volume"], 10).rename("VMA10")
Vma2 = talib.SMA(df["Volume"], 20).rename("VMA20")
Vma3 = talib.SMA(df["Volume"], 60).rename("VMA60")

MFI = talib.MFI(df["High"], df["Low"], df["Close"], df["Volume"], 10).rename("MFI")

RP = (abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"])).rename("RP")
Ch = df["Close"].pct_change().rename("Change")

all_df = pd.concat([df, bb_u, bb_l, macdf, macds, macdh, ma1, ma2, ma3, MFI, RP, Ch, Vma1, Vma2, Vma3], axis = 1).dropna().sort_index()

#
all_df.to_csv("REAL_ALL_TA15.csv", index=True)
#
all_df = pd.read_csv("TA.csv", index_col = 0)
#

