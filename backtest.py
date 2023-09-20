#0207ok

from backtesting import Backtest, Strategy
from backtesting.lib import crossover, TrailingStrategy, SignalStrategy, resample_apply

from backtesting.test import SMA
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from datetime import date
from datetime import datetime

import time
import talib
import seaborn as sns
sns.set_theme(style = "darkgrid")

import matplotlib.dates as mdates

dt = datetime.combine(date.today(), datetime.min.time())

# 取得股價
from FinMind.data import DataLoader

# set_time_stamp
def set_time(df):
    for i in range(len(df)):
        df["date"][i] = datetime.strptime(df["date"][i], '%Y-%m-%d').date()
        df["date"][i] = datetime.combine(df["date"][i], datetime.min.time())
    return df

"""
"""
num = "0050"
s_date = "2005-01-02"
e_date = "2023-01-25"

"""
"""
# 下載台股股價資料
def get_stock(num, s_date, e_date):
    dl = DataLoader()
    stock_data = dl.taiwan_stock_daily(stock_id = num, start_date = s_date, end_date = e_date)
    stock_data.rename(columns={'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close'}, inplace=True)
    stock_data.Trading_Volume //= 1000
    stock_data.Trading_money //= 100000000
    
    set_time(stock_data)
    
    stock_data = stock_data.set_index("date")
    stock_data['quarter'] = pd.PeriodIndex(stock_data.index, freq='Q')
    stock_data['quarterc'] = stock_data['quarter'].astype('str')

    stock_data = stock_data.drop(['stock_id'], axis=1)
    
    #margin_short
    df_ms = dl.taiwan_stock_margin_purchase_short_sale(
        stock_id = num,
        start_date = s_date,
        end_date = e_date)
    
    #big three
    df_bt = dl.taiwan_stock_institutional_investors(
        stock_id = num,
        start_date = s_date,
        end_date = e_date)

    df = df_ms.copy().drop(["Note"], axis = 1)
    dff = df_bt.copy()

    #for margin purchase and short sale
    #2010-02-26始有之
    for i in range(len(df)):
        df["date"][i] = datetime.strptime(df["date"][i], '%Y-%m-%d').date()
        df["date"][i] = datetime.combine(df["date"][i], datetime.min.time())
    df = df.set_index("date")
    S_real_margin = (df.MarginPurchaseTodayBalance - df.MarginPurchaseYesterdayBalance).rename("real_margin")
    S_real_short = (df.ShortSaleTodayBalance - df.ShortSaleYesterdayBalance).rename("real_short")
    S_margin_percent = ((df.MarginPurchaseTodayBalance / df.MarginPurchaseLimit)*100).rename("margin_percent")
    S_short_percent = ((df.ShortSaleTodayBalance / df.ShortSaleLimit)*100).rename("short_percent")
    S_MarginPurchaseTodayBalance = df.MarginPurchaseTodayBalance.rename("margin_bal")
    S_ShortSaleTodayBalance = df.ShortSaleTodayBalance.rename("short_bal")

    margin_short_pd = pd.concat([S_real_margin, S_real_short,S_margin_percent, S_short_percent, S_MarginPurchaseTodayBalance, S_ShortSaleTodayBalance], axis = 1)

    # for big rich (ok)
    dff["net_buy"] = (dff.buy - dff.sell)//1000

    p1 = dff[dff.name == "Foreign_Investor"].reset_index(drop = True)
    p2 = dff[dff.name == "Investment_Trust"].reset_index(drop = True)
    p3 = dff[dff.name == "Dealer_self"].reset_index(drop = True)

    p_list = [p1,p2,p3]
    for dfs in p_list:
        set_time(dfs)
        
    p1 = p1.set_index("date")["net_buy"].rename("Fore_net_buy")
    p2 = p2.set_index("date")["net_buy"].rename("Trust_net_buy")
    p3 = p3.set_index("date")["net_buy"].rename("Self_net_buy")
    
    #自2012-05-02才有，自營更自2014-12-01才有之
    big_three_pd = pd.concat([p1, p2, p3], axis = 1)
    big_three_pd = big_three_pd.fillna(0)

    stock_data = stock_data.copy()
    
    stock_data = stock_data.merge(big_three_pd, how='left', on='date')
    stock_data = stock_data.fillna(0)    
    stock_data = stock_data.merge(margin_short_pd, how='left', on='date')
    stock_data = stock_data.fillna(0)     
    
    stock_data["qq_ratio"] = stock_data.short_bal / stock_data.margin_bal

    return stock_data


#股票月營收資料
def basic_data(num, s_date, e_date):
    api = DataLoader()
    # api.login_by_token(api_token='token')
    # api.login(user_id='user_id',password='password')
    df = api.taiwan_stock_month_revenue(
        stock_id = num, start_date = s_date, end_date = e_date)
    df["time_stamp"] = pd.DataFrame([i[0:7] for i in df.date[:]])
    for i in range(len(df["date"])):
        df["date"][i] = datetime.strptime(df["date"][i], '%Y-%m-%d').date()
        df["date"][i] = datetime.combine(df["date"][i], datetime.min.time())

    df = df.set_index(df.date)
    df = df.drop(["date", "stock_id", "country"], axis = 1)
    return df


#顧名思義...
def next_trading_day(df, strr):
    
    #temp_date = df.index[0].strftime("%Y-%m-%d")
    temp = str(strr)
    #year_d = temp[0:4]
    #month_d = temp[5:7]
    #day_d = temp[8:10]
    df["num"] = range(0,len(df))
    
    return df.iloc[df[df.index == temp].num.values + 1]


#標定營收日期
def rev_date(df):
    
    ezdff = df[df.index.day.isin([11,10,9,8,7,6,5,4,3,2,1])].sort_index(ascending=True)
    
    zz = pd.DataFrame(columns = df.columns)
    res = pd.DataFrame(columns = df.columns)
    start_year = df.index[0].strftime("%Y")
    start_month = df.index[0].strftime("%m")
    end_day = df.index[-1].strftime("%d")
    
    for i in range(len(ezdff)-1):
        if ezdff[1:].index[i].day < ezdff[:-1].index[i].day:
            res = pd.concat([zz, pd.DataFrame(ezdff.iloc[i,:]).T])
        else:
            break
    
    new_temp = ezdff.loc[start_year + "-" + str(int(start_month)+1) + "-01":]
    
    for i in range(len(new_temp)-1):
        if ezdff[1:].index[i].day < ezdff[:-1].index[i].day:
            res = pd.concat([res, pd.DataFrame(ezdff.iloc[i,:]).T])
        else:
            pass
    
    if int(end_day) > 10:
        date_list_temp = ezdff.index.to_list()
        res = pd.concat([res, next_trading_day(df, date_list_temp[-1].strftime("%Y-%m-%d"))])
    
    res = res.drop(["quarter", "quarterc"], axis = 1).apply(pd.to_numeric)
    
    return res


def Hi_N(arr: pd.Series, n: int):
    return pd.Series(arr.High).shift(n).rolling(n).max().fillna(0)


def Lo_N(arr: pd.Series, n: int):
    return pd.Series(arr.Low).shift(n).rolling(n).min()

"""
"""

"""
"""

start_t = time.time()

stock_num = "6223"
s_date = "2010-01-02"
e_date = "2023-01-25"

stock_data = get_stock(stock_num, s_date, e_date)
#month_data = basic_data(stock_num, s_date, e_date)
#rev_data = rev_date(stock_data)

end_t = time.time()
print("run_time：%f sec" % (end_t - start_t))

"""
"""
df = stock_data.copy()
aa = Hi_N(df, 60)
"""
"""

class TRYY(SignalStrategy, TrailingStrategy):
    def init(self):
        super().init()
        #vol = self.data.Trading_Volume
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 20)
        self.ma2 = self.I(SMA, price, 60)
        self.h_N = self.I(Hi_N, self.data, 60)
        self.l_N = self.I(Lo_N, self.data, 60)
        self.bbu, self.bbm, self.bbl = self.I(talib.BBANDS, price, 20, 2)
        self.macdf, self.macds, self.macdh = self.I(talib.MACD, price, 12, 26, 9)
        self.rsi = self.I(talib.RSI, price, 14)
        self.K, self.D = self.I(talib.STOCH, self.data.High, self.data.Low, price)
        self.weekly_p = resample_apply("W-FRI", SMA, price, 20)
        
    def next(self):
        super().next()
        mtp = 1
        cprice = self.data.Close[-1]
        if (crossover(self.K, self.D) and (not self.position) and (self.ma1 > self.ma2)):
            self.buy()
            self.set_trailing_sl(mtp)
            
        elif ((crossover(self.D, self.K) or crossover(self.ma2, self.ma1)) and self.position.is_long):
            self.position.close()     
                        
"""     
        elif (crossover(self.l_N, cprice) and (not self.position) and (self.ma1 < self.ma2)):
            self.sell()
            self.set_trailing_sl(mtp)
            
        elif ((crossover(self.ma1, self.ma2)) and self.position.is_short):
            self.position.close()
"""            

ct = Backtest(stock_data, TRYY, commission = 0.00585, exclusive_orders = True)
statc = ct.run()
sns.set(rc = {'figure.figsize':(14,10)})
res = sns.lineplot(data = statc['_equity_curve'].Equity).set(title = stock_num)

#所以說，TRYY是Backtest Class中的strategy 參數(名稱)
#run plot method 就是Backtest可以使用的方法(method)
#
#Order Class，以Strategy.buy() and Strategy.sell()處理之
#
#Trade Class
#
np.corrcoef(stock_data.Close ,stock_data.qq_ratio)

"""   
    def next(self):
        super().next()
        
        cprice = self.data.Close[-1]
        if (crossover(self.bbu, cprice) and (not self.position)):
            self.buy(sl = cprice*0.85)
        elif crossover(cprice, self.bbu):
            self.position.close()  
"""

class newss(SignalStrategy, TrailingStrategy):
    def init(self):
        super().init()
        vol = self.data.Volume
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 20)
        self.ma2 = self.I(SMA, price, 60)
        self.vma1 = self.I(SMA, vol, 20)
        self.vma2 = self.I(SMA, vol, 60)
        self.bbu, self.bbm, self.bbl = self.I(talib.BBANDS, price, 20, 2)
        self.macdf, self.macds, self.macdh = self.I(talib.MACD, price, 12, 26, 9)
        self.rsi = self.I(talib.RSI, price, 14)
        self.K, self.D = self.I(talib.STOCH, self.data.High, self.data.Low, price)
            
    def next(self):
        super().next()
        mtp = 2
        #cprice = self.data.Close[-1]
        if (crossover(self.macdh, 0) and (not self.position) and (self.ma1 > self.ma2)):
            self.buy()
            self.set_trailing_sl(mtp)
            
        elif ((crossover(self.ma2, self.ma1)) and self.position.is_long):
            self.position.close()     
     
        elif (crossover(0, self.macdh) and (not self.position) and (self.ma1 < self.ma2)):
            self.sell()
            self.set_trailing_sl(mtp)
            
        elif ((crossover(self.ma1, self.ma2)) and self.position.is_short):
            self.position.close()
            
dt = Backtest(stock_data, newss, commission = 0.00585, exclusive_orders = True, cash = 1000000)
statd = dt.run()
statd
sns.set(rc = {'figure.figsize':(14,10)})
res = sns.lineplot(data = statd['_equity_curve'].Equity).set(title = stock_num)




"""
Backtest.__init__
#statc.__getattr__
#statc.__init__


#plot on strategies C / quarterly
result_stock = statc['_equity_curve']

result_stock['quarter'] = pd.PeriodIndex(result_stock.index, freq='Q')
result_stock['quarterc'] = result_stock['quarter'].astype('string')
result_stock = result_stock.set_index("quarterc")

f1 = plt.figure("Quarterly",figsize=(14,10))
ax = plt.subplot(1,1,1)
plt.plot(result_stock.Equity)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))

dayloc = mdates.MonthLocator(bymonth=(3,6,9,12))
ax.xaxis.set_major_locator(dayloc)

plt.plot(statc['_equity_curve'].Equity)

plt.plot_date(stock_data.iloc[:,-1], statc['_equity_curve'].Equity)
plt.plot(stock_data.iloc[:,-1])

"""


#try in vextorize the signal trading
signal = (stock_data.Close >= stock_data.Open).astype(int).diff().fillna(0)
signal_t = signal.replace(-1, 0)  # Upwards/long only

tt = (stock_data.Close >= stock_data.Open).astype(int)




VAR = talib.VAR(df.Close, timeperiod=5, nbdev=1)
LINEARREG_INTERCEPT = talib.LINEARREG_INTERCEPT(df.Close, timeperiod=14)



