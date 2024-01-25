import requests, math, datetime, time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

loc_dt = datetime.datetime.today()
time_del = datetime.timedelta(days=1) 
prev_days = 45
start_date = loc_dt - datetime.timedelta(days = prev_days)
ticker = "^TWII"

def get_daily_quotes_MXF(date):

    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    myobj_MXF = {'queryDate': date, "commodityId":"MXF"}
    response_MXF = requests.post(url, data = myobj_MXF)
    soup_MXF = BeautifulSoup(response_MXF.text,features="html.parser")
    table_MXF = soup_MXF.find("table",class_="table_f")
    try:
        MXF = table_MXF.find_all("tr")[-1].find_all("td")
        MXF = [r.text.strip() for r in MXF]
        MXF = [int(i.replace(",",""))for i in MXF]
        MXF = [MXF[6],MXF[8]]
      
        return MXF #[big3_long_OI, big3_short_OI]
    except:
        return ['None']*2

def get_daily_quotes_MTX(date):

    url = "https://www.taifex.com.tw/cht/3/futDailyMarketReport"
    myobj_MTX = {'queryDate': date, "MarketCode":'0',"commodity_id":"MTX"}
    response_MTX = requests.post(url, data = myobj_MTX)
    soup_MTX = BeautifulSoup(response_MTX.text,features="html.parser")
    table_MTX = soup_MTX.find("table",class_="table_f")
    try:
        
        MTX = table_MTX.find_all("tr")[3].find_all("td")
        MTX = [r.text.strip() for r in MTX]
        MTX = int(MTX[-5])
      
        return MTX #MTX_OI_overall
    except:
        return ['None']

#retrieve data
def re_data():
        
    date_list=[]
    up_list=[]
    down_list=[]
    total_io =[]
    
    print("Starting from : \n")
    
    for i in range(prev_days+1):
        temp_dt = start_date + datetime.timedelta(days = i) 
        temp_dt = temp_dt.strftime("%Y/%m/%d")
        print(temp_dt)
        
        if get_daily_quotes_MXF(temp_dt)[0] != "None":
            up_list.append(get_daily_quotes_MXF(temp_dt)[0])
            down_list.append(get_daily_quotes_MXF(temp_dt)[1])
            total_io.append(get_daily_quotes_MTX(temp_dt))
            date_list.append(temp_dt)
            time.sleep(abs(np.random.normal(0,0.3,1))[0])
        else:
            pass
    
        
    np.set_printoptions(2, suppress=True)
    
    real_up = np.asarray(total_io) - np.asarray(up_list)
    real_down = np.asarray(total_io) - np.asarray(down_list)
    real_net = real_up - real_down
    real_percent = np.around(100 * real_net / np.asarray(total_io),2)
    
    df = np.asarray(yf.download(ticker , start_date)["Close"])
    
    real_df = pd.DataFrame(data = {"date":date_list,"real_up":real_up, "real_down":real_down,"real_net":real_net, "real_percent":real_percent,"TWI":df})
    real_df["date"] = pd.to_datetime(real_df["date"], format = '%Y/%m/%d')
    real_df["date"] = real_df["date"].dt.strftime("%Y/%m/%d")
    
    return real_df

def draw_graph(real_df):
    
    #original
    #Create combo chart
    import seaborn as sns
    sns.set_theme(style = "dark")
    
    font_size = 28
    fig, ax1 = plt.subplots(figsize=(14,8))
    color = 'tab:green'
    ax1.set_title('MTX vs TWI' + "\n" + str(real_df["date"].iloc[0]) + "_to_" + str(real_df["date"].iloc[-1]), fontsize=font_size)
    cols = ['red' if x < 0 else 'green' for x in real_df.real_net]
    
    ax1 = sns.barplot(x=real_df["date"], y=real_df["real_net"], data = real_df, palette=cols)
    ax1.set_ylabel('MTX_OI', fontsize=font_size, color=color)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    color = 'tab:blue'
    
    ax2.set_ylabel('Taiwan Index', fontsize=font_size, color=color)
    ax2 = sns.lineplot(x=real_df.index, y=real_df.TWI, data = real_df, sort=False, color=color)
    ax2.tick_params(axis='y', color=color)
    
    new_ticks = [i.get_text() for i in ax1.get_xticklabels()]
    tempt_step = math.floor(prev_days/10)
    plt.xticks(range(0, len(new_ticks), tempt_step), new_ticks[::tempt_step])
    plt.show()
    
def main():
    real_df = re_data()
    draw_graph(real_df)
    
main()
