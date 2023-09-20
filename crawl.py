# -*- coding: utf-8 -*-
"""
Created on Sat May  1 16:09:38 2021

@author: crazy
"""
import pandas as pd
from lxml import etree
import os, requests, time, datetime

error_counter = 0

#選取日期，設定URL
def set_web(day01, day02, no_of_broker):
    global url_c
    url_c = "https://histock.tw/stock/brokerprofit.aspx?bno="    
    url_c = url_c + str(no_of_broker) + "&from=" + str(day01) + "&to=" + str(day02)
    print("history url is : " + url_c)
    
def set_today_web(no_of_broker):
    global url_c
    url_c = "https://histock.tw/stock/brokerprofit.aspx?bno="    
    td = time.strftime("%Y%m%d", time.gmtime())
    url_c = url_c + str(no_of_broker) + "&from=" + td + "&to=" + td
    print("today url is : " + url_c)
    
#爬網頁
def get_web():
    
    global error_counter
    my_cookie = {"個資"}
    try:
        headers = {"個資"}
        res = requests.get(url_c, headers = headers, cookies = my_cookie)
        content = res.content.decode()
        global html
        html = etree.HTML(content)
        print("get data")
    
    except:
        print("ERROR")
        error_counter += 1

#從網頁資料抓取表格
def get_table():
    
    global error_counter
    try:
        table = html.xpath("/html/body/form/div[4]/div[3]/div[2]/div[1]/div[1]/div[3]/div/table")
        table = etree.tostring(table[0], encoding = 'utf-8').decode()
        global df
        df = pd.read_html(table, encoding = 'utf-8', header = 0)[0]
        df['net'] = df['買張'] * df['均買'] - df['賣張'] * df['均賣']
        df['percent'] = df['買賣超'] / (df['買張']+df['賣張'])
        df['NP'] = df['percent'] * df['net']
        df = df.sort_values(['net'], ascending = False)
        print("table got")

    except:
        print("ERROR")
        error_counter += 1

#excel處存資料
def create_excel_today():
    df = pd.DataFrame()
    todat_date = datetime.date.today()
    df.to_excel(str(todat_date) + "_" + "today_report.xlsx", index = 0)
    
def create_excel_history():
    df=pd.DataFrame()
    df.to_excel("history_report.xlsx", index = 0)
    
def do_all_history():
    global dff
    global DF
    global writer
    dff=[]
    day01 = input("input from date (yyyymmdd) : ")
    day02 = input("input end date (yyyymmdd) : ")
    for i in range(len(name_list_no)):
        no_b=name_list_no[i]
        set_web(day01,day02,no_b)
        
        get_web()
        get_table()
        
        print("")
        dff.append(df)
        time.sleep(1)
    
    create_excel_history()
    DF = pd.read_excel("history_report.xlsx")
    writer = pd.ExcelWriter("history_report.xlsx")
    
    for i in range(len(name_dict)):
        df_temp = dff[i]
        name = str(name_list_no[i]) + ' ' + str(name_list_name[i])
        df_temp.to_excel(writer, sheet_name = name, index = None, na_rep = 'NaN')
    writer.close()
    
def do_all_today():
    global dff
    global DF
    global writer
    dff=[]

    for i in range(len(name_dict)):
        no_b = name_list_no[i]
        set_today_web(no_b)

        get_web()
        get_table()

        print("")
        dff.append(df)
        time.sleep(1)
    
    create_excel_today()
    DF = pd.read_excel("today_report.xlsx")
    writer = pd.ExcelWriter("today_report.xlsx")
            
    for i in range(len(name_dict)):
        df_temp = dff[i]
        name = str(name_list_no[i]) + ' ' + str(name_list_name[i])
        df_temp.to_excel(writer, sheet_name = name, index = None, na_rep = 'NaN')
    writer.close()
    
'*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*'
'''
#excel重複寫入(先備而不用><) 
from openpyxl import load_workbook

def excelAddSheet(excel_name, excelWriter, sheet_name):
    dataframe = pd.read_excel(excel_name)
    book = load_workbook(excelWriter.path)
    excelWriter.book = book
    dataframe.to_excel(excel_writer = excelWriter, sheet_name = sheet_name)
    excelWriter.close()
    output_dir = cwd
    excelWriter = pd.ExcelWriter(os.path.join(output_dir, 'all1.xlsx'),engine = 'openpyxl')
    pd.DataFrame().to_excel(os.path.join(output_dir, 'all1.xlsx'))
'''
'*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*''*'

def main():
    global url_c, html, df, cwd,name_dict, name_list_no, name_list_name
    
    name_dict = {"關注清單"}
    
    name_list_no = list(name_dict.values())
    name_list_name = list(name_dict.keys())
    
    url_c = ""
    os.chdir(r"C:\Users\crazy\Desktop")
    
    task = input("Today > 1, History > 2: ")
    if task == "1":
        do_all_today()
    elif task == "2":
        do_all_history()
    else:
        print("Error, plz type again...")
    
    if (error_counter != 0):
        print("Total Error Number : " + str(error_counter))
    else:
        print("Done~")
        
if __name__ == '__main__':
    main()
