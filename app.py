from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *


#======這裡是呼叫的檔案內容=====

#======python的函數庫==========
import tempfile, os
import datetime
import lxml
#======python的函數庫==========
import pandas as pd
from imgurpython import ImgurClient
import pyimgur

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#======python的函數庫==========

client_id = 'fb19aaf7c71e5c3'
client_secret = '363e33f42283898dd24892bdd1abb48cbcb0075b'
access_token = "0d7c6e53ab21eaaa5cce482a44bcad4191730ae1"
refresh_token = "f0a7cac532f15f5415ef124a1ebeebd64cf3086f"
album = "yCuOV6e"


client = ImgurClient(client_id, client_secret, access_token, refresh_token)

date = str(datetime.date.today())

app = Flask(__name__)

static_tmp_path = (r".\static\tmp")
local_img_file = "test.png"
PATH = static_tmp_path + "\\" + local_img_file

# Channel Access Token
line_bot_api = LineBotApi('bM6YG00kfdy0Auz8GI1kyFnC6T1YM+0VD+woVlujtCvAI5gsTTOn9h7rD0rBEzkMkItAfC75giwrwCeH8iPuloIc47fiAW/VjzcX42i5S360XrAsx3c5S+qIVDOM2qQ3uhysuPLApajTfrg75mS5hQdB04t89/1O/w1cDnyilFU=')
# Channel Secret
handler = WebhookHandler('38966518b4c18cefb88f9c85dde3fe74')


# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])

def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

def upload(client_data, PATH, album , name = 'test-name!' ,title = 'test-title' ):
    config = {
        'album':  album,
        'name': name,
        'title': title,
        'description': ''
    }

    print("Uploading image... ")
    image = client_data.upload_from_path(PATH, config=config, anon=False)
    print("Done")

    return image

def Dealt_stock():
    message = ImageSendMessage(
                    original_content_url='https://i.imgur.com/O5A2slq.png',
                    preview_image_url='https://i.imgur.com/O5A2slq.png'
                )                
    return message

def last_try():

    date_tse = date.replace('-','')
    str1 = 'https://www.twse.com.tw/announcement/punish?response=html&startDate='
    str2 = '&endDate='
    str3 = '&stockNo=&sortKind=DATE&querytype=3&selectType=&proceType=&remarkType='
    tse = pd.read_html(str1 + date_tse + str2 + date_tse + str3)[0]

    drop_index = []
    for row in range(len(tse)):
        code = str( tse['證券代號'][row] )
        if len(code) > 4 :
            drop_index.append(row)
    tse.drop(index = drop_index, columns = ['編號','處置內容', '處置條件', '累計'], inplace = True)
    tse.drop_duplicates(subset = ['證券代號'], keep = 'first', inplace = True)
    tse[['證券代號']] = tse[['證券代號']].astype(str)
    # print(tse.columns)
    tse.columns = ['公布日期', '證券代號', '證券名稱', '處置期間', '處置措施']

    date_otcT = date.replace('-','/')
    date_otc = date_otcT.replace(date_otcT[:4], str( int(date_otcT[:4]) - 1911 ) )
    str1 = 'https://www.tpex.org.tw/web/bulletin/disposal_information/disposal_information_print.php?l=zh-tw&sd='
    str2 = '&ed='
    str3 = '&code=&choice_type=all_category&stk_cotegory=-1&disposal_measure=-1&group_type=group_date'
    otc = pd.read_html(str1 + date_otc + str2 + date_otc + str3)[0]

    drop_index = []
    pun = []
    for row in range(len(otc)):
        code = str( otc['證券代號'][row] )
        if len(code) > 4 :
            drop_index.append(row)
        five = otc['處置內容'][row].find('每5分鐘')
        ten = otc['處置內容'][row].find('每10分鐘')
        if five >= 0 or ten >= 0 : pun.append('第一次處置')
        else: pun.append('第二次處置')
    otc['處置措施'] = pun
    otc.drop(index = drop_index, columns = ['編號', '處置原因', '累計','處置內容'], inplace = True)# 
    otc.drop_duplicates(subset = ['證券代號'], keep = 'first', inplace = True)
    # print(otc.columns)
    otc.columns = ['公布日期', '證券代號', '證券名稱', '處置期間', '處置措施']

    # 最後整理
    df = pd.DataFrame()        
    df = pd.concat([tse, otc])
    df.sort_values(by = ['公布日期'], ascending = True, inplace = True)
    df.reset_index(drop = True, inplace = True)

    df.sort_values(by = ['公布日期'], ascending = True, inplace = True)
    df.reset_index(drop = True, inplace = True)


    plt.style.use('ggplot')
    plt.figure("test", figsize = (10,4), dpi = 200)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    ax = plt.axes(frame_on=False)# 不要額外框線
    ax.xaxis.set_visible(False)  # 隱藏X軸刻度線
    ax.yaxis.set_visible(False)  # 隱藏Y軸刻度線
    plt.title("處置股資料" + "\n" + date)
    pd.plotting.table(ax, df, loc='center')

    plt.savefig(PATH)

    im = pyimgur.Imgur(client_id)
    uploaded_image = im.upload_image(PATH)

    message = uploaded_image.link
    return message

def Dealt_stock_GG():
    message = ImageSendMessage(
                    original_content_url=str(last_try()),
                    preview_image_url=str(last_try())
                )                
    return message

def Dealt_stock_G():

    message = TextSendMessage(text=last_try())              
    return message

def help_text():
    message = TextSendMessage(text="目前處置股僅平日可用" + "\n" + "輸入 : 處置股" + "\n" + "輸入 : cat")
    return message


# 處理訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text
    if '最新合作廠商' in msg:
        message = imagemap_message()
        line_bot_api.reply_message(event.reply_token, message)
    elif '最新活動訊息' in msg:
        message = buttons_message()
        line_bot_api.reply_message(event.reply_token, message)
    elif '註冊會員' in msg:
        message = Confirm_Template()
        line_bot_api.reply_message(event.reply_token, message)
    elif '旋轉木馬' in msg:
        message = Carousel_Template()
        line_bot_api.reply_message(event.reply_token, message)
    elif '圖片畫廊' in msg:
        message = test()
        line_bot_api.reply_message(event.reply_token, message)
    elif '功能列表' in msg:
        message = function_list()
        line_bot_api.reply_message(event.reply_token, message)
    elif event.message.text == "cat":
        message = Dealt_stock()
        line_bot_api.reply_message(event.reply_token, message)
    elif event.message.text == "處置股link":
        message = Dealt_stock_G()
        line_bot_api.reply_message(event.reply_token, message)
    elif event.message.text == "處置股":
        message = Dealt_stock_GG()
        line_bot_api.reply_message(event.reply_token, message)

    elif event.message.text == "help":
        message = help_text()
        line_bot_api.reply_message(event.reply_token, message)
    else:
        message = TextSendMessage(text=msg)
        line_bot_api.reply_message(event.reply_token, message)

@handler.add(PostbackEvent)
def handle_message(event):
    print(event.postback.data)


@handler.add(MemberJoinedEvent)
def welcome(event):
    uid = event.joined.members[0].user_id
    gid = event.source.group_id
    profile = line_bot_api.get_group_member_profile(gid, uid)
    name = profile.display_name
    message = TextSendMessage(text = f'{name} 歡迎加入' + "\n" + "輸入 help 看更多><" )
    line_bot_api.reply_message(event.reply_token, message)
        
        
import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
