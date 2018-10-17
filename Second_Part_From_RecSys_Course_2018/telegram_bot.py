import requests
##in order to receive notification you can contact amborgio_trading_bot on ambrogio_telegram and get your chat id from the following page
#https://api.telegram.org/bot494949207:AAFnbyU_6LSaXqfgVbFK4Qt0UHCLnAbUX1M/getUpdates

class TelegramBot:
    def __init__(self,chat_id):

        self.chat_id = chat_id

    def send_message(self,text):
        print(text)
        message=requests.utils.quote(text)
        requests.get('https://api.telegram.org/bot494949207:AAFnbyU_6LSaXqfgVbFK4Qt0UHCLnAbUX1M/sendMessage?chat_id='+self.chat_id+'&text='+message)


