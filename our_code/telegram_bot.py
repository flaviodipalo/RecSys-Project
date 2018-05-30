import requests

#https://api.telegram.org/bot494949207:AAFnbyU_6LSaXqfgVbFK4Qt0UHCLnAbUX1M/getUpdates

class TelegramBot:
    def __init__(self,chat_id):

        self.chat_id = chat_id

    def send_message(self,text):
        print(text)
        message=requests.utils.quote(text)
        requests.get('https://api.telegram.org/bot494949207:AAFnbyU_6LSaXqfgVbFK4Qt0UHCLnAbUX1M/sendMessage?chat_id='+self.chat_id+'&text='+message)
