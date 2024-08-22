import requests
import hashlib
import time
import random
import json
from googletrans import Translator
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class TranslationTool:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text, src='en', dest='zh-cn'):
        """
        翻译指定的文本。
        :param text: 要翻译的文本字符串。
        :param src: 源语言代码，默认是英文 'en'。
        :param dest: 目标语言代码，默认是中文简体 'zh-cn'。
        :return: 翻译后的文本字符串。
        """
        translated = self.translator.translate(text, src=src, dest=dest)
        return translated.text
    

class YoudaoTranslationTool:
    def __init__(self, app_id, app_secret):
        self.app_id = app_id
        self.app_secret = app_secret
        self.url = "https://openapi.youdao.com/api"
    def generate_sign(self, q, salt):
        sign_str = self.app_id + q + str(salt) + self.app_secret
        sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()
        return sign
    def translate(self, text, src='en', dest='zh-CHS'):
        salt = random.randint(1, 65536)
        sign = self.generate_sign(text, salt)
        params = {
            'q': text,
            'from': src,
            'to': dest,
            'appKey': self.app_id,
            'salt': salt,
            'sign': sign
        }
        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            result = response.json()
            if 'translation' in result:
                return result['translation'][0]
            else:
                return "Translation error: " + result.get('errorCode', 'Unknown error')
        except requests.exceptions.RequestException as e:
            return f"Network error occurred: {e}"
        except Exception as e:
            return f"An error occurred: {e}"

# # 使用示例
# app_id = '553c1ee6d3f9f808'
# app_secret = 'iOiNGdr2OtJrspEciUDOirk6wnKzcmP5'
# translation_tool = YoudaoTranslationTool(app_id, app_secret)
# result = translation_tool.translate("Provide a one-sentence caption for the provided image.")
# print(result)
