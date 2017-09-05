import urllib.request
import urllib.parse
import json
from pprint import pprint

# YQLをAPI経由で利用します
url = "https://query.yahooapis.com/v1/public/yql"
params = {
    "q": 'select * from yahoo.finance.xchange where pair in ("USDJPY")',
    "format": "json",
    "env": "store://datatables.org/alltableswithkeys"
}
url += "?" + urllib.parse.urlencode(params)
res = urllib.request.urlopen(url)

# 結果はJSON形式で受け取ることができます
result = json.loads(res.read().decode('utf-8'))
pprint(result)
"""
{'query': {'count': 1,
           'created': '2016-08-22T02:57:07Z',
           'lang': 'en-US',
           'results': {'rate': {'Ask': '100.6850',
                                'Bid': '100.6380',
                                'Date': '8/21/2016',
                                'Name': 'USD/JPY',
                                'Rate': '100.6380',
                                'Time': '10:58pm',
                                'id': 'USDJPY'}}}}
"""

# その中から必要な情報（今回はUSD→JPYの為替レート）を取得します
rate = result["query"]["results"]["rate"]["Rate"]
print('USD/JPY:', rate)
# USD/JPY: 100.6380
