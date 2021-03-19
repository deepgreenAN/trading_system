import requests
from pathlib import Path
import pandas as pd
import datetime
import time
import io
import jpholiday
import numpy as np


def make_source_with_naikaku(source_path):
    """
    内閣府のサイトから得られるデータ(https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv)
    を適切なフォーマット(header, indexの無い日にち，祝日名)に変換して引数のソースディレクトリに保存する関数．
    振替休日は休日と表記される．1955年からのデータであることに注意
    """
    url = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"
    req = requests.get(url)
    csv_bin = io.BytesIO(req.content)
    my_parser = lambda date: datetime.datetime.strptime(date, "%Y/%m/%d")
    holiday_df_csv = pd.read_csv(csv_bin,
                             names=["date", "holiday_name"],
                             index_col="date",
                             parse_dates=True,
                             date_parser=my_parser,
                             header=0,
                             encoding="shift-jis"
                            )
    
    holiday_df_csv.to_csv(source_path, header=False)


def make_source_with_api(source_path):
    """
    日本の祝日API(https://holidays-jp.github.io/)を用いて取得したデータを
    適切なフォーマット(header, indexの無い日にち，祝日名)に変換して引数のソースディレクトリに保存する関数．
    振替休日は振替元も同時に表示される．2015年からの祝日であることに注意
    """
    date_holidayname = {"date":[], "holiday_name":[]}

    for year in range(2015, datetime.datetime.now().year+1):
        url = "https://holidays-jp.github.io/api/v1/{}/date.json".format(year)
        req = requests.get(url)
        date_dict = req.json()

        date_holidayname["date"].extend(list(date_dict.keys()))
        date_holidayname["holiday_name"].extend(list(date_dict.values()))
        time.sleep(0.1)  # スクレイピングでサーバーの負荷を軽減するため

    holiday_df_api = pd.DataFrame(date_holidayname)
    holiday_df_api.to_csv(source_path, index=False, header=False)


def make_source_with_jpholiday(source_path):
    """
    jpholiday(https://pypi.org/project/jpholiday/)を用いて取得したデータを
    適切なフォーマット(header, indexの無い日にち，祝日名)に変換して引数のソースディレクトリに保存する関数．
    振替休日は振替元も同時に表示される．1955年からの祝日であることに注意
    """ 
    start_date = datetime.date(1955,1,1)
    end_date = datetime.date(datetime.datetime.now().year+1,12,31)
    holidays_array = np.array(jpholiday.between(start_date, end_date))
    
    holiday_df_jpholiday = pd.DataFrame(holidays_array[:,1,None], 
                          columns=["holiday_name"],
                          index=pd.DatetimeIndex(holidays_array[:,0])
                         )
    holiday_df_jpholiday.index.name = "date"
    holiday_df_jpholiday.to_csv(source_path, header=False)

if __name__ == "__main__":
    naikaku_source_path = Path("../source/holiday_naikaku.csv")
    make_source_with_naikaku(naikaku_source_path)
    api_source_path = Path("../source/holiday_api.csv")
    make_source_with_api(api_source_path)
    jpholiday_source_path = Path("../source/holiday_jpholiday.csv")
    make_source_with_jpholiday(jpholiday_source_path)