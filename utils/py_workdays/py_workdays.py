import jpholiday
from pytz import timezone
import datetime

import numpy as np
import pandas as pd



def check_jst_datetimes_to_naive(*arg_datetimes):
    """
    ＊*今のところ，ローカルが東京でないnaiveなdatetimeはそのまま通してしまう
    引数のタイムゾーンが同じかどうかチェックし，存在するなら日本であるかチェック
    awareな場合は，naiveに変更
    """
    jst_timezone = timezone("Asia/Tokyo")
    tz_info_set = set([one_datetime.tzinfo for one_datetime in arg_datetimes])
    if len(tz_info_set) > 1:
        raise Exception("timezones are different")
        
    datetimes_tzinfo = list(tz_info_set)[0]
    
    if datetimes_tzinfo is not None:  # 長さが1のはず
        if timezone(str(datetimes_tzinfo)) != jst_timezone:
            raise Exception("timezones must be Asia/Tokyo")
        # naiveなdatetimeに変更
        arg_datetimes = [one_datetime.replace(tzinfo=None) for one_datetime in arg_datetimes]
        
    return tuple(arg_datetimes)


def get_holiday_jp(start_date, end_date, with_name=False):
    """
    期間を指定して祝日を取得
    
    start_date: datetime.date
        開始時刻のdate
    end_datetime: datetime.date
        終了時刻のdate
    eith_name: bool
        休日の名前を出力するかどうか
    to_date: bool
        出力をdatetime.datetimeにするかdatetime.dateにするか
    """
    assert isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date)
    
    holydays_array = np.array(jpholiday.between(start_date, end_date))
    
    if not with_name:
        return holydays_array[:,0].copy()
    
    return holydays_array


class Option():
    """
    オプションの指定のためのクラス
    """
    def __init__(self):
        self._holiday_start_year = datetime.datetime.now().year-5
        self._holiday_end_year = datetime.datetime.now().year
        
        # 利用する休日のarray
        self._holidays_date_array =  get_holiday_jp(start_date=datetime.date(self._holiday_start_year,1,1),
                                                    end_date=datetime.date(self._holiday_end_year,12,31),
                                                    with_name=False,
                                                   )
        # 利用する休日のDatetimeIndex
        self._holidays_datetimeindex = pd.DatetimeIndex(self._holidays_date_array)
    
    @property
    def holiday_start_year(self):
        return self._holiday_start_year
    
    @property
    def holiday_end_year(self):
        return self._holiday_end_year
        
    @holiday_start_year.setter
    def holiday_start_year(self, year):
        self._holiday_start_year = year
        # 利用する休日のarray
        self._holidays_date_array =  get_holiday_jp(start_date=datetime.date(self._holiday_start_year,1,1),
                                                    end_date=datetime.date(self._holiday_end_year,12,31),
                                                    with_name=False,
                                                   )
        # 利用する休日のDatetimeIndex
        self._holidays_datetimeindex = pd.DatetimeIndex(self._holidays_date_array)
        
    @holiday_end_year.setter
    def holiday_end_year(self, year):
        self._holiday_end_year = year
        # 利用する休日のarray
        self._holidays_date_array =  get_holiday_jp(start_date=datetime.date(self._holiday_start_year,1,1),
                                                    end_date=datetime.date(self._holiday_end_year,12,31),
                                                    with_name=False,
                                                   )
        # 利用する休日のDatetimeIndex
        self._holidays_datetimeindex = pd.DatetimeIndex(self._holidays_date_array)
        
    @property
    def holidays_date_array(self):
        return self._holidays_date_array
    
    @property
    def holidays_datetimeindex(self):
        return self._holidays_datetimeindex


# Optionの作成
option = Option()


def get_workdays_jp(start_date, end_date, return_as="date", end_include=False):
    """
    営業日を取得
    
    start_date: datetime.date
        開始時刻のdate
    end_datetime: datetime.date
        終了時刻のdate
    return_as: str, defalt: 'dt'
        返り値の形式
        - 'dt':pd.DatetimeIndex
        - 'date': datetime.date array
    end_include: bool
        最終日も含めて出力するか
    """
    assert isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date)
    # 返り値の形式の指定
    return_as_set = {"dt", "date"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))
    
    # datetime.dateをpd.Timestampに変換(datetime.dateは通常pd.DatetimeIndexと比較できないため)
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)
    
    # 期間中のholidayを取得
    holidays_in_span_index = (start_timestamp<=option.holidays_datetimeindex)&(option.holidays_datetimeindex<end_timestamp)  # DatetimeIndexを使うことに注意
    holidays_in_span_array = option.holidays_date_array[holidays_in_span_index]  # ndarrayを使う

    # 期間中のdatetimeのarrayを取得
    if end_include:
        days_datetimeindex = pd.date_range(start=start_date, end=end_date, freq="D")  # 最終日も含める
    else:
        days_datetimeindex = pd.date_range(start=start_date, end=end_date-datetime.timedelta(days=1), freq="D")  # 最終日は含めない
    
    
    # 休日に含まれないもの，さらに土日に含まれないもののboolインデックスを取得
    holiday_bool_array = np.in1d(days_datetimeindex.date, holidays_in_span_array)  # 休日であるかのブール(pd.DatetimeIndex.isin)でもいい
    sun_or_satur_bool_array = (days_datetimeindex.weekday==5) | (days_datetimeindex.weekday==6)  # 土曜日or日曜日
    
    workdays_bool_array = (~holiday_bool_array)&(~sun_or_satur_bool_array)  # 休日でなく土日でない
    
    workdays_datetimeindex = days_datetimeindex[workdays_bool_array].copy()
    if return_as=="dt":
        return workdays_datetimeindex
    elif return_as=="date":
        return workdays_datetimeindex.date


def get_not_workdays_jp(start_date, end_date, return_as="date", end_include=False):
    """
    非営業日を取得(土日or祝日)
    
    start_date: datetime.date
        開始時刻のdate
    end_datetime: datetime.date
        終了時刻のdate
    return_as: str, defalt: 'dt'
        返り値の形式
        - 'dt':pd.DatetimeIndex
        - 'date': datetime.date array
    end_include: bool
        最終日も含めて出力するか
    """
    assert isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date)
    # 返り値の形式の指定
    return_as_set = {"dt", "date"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))
    
    # datetime.dateをpd.Timestampに変換(datetime.dateは通常pd.DatetimeIndexと比較できないため)
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)
    
    # 期間中のholidayを取得
    holidays_in_span_index = (start_timestamp<=option.holidays_datetimeindex)&(option.holidays_datetimeindex<end_timestamp)  # DatetimeIndexを使うことに注意
    holidays_in_span_array = option.holidays_date_array[holidays_in_span_index]  # ndarrayを使う

    # 期間中のdatetimeのarrayを取得
    if end_include:
        days_datetimeindex = pd.date_range(start=start_date, end=end_date, freq="D")  # 最終日も含める
    else:
        days_datetimeindex = pd.date_range(start=start_date, end=end_date-datetime.timedelta(days=1), freq="D")  # 最終日は含めない
    
    # 休日に含まれないもの，さらに土日に含まれないもののboolインデックスを取得
    holiday_bool_array = np.in1d(days_datetimeindex.date, holidays_in_span_array)  # 休日であるかのブール(pd.DatetimeIndex.isin)でもいい
    sun_or_satur_bool_array = (days_datetimeindex.weekday==5) | (days_datetimeindex.weekday==6)  # 土曜日or日曜日
    
    not_workdays_bool_array = holiday_bool_array | sun_or_satur_bool_array  # 休日あるいは土日
    
    not_workdays_datetimeindex = days_datetimeindex[not_workdays_bool_array].copy()
    if return_as=="dt":
        return not_workdays_datetimeindex
    elif return_as=="date":
        return not_workdays_datetimeindex.date
    

def check_workday_jp(select_date):
    """
    与えられたdatetime.dateが営業日であるかどうかを出力する
    select_date: datetime.date
        入力するdate
    """
    assert isinstance(select_date, datetime.date)
    select_date_array = np.array([select_date])  # データ数が一つのndarray
    # 休日であるかどうか
    is_holiday = np.in1d(select_date_array, option.holidays_date_array).item()  # データ数が一つのため，item
    
    # 土曜日か日曜日であるかどうか
    is_sun_satur = (select_date.weekday()==5) or (select_date.weekday()==6)
    
    is_workday = (not is_holiday) and (not is_sun_satur)
    
    return is_workday


def get_next_workday_jp(select_date, return_as="date", days=1):
    """
    select_date: datetime.date
        指定する日時
    return_as: str, defalt: 'dt'
        返り値の形式
        - 'dt':pd.Timstamp
        - 'datetime': datetime.datetime array
    """
    assert isinstance(select_date, datetime.date)
    # 返り値の形式の指定
    return_as_set = {"dt", "date"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))
        
    def get_next_workday_jp_gen(select_date):
        add_days=1
        while True:
            next_day = select_date + datetime.timedelta(days=add_days)
            if check_workday_jp(next_day):
                yield next_day
            add_days += 1
    
    next_workday_gen = get_next_workday_jp_gen(select_date)
    for i in range(days):
        next_day = next(next_workday_gen)
    
    if return_as=="date":
        return next_day
    elif return_as=="dt":
        return pd.Timestamp(next_day)
        

def extract_workdays_jp_index(dt_index, return_as="index"):
    """
    pd.DatetimeIndexから，営業日のデータのものを抽出
    dt_index: pd.DatetimeIndex
        入力するDatetimeIndex，すでにdatetimeでソートしていることが前提
    return_as: str
        出力データの形式
        - "index": 引数としたdfの対応するインデックスを返す
        - "bool": 引数としたdfに対応するboolインデックスを返す
    """
    # 返り値の形式の指定
    return_as_set = {"index", "bool"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))
        
    # すでにtimestampでソートさてている前提
    start_datetime = dt_index[0].to_pydatetime()
    end_datetime = dt_index[-1].to_pydatetime()
    
    start_datetime, end_datetime = check_jst_datetimes_to_naive(start_datetime, end_datetime)  # 二つのdatetimeのタイムゾーンをチェック・naiveに変更
    
    # 期間内のholidayを取得
    holidays_in_span_index = ((start_datetime-datetime.timedelta(days=1))<option.holidays_datetimeindex)&\
    (option.holidays_datetimeindex<=end_datetime)  # DatetimeIndexを使うことに注意, 当日を含めるため，startから1を引いている．
    holidays_in_span_array = option.holidays_date_array[holidays_in_span_index]  # ndarrayを使う
    
    # 休日に含まれないもの，さらに土日に含まれないもののboolインデックスを取得    
    holiday_bool_array = np.in1d(dt_index.date, holidays_in_span_array)  # 休日
    sun_or_satur_bool_array = (dt_index.weekday==5) | (dt_index.weekday==6)  # 土曜日or日曜日
    
    workdays_bool_array = (~holiday_bool_array)&(~sun_or_satur_bool_array)  # 休日でなく土日でない
    if return_as=="bool":  # boolで返す場合
        return workdays_bool_array
    
    elif return_as=="index":  # indexで返す場合
        workdays_df_indice = dt_index[workdays_bool_array]
        return workdays_df_indice


def extract_workdays_jp(df, return_as="df"):
    """
    データフレームから，営業日のデータのものを抽出．出力データ形式をreturn_asで指定する．
    df: pd.DataFrame(インデックスとしてpd.DatetimeIndex)
        入力データ
    return_as: str
        出力データの形式
        - "df": 抽出した新しいpd.DataFrameを返す
        - "index": 引数としたdfの対応するインデックスを返す
        - "bool": 引数としたdfに対応するboolインデックスを返す
    """
    
    # 返り値の形式の指定
    return_as_set = {"df", "index", "bool"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))
    
    workdays_bool_array = extract_workdays_jp_index(df.index, return_as="bool")
    if return_as=="bool":
        return workdays_bool_array
    
    workdays_df_indice = df.index[workdays_bool_array]
    if return_as=="index":
        return workdays_df_indice

    out_df = df.loc[workdays_df_indice].copy()
    return out_df


def extract_intraday_jp_index(dt_index, return_as="index"):
    """
    pd.DatetimeIndexから，日中(9時から11時半，12時半から15時)のデータのものを抽出．出力データ形式をreturn_asで指定する．
    dt_index: pd.DatetimeIndex
        入力するDatetimeIndex
    return_as: str
        出力データの形式
        - "index": 引数としたdfの対応するインデックスを返す
        - "bool": 引数としたdfに対応するboolインデックスを返す
    """
    
    # 返り値の形式の指定
    return_as_set = {"index", "bool"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))    
  
    bool_array = np.array([False]*len(dt_index))
    
    # 午前をTrueに
    am_indice = dt_index.indexer_between_time(start_time=datetime.time(9,0), end_time=datetime.time(11,30), include_end=False)
    bool_array[am_indice] = True
    
    # 午後をTrueに
    pm_indice = dt_index.indexer_between_time(start_time=datetime.time(12,30), end_time=datetime.time(15,0), include_end=False)
    bool_array[pm_indice] = True
    
    if return_as=="bool":
        return bool_array

    elif return_as=="index":
        intraday_indice = dt_index[bool_array]
        return intraday_indice


def extract_intraday_jp(df, return_as="df"):
    """
    データフレームから，日中(9時から11時半，12時半から15時)のデータのものを抽出．出力データ形式をreturn_asで指定する．
    df: pd.DataFrame(インデックスとしてpd.DatetimeIndex)
        入力データ
    return_as: str
        出力データの形式
        - "df": 抽出した新しいpd.DataFrameを返す
        - "index": 引数としたdfの対応するインデックスを返す
        - "bool": 引数としたdfに対応するboolインデックスを返す
    """
    
    # 返り値の形式の指定
    return_as_set = {"df", "index", "bool"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))    
  
    intraday_bool_array = extract_intraday_jp_index(df.index, return_as="bool")
    if return_as=="bool":
        return intraday_bool_array
    
    intraday_indice = df.index[intraday_bool_array]
    if return_as=="index":
        return intraday_indice
    
    out_df = df.loc[intraday_indice].copy()
    return out_df


def extract_workdays_intraday_jp_index(dt_index, return_as="index"):
    """
    pd.DatetimeIndexから，営業日+日中(9時から11時半，12時半から15時)のデータのものを抽出．出力データ形式をreturn_asで指定する．
    dt_index: pd.DatetimeIndex
        入力するDatetimeIndex
    return_as: str
        出力データの形式
        - "index": 引数としたdfの対応するインデックスを返す
        - "bool": 引数としたdfに対応するboolインデックスを返す
    """

    # 返り値の形式の指定
    return_as_set = {"index", "bool"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))
        
    workday_bool_array = extract_workdays_jp_index(dt_index, return_as="bool")
    intraday_bool_array = extract_intraday_jp_index(dt_index, return_as="bool")
    
    workday_intraday_bool_array = workday_bool_array & intraday_bool_array
    if return_as=="bool":
        return workday_intraday_bool_array
    elif return_as=="index":
        workday_intraday_indice = dt_index[workday_intraday_bool_array]
        return workday_intraday_indice


def extract_workdays_intraday_jp(df, return_as="df"):
    """
    データフレームから，営業日+日中(9時から11時半，12時半から15時)のデータのものを抽出．出力データ形式をreturn_asで指定する．
    df: pd.DataFrame(インデックスとしてpd.DatetimeIndex)
        入力データ
    return_as: str
        出力データの形式
        - "df": 抽出した新しいpd.DataFrameを返す
        - "index": 引数としたdfの対応するインデックスを返す
        - "bool": 引数としたdfに対応するboolインデックスを返す
    """
    
    # 返り値の形式の指定
    return_as_set = {"df", "index", "bool"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))    
       
    workday_intraday_bool_array = extract_workdays_intraday_jp_index(df.index, return_as="bool")
    
    if return_as=="bool":
        return workday_intraday_bool_array
    
    workday_intraday_indice = df.index[workday_intraday_bool_array]
    
    if return_as=="index":
        return workday_intraday_indice
    
    out_df = df.loc[workday_intraday_indice].copy()
    return out_df


if __name__ == "__main__":
    import pickle
    ##########################
    ### get_holiday_jp
    ##########################
    
    start_date = datetime.date(2019, 1, 1)
    end_date = datetime.date(2020, 12, 31)

    holidays = get_holiday_jp(start_date, end_date, with_name=False)
    print(holidays)

    
    ##########################
    ### get_workdays_jp
    ##########################

    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)

    workdays = get_workdays_jp(start_date, end_date, return_as="date", end_include=False)
    print("workdays:",workdays)

    
    ##########################
    ### get_not_workdays_jp
    ##########################

    start_datetime = datetime.date(2019, 1, 1)
    end_datetime = datetime.date(2019, 12, 31)

    not_workdays = get_not_workdays_jp(start_date, end_date, return_as="dt")
    print("not_workdays:",not_workdays)

    
    ##########################
    ### check_workday_jp
    ##########################
    
    select_date = datetime.date(2019, 1, 4)
    print(check_workday_jp(select_date))


    ##########################
    ### get_next_workday_jp
    ##########################

    jst_timezone = timezone("Asia/Tokyo")
    select_datetime = jst_timezone.localize(datetime.datetime(2020, 1, 1, 0, 0, 0))

    next_workday = get_next_workday_jp(select_datetime, days=6, return_as="dt")
    print("next workday",next_workday)

    # データの作成
    with open("aware_stock_df.pickle", "rb") as f:
        aware_stock_df = pickle.load(f)

    print("aware_stock_df",aware_stock_df)

    with open("naive_stock_df.pickle", "rb") as f:
        naive_stock_df = pickle.load(f)

    print("naive_stock_df at 9:00",naive_stock_df.at_time(datetime.time(9,0)))

    ##########################
    ### extract_workdays_jp
    ##########################
    
    extracted_stock_df = extract_workdays_jp(aware_stock_df, return_as="df")
    print("workdays stock_df at 9:00",extracted_stock_df.at_time(datetime.time(9,0)))

    ##########################
    ### extract_intraday_jp
    ##########################

    extracted_df = extract_intraday_jp(naive_stock_df, return_as="df")
    print("intraday stock_df at 8:00",extracted_df.at_time(datetime.time(8,0)))

    ##########################
    ### extract_workdays_intraday_jp
    ##########################
    
    extracted_df = extract_workdays_intraday_jp(aware_stock_df, return_as="df")
    print("workday intraday stock_df at 9:00",extracted_df.at_time(datetime.time(9,0)))
    print("workday intraday stock_df at 8:00",extracted_df.at_time(datetime.time(8,0)))
