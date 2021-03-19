import jpholiday
import workdays
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


def get_holiday_jp(start_datetime, end_datetime, with_name=False, to_date=True):
    """
    期間を指定して祝日を取得．遅いので，一度しか呼ばないようにするべき．
    
    start_datetime: datetime.datetime
        開始時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    end_datetime: datetime.datetime
        終了時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    eith_name: bool
        休日の名前を出力するかどうか
    to_date: bool
        出力をdatetime.datetimeにするかdatetime.dateにするか
    """
    start_datetime, end_datetime = check_jst_datetimes_to_naive(start_datetime, end_datetime)
    
    if to_date:
        start_datetime = start_datetime.date()
        end_datetime = end_datetime.date()
    
    holydays_array = np.array(jpholiday.between(start_datetime, end_datetime))
    
    if not with_name:
        return holydays_array[:,0].copy()
    
    return holydays_array


holiday_start_year = 2015  # 利用する休日の開始年
holiday_end_year = 2020  # 利用する休日の終了年（含める）

# 利用する休日のndarray
holidays_date_array = get_holiday_jp(start_datetime=datetime.datetime(holiday_start_year,1,1,0,0,0),
                                     end_datetime=datetime.datetime(holiday_end_year,12,31,23,59,59),
                                     with_name=False,
                                    )
#print(holidays_date_array[:10])

# 利用する休日のpd.DatetimeIndex
holidays_datetimeindex = pd.DatetimeIndex(holidays_date_array)
#print(holidays_datetimeindex)


def get_workdays_jp(start_datetime, end_datetime, return_as="dt", end_include=False):
    """
    営業日を取得
    
    start_datetime: datetime.datetime
        開始時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    end_datetime: datetime.datetime
        終了時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    return_as: str, defalt: 'dt'
        返り値の形式
        - 'dt':pd.DatetimeIndex
        - 'date': datetime.date array
        - 'datetime': datetime.datetime array 
    end_include: bool
        最終日も含めて出力するか
    """
    # 返り値の形式の指定
    return_as_set = {"dt", "datetime", "date"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))
    
    start_datetime, end_datetime = check_jst_datetimes_to_naive(start_datetime, end_datetime)
    
    # 期間中のholidayを取得
    holidays_in_span_index = (start_datetime<=holidays_datetimeindex)&(holidays_datetimeindex<end_datetime)  # DatetimeIndexを使うことに注意
    holidays_in_span_array = holidays_date_array[holidays_in_span_index]  # ndarrayを使う

    # 期間中のdatetimeのarrayを取得
    if end_include:
        days_datetimeindex = pd.date_range(start=start_datetime, end=end_datetime, freq="D")  # 最終日も含める
    else:
        days_datetimeindex = pd.date_range(start=start_datetime, end=end_datetime-datetime.timedelta(days=1), freq="D")  # 最終日は含めない
    
    
    # 休日に含まれないもの，さらに土日に含まれないもののboolインデックスを取得
    holiday_bool_array = np.in1d(days_datetimeindex.date, holidays_in_span_array)  # 休日であるかのブール(pd.DatetimeIndex.isin)でもいい
    sun_or_satur_bool_array = (days_datetimeindex.weekday==5) | (days_datetimeindex.weekday==6)  # 土曜日or日曜日
    
    workdays_bool_array = (~holiday_bool_array)&(~sun_or_satur_bool_array)  # 休日でなく土日でない
    
    workdays_datetimeindex = days_datetimeindex[workdays_bool_array].copy()
    if return_as=="dt":
        return workdays_datetimeindex
    elif return_as=="date":
        return workdays_datetimeindex.date
    elif return_as=="datetime":
        workdays_array = workdays_datetimeindex.to_pydatetime()
        return workdays_array


def get_not_workdays_jp(start_datetime, end_datetime, return_as="dt", end_include=False):
    """
    非営業日を取得(土日or祝日)
    
    start_datetime: datetime.datetime
        開始時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    end_datetime: datetime.datetime
        囚虜時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    return_as: str, defalt: 'dt'
        返り値の形式
        - 'dt':pd.DatetimeIndex
        - 'date': datetime.date array
        - 'datetime': datetime.datetime array
    end_include: bool
        最終日も含めて出力するか
    """
    start_datetime, end_datetime = check_jst_datetimes_to_naive(start_datetime, end_datetime)
    
    # 期間中のholidayを取得
    holidays_in_span_index = (start_datetime<=holidays_datetimeindex)&(holidays_datetimeindex<end_datetime)  # DatetimeIndexを使うことに注意
    holidays_in_span_array = holidays_date_array[holidays_in_span_index]  # ndarrayを使う

    # 期間中のdatetimeのarrayを取得
    if end_include:
        days_datetimeindex = pd.date_range(start=start_datetime, end=end_datetime, freq="D")  # 最終日も含める
    else:
        days_datetimeindex = pd.date_range(start=start_datetime, end=end_datetime-datetime.timedelta(days=1), freq="D")  # 最終日は含めない
    
    # 休日に含まれないもの，さらに土日に含まれないもののboolインデックスを取得
    holiday_bool_array = np.in1d(days_datetimeindex.date, holidays_in_span_array)  # 休日であるかのブール(pd.DatetimeIndex.isin)でもいい
    sun_or_satur_bool_array = (days_datetimeindex.weekday==5) | (days_datetimeindex.weekday==6)  # 土曜日or日曜日
    
    not_workdays_bool_array = holiday_bool_array | sun_or_satur_bool_array  # 休日あるいは土日
    
    not_workdays_datetimeindex = days_datetimeindex[not_workdays_bool_array].copy()
    if return_as=="dt":
        return not_workdays_datetimeindex
    elif return_as=="date":
        return not_workdays_datetimeindex.date
    elif return_as=="datetime":
        not_workdays_array = not_workdays_datetimeindex.to_pydatetime()        
        return not_workdays_array
    

def check_workday_jp(select_datetime):
    """
    与えられたdatetime.datetimeが営業日であるかどうかを出力する
    select_datetime: datetime.datetime
        入力するdatetime
    """
    select_datetime_array = np.array([select_datetime.date()])  # データ数が一つのndarray
    # 休日であるかどうか
    is_holiday = np.in1d(select_datetime_array, holidays_date_array).item()  # データ数が一つのため，item
    
    # 土曜日か日曜日であるかどうか
    is_sun_satur = (select_datetime.weekday()==5) or (select_datetime.weekday()==6)
    
    is_workday = (not is_holiday) and (not is_sun_satur)
    
    return is_workday


def get_next_workday_jp(select_datetime, return_as="datetime", days=1):
    """
    select_datetime: datetime.datetime
        指定する日時
    return_as: str, defalt: 'dt'
        返り値の形式
        - 'dt':pd.Timstamp
        - 'date': datetime.date array
        - 'datetime': datetime.datetime array
    """
    # 返り値の形式の指定
    return_as_set = {"dt", "datetime", "date"}
    if not return_as in return_as_set:
        raise Exception("return_as must be any in {}".format(return_as_set))
        
    def get_next_workday_jp_gen(select_datetime):
        add_days=1
        while True:
            next_day = select_datetime + datetime.timedelta(days=add_days)
            if check_workday_jp(next_day):
                yield next_day
            add_days += 1
    
    next_workday_gen = get_next_workday_jp_gen(select_datetime)
    for i in range(days):
        next_day = next(next_workday_gen)
    
    if return_as=="datetime":
        return next_day
    elif return_as=="date":
        return next_day.date()
    elif return_as=="dt":
        return pd.Timestamp(next_day)
        

def extract_workdays_jp_index(dt_index, return_as="index"):
    """
    pd.DatetimeIndexから，営業日のデータのものを抽出
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
        
    # すでにtimestampでソートさてている前提
    start_datetime = dt_index[0].to_pydatetime()
    end_datetime = dt_index[-1].to_pydatetime()
    
    start_datetime, end_datetime = check_jst_datetimes_to_naive(start_datetime, end_datetime)
    
    # 期間内のholidayを取得
    holidays_in_span_index = ((start_datetime-datetime.timedelta(days=1))<holidays_datetimeindex)&\
    (holidays_datetimeindex<=end_datetime)  # DatetimeIndexを使うことに注意, 当日を含めるため，startから1を引いている．
    holidays_in_span_array = holidays_date_array[holidays_in_span_index]  # ndarrayを使う
    
    # 休日に含まれないもの，さらに土日に含まれないもののboolインデックスを取得    
    holiday_bool_array = np.in1d(dt_index.date, holidays_in_span_array)  # 休日であるかのブール(pd.DatetimeIndex.isin)でもいい
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

    ##########################
    ### get_holiday_jp
    ##########################
    
    start_datetime = datetime.datetime(2019, 1, 1, 0, 0, 0)
    end_datetime = datetime.datetime(2020, 12, 31, 0, 0, 0)

    #jst_timezone = timezone("Asia/Tokyo")
    #start_datetime = jst_timezone.localize(datetime.datetime(2019, 1, 1, 0, 0, 0))
    #end_datetime = jst_timezone.localize(datetime.datetime(2020, 12, 31, 0, 0, 0))

    holydays = get_holiday_jp(start_datetime, end_datetime, with_name=False, to_date=True)
    print("holydays:",holydays)

    
    ##########################
    ### get_workdays_jp
    ##########################

    #start_datetime = datetime.datetime(2019, 1, 1, 0, 0, 0)
    #end_datetime = datetime.datetime(2019, 12, 31, 0, 0, 0)

    jst_timezone = timezone("Asia/Tokyo")
    start_datetime = jst_timezone.localize(datetime.datetime(2020, 1, 1, 0, 0, 0))
    end_datetime = jst_timezone.localize(datetime.datetime(2020, 12, 31, 0, 0, 0))

    workdays = get_workdays_jp(start_datetime, end_datetime)
    print("workdays:",workdays)

    
    ##########################
    ### get_not_workdays_jp
    ##########################

    #start_datetime = datetime.datetime(2019, 1, 1, 0, 0, 0)
    #end_datetime = datetime.datetime(2019, 12, 31, 0, 0, 0)

    jst_timezone = timezone("Asia/Tokyo")
    start_datetime = jst_timezone.localize(datetime.datetime(2020, 1, 1, 0, 0, 0))
    end_datetime = jst_timezone.localize(datetime.datetime(2020, 12, 31, 0, 0, 0))

    not_workdays = get_not_workdays_jp(start_datetime, end_datetime, to_date=False)
    print("not_workdays:",not_workdays)