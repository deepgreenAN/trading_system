import pandas as pd
import datetime
from datetime import timedelta
from pytz import timezone
import pickle
import warnings

from py_workdays import option, get_near_workday_intraday_jp, check_workday_intraday_jp
from py_workdays import get_next_border_workday_intraday_jp, get_previous_border_workday_intraday_jp

# 自分が利用する足
available_sample_type = set(["S", "1S", "30S", "T", "1T", "5T", "10T", "30T", "H","1H", "2H", "4H", 
                             "12H", "D", "1D", "B", "1B", "W", "1W", "2W", "M", "1M", "Q", "1Q", "Y", "1Y"])

middle_type_dict ={"S":"S","1S":"S","30S":"30S","min":"T","1min":"T","T":"T","1T":"T","5T":"5T","10T":"10T","30T":"30T",
                   "H":"H","1H":"H","2H":"2H","4H":"4H","12H":"12H","D":"D","1D":"D",
                   "B":"B","1B":"B","W":"W","1W":"W","2W":"2W","M":"M","1M":"M","Q":"Q","1Q":"Q",
                   "Y":"Y","1Y":"Y"
                   }


# frequencyとtimedeltaの辞書Sから2Wまで
type_deltataime_dict = {
    "S":timedelta(seconds=1),"30S":timedelta(seconds=30),"T":timedelta(minutes=1),"5T":timedelta(minutes=5),"10T":timedelta(minutes=10),
    "30T":timedelta(minutes=30),"H":timedelta(hours=1),"2H":timedelta(hours=2),"4H":timedelta(hours=4),"12H":timedelta(hours=12),
    "D":timedelta(days=1),"W":timedelta(days=7),"2W":timedelta(days=14)
}


seconds_freq = {type_deltataime_dict[i].total_seconds():i for i in type_deltataime_dict.keys()}

freq_seconds = {i:type_deltataime_dict[i].total_seconds() for i in type_deltataime_dict.keys()}


def middle_sample_type_with_check(freq):
    """
    与えられたfreq_strが使えるものかどうか判断し，pandasで利用可能なfreq_strを返す．
    freq: str
        frequencyを意味する文字列
    """
    if freq not in available_sample_type:
        raise ValueError("This freq is invalid")

    return middle_type_dict[freq]

def get_freq_from_sec(seconds):
    """
    指定した秒数から，pandasで指定されるfrequency strを出力する．ただし，"S"から"2W"まで
    seconds: int
        frequencyを求めたい整数
    """
    if seconds not in seconds_freq:
        raise ValueError("This seconds is invalid")

    return seconds_freq[seconds]

def get_sec_from_freq(freq):
    """
    指定したfrequency strから，秒数を出力する関数．ただし"S"から"2W"まで
    freq: str
        秒数を求めたいfrequency str
    """
    if freq not in freq_seconds:
        raise ValueError("This freq is invalid")
    return freq_seconds[freq]


def get_df_freq(df):
    """
    pandasのdfのfreqを出力する関数
    """

    if df.index.freqstr is not None:
        return df.index.freqstr
        
    if len(df.index) < 2:
        raise ValueError("cannot caolculate frequency of dataframe")
        
    first_datetime = df.index[0].to_pydatetime()  # datetime.datetimeに変換
    second_datetime = df.index[1].to_pydatetime()  # datetime.datetimeに変換

    diff_seconds = second_datetime.timestamp() - first_datetime.timestamp()  # unixに変換
    return get_freq_from_sec(diff_seconds)


def get_floor_mod_datetime(select_datetime, freq_str, will_warn=False):
    """
    指定した日時を指定する周期に応じて切り捨てる
    select_datetime: datetime.datetime
        指定日時
    freq_str: freq_str
        サンプリング周期に対応する文字列
    """
    if get_sec_from_freq(freq_str)  > get_sec_from_freq("D"):
        raise Exception("This function for frequency lowwer then day")
    sec_from_freq = get_sec_from_freq(freq_str)
    select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか

    if select_is_aware:
        timezone_datetime = select_datetime
    else:
        utc_timezone = timezone("UTC")  # これはどこでもよい
        timezone_datetime = utc_timezone.localize(select_datetime)
        
    select_time_unix =  timezone_datetime.timestamp() + timezone_datetime.utcoffset().total_seconds() 
    select_time_mod = int(select_time_unix) % int(sec_from_freq)
    if select_time_mod != 0 and will_warn:  # 余りが出てしまった場合
        warnings.warn("input datetime is floord")
        
    return select_datetime - timedelta(seconds=select_time_mod)


def get_ceil_mod_datetime(select_datetime, freq_str, will_warn=False):
    """
    指定した日時を指定する周期に応じて切り捨てる
    select_datetime: datetime.datetime
        指定日時
    freq_str: freq_str
        サンプリング周期に対応する文字列
    """
    if get_sec_from_freq(freq_str)  > get_sec_from_freq("D"):
        raise Exception("This function for frequency lowwer then day")
    sec_from_freq = get_sec_from_freq(freq_str)
    select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか

    if select_is_aware:
        timezone_datetime = select_datetime
    else:
        utc_timezone = timezone("UTC")  # これはどこでもよい
        timezone_datetime = utc_timezone.localize(select_datetime)
        
    select_time_unix =  timezone_datetime.timestamp() + timezone_datetime.utcoffset().total_seconds() 
    select_time_mod = int(select_time_unix) % int(sec_from_freq)
    if select_time_mod != 0:  # 余りが出てしまった場合
        if will_warn:
            warnings.warn("input datetime is ceiled")
        return select_datetime +timedelta(seconds=(sec_from_freq-select_time_mod))
    else:
        return select_datetime


def get_next_datetime(select_datetime, freq_str, clear_mod=True):
    """
    与えられた日時から，frequency_strを考慮して次の時間を求めるための関数．余りが出てしまった場合， warningを出す．
    "D"以下，さらに"W","2W","M","Y"を前提としている．
    select_datetime: datetime.datetime(aware or naive)
        与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
    freq_str: str
        frequencyに対応する文字列
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    #qfrom IPython.core.debugger import Pdb; Pdb().set_trace()
    freq_str = middle_sample_type_with_check(freq_str)

    select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか
    if select_is_aware:
        select_timezone = timezone(str(select_datetime.tzinfo))

    if clear_mod:  # 余りを無視する
        if freq_str in {"W","2W"}:  # 週足あるいは2週足を求めるとき
            years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
            hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

            weekday = select_datetime.weekday()
            if {hours, minutes, seconds} != {0} or weekday != 6:  # 時間以下が全て0でないときあるいは，日曜でなかった場合
                warnings.warn("input time is not coresponded to frequency. resampling result may be wrong")
                if {hours, minutes, seconds} != {0}:
                    hours, minutes, seconds = 0,0,0

                weekday += 1  # 月曜始まりを日曜始まりに強引に変更
                if freq_str == "W":  # 週足の場合
                    add_days = 7 - weekday
                elif freq_str == "2W":  # 2週足の場合
                    add_days = 14 - weekday
            else:  # 日曜だった場合
                if freq_str == "W":  # 週足の場合
                    add_days= 7
                elif freq_str == "2W":  # 2週足の場合
                    add_days = 14

            next_datetime = datetime.datetime(years, months, days+add_days, hours, minutes, seconds)
            if select_is_aware:  # select_datetmeがawareな場合
                next_datetime = select_timezone.localize(next_datetime)

        elif freq_str == "M":  # 月足を求めるとき
            years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
            hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

            if {days, hours, minutes, seconds} != {0}:  # 日にち以下がすべて0でないとき(月の開始時でないとき)
                warnings.warn("input time is not coresponded to frequency. next datetime may be wrong")
                days, hours, minutes, seconds = 1,0,0,0

            if months == 12:  # 12月のとき
                years += 1
                months = 1  # １月にする
            else:  # 月の開始時のとき
                months += 1  # 月を一つ加える
            next_datetime = datetime.datetime(years, months, days, hours, minutes, seconds)
            if select_is_aware:  # select_datetmeがawareな場合
                next_datetime = select_timezone.localize(next_datetime)

        elif freq_str == "Y":  # 年足で求めるとき（元旦始まり）
            years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
            hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

            if {months, days, hours, minutes, seconds} != {0}:  # 月以下がすべて0でないとき
                warnings.warn("input time is not coresponded to frequency. next datetime may be wrong")
                months, days, hours, minutes, seconds = 1,1,0,0,0

            years += 1
            next_datetime = datetime.datetime(years, months, days, hours, minutes, seconds)
            if select_is_aware:  # select_datetmeがawareな場合
                next_datetime = select_timezone.localize(next_datetime)

        else:  # それ以下の足を求める場合
            sec_from_freq = get_sec_from_freq(freq_str)
            next_datetime = get_floor_mod_datetime(select_datetime, freq_str, will_warn=True) + timedelta(seconds=sec_from_freq)
            return next_datetime

    else:  # （余りを残すとき）
        if freq_str in {"W","2W"}:  # 週足あるいは2週足を求めるとき
            if freq_str == "W":  # 週足を求める場合
                next_datetime = select_datetime + timedelta(days=7)
            elif freq_str == "2W":  # 2週足を求める場合
                next_datetime = select_datetime + timedelta(days=14)

        elif freq_str == "M":  # 月足の場合
            years, months = select_datetime.year, select_datetime.month
            if months == 12:  # 12月のとき
                years += 1
                months = 1  # １月にする
            else:
                months += 1  # 月を一つ加える
            next_datetime = datetime.datetime(years, months, 1, 0, 0, 0)
            if select_is_aware:  # select_datetmeがawareな場合
                next_datetime = select_timezone.localize(next_datetime)

        elif freq_str == "Y":  # 年足の場合，元旦始まり
            years = select_datetime.year
            years += 1
            next_datetime = datetime.datetime(years, 1, 1, 0, 0, 0)
            if select_is_aware:  # select_datetmeがawareな場合
                next_datetime = select_timezone.localize(next_datetime)

        else:  # それ以下の足の場合
            sec_from_freq = get_sec_from_freq(freq_str)
            next_datetime = select_datetime + timedelta(seconds=sec_from_freq)

    return next_datetime


def get_previous_datetime(select_datetime, freq_str, clear_mod=True):
        """
        与えられた日時から，frequency_strを考慮して前の時間を求めるための関数．必要ないだろうが，一応作っておく
        余りが出てしまった場合， warningを出す．
        "D"以下，さらに"W","2W","M","Y"を前提としている．
        select_datetime: datetime.datetime(aware or naive)
            与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
        freq_str: str
            frequencyに対応する文字列
        clear_mod: bool
            与える日時の余りを削除するかどうか
        """
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        freq_str = middle_sample_type_with_check(freq_str)

        select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか
        if select_is_aware:
            select_timezone = timezone(str(select_datetime.tzinfo))

        if clear_mod:  # 余りを無視する
            if freq_str in {"W","2W"}:  # 週足あるいは2週足を求めるとき
                years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
                hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

                weekday = select_datetime.weekday()
                if {hours, minutes, seconds} != {0} or weekday != 6:  # 時間以下が全て0でないときあるいは，日曜でなかった場合
                    warnings.warn("input time is not coresponded to frequency. resampling result may be wrong")
                    if {hours, minutes, seconds} != {0}:
                        hours, minutes, seconds = 0,0,0
        
                    weekday += 1  # 月曜始まりを日曜始まりに強引に変更
                    if freq_str == "W":  # 週足の場合
                        sub_days =  weekday
                    elif freq_str == "2W":  # 2週足の場合
                        sub_days = 7 + weekday
                else:  # 日曜だった場合
                    if freq_str == "W":  # 週足の場合
                        sub_days= 7
                    elif freq_str == "2W":  # 2週足の場合
                        sub_days = 14

                prev_datetime = datetime.datetime(years, months, days, hours, minutes, seconds) + timedelta(days=-sub_days)
                if select_is_aware:  # select_datetmeがawareな場合
                    prev_datetime = select_timezone.localize(prev_datetime)
                
            elif freq_str == "M":  # 月足を求めるとき
                years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
                hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

                if {days, hours, minutes, seconds} != {0}:  # 日にち以下がすべて0でないとき(月の開始時でないとき)
                    warnings.warn("input time is not coresponded to frequency. prev datetime may be wrong")
                    days, hours, minutes, seconds = 1,0,0,0

                if months == 1:  # 1月のとき
                    years -= 1
                    months = 12  # 12月にする
                else:  # 月の開始時のとき
                    months -= 1  # 月を一つ減らす
                prev_datetime = datetime.datetime(years, months, days, hours, minutes, seconds)
                if select_is_aware:  # select_datetmeがawareな場合
                    prev_datetime = select_timezone.localize(prev_datetime)

            elif freq_str == "Y":  # 年足で求めるとき（元旦始まり）
                years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
                hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

                if {months, days, hours, minutes, seconds} != {0}:  # 月以下がすべて0でないとき
                    warnings.warn("input time is not coresponded to frequency. prev datetime may be wrong")
                    months, days, hours, minutes, seconds = 1,1,0,0,0
                
                years -= 1
                prev_datetime = datetime.datetime(years, months, days, hours, minutes, seconds)
                if select_is_aware:  # select_datetmeがawareな場合
                    prev_datetime = select_timezone.localize(prev_datetime)

            else:  # それ以下の足を求める場合
                sec_from_freq = get_sec_from_freq(freq_str)
                previous_datetime = get_ceil_mod_datetime(select_datetime, freq_str, will_warn=True) - timedelta(seconds=sec_from_freq)
                return previous_datetime          

        else:  # （余りを残すとき）
            if freq_str in {"W","2W"}:  # 週足あるいは2週足を求めるとき
                if freq_str == "W":  # 週足を求める場合
                    prev_datetime = select_datetime + timedelta(days=-7)
                elif freq_str == "2W":  # 2週足を求める場合
                    prev_datetime = select_datetime + timedelta(days=-14)
                if select_is_aware:  # select_datetmeがawareな場合
                    prev_datetime = select_timezone.localize(prev_datetime)

            elif freq_str == "M":  # 月足の場合
                years, months = select_datetime.year, select_datetime.month
                if months == 1:  # 1月のとき
                    years -= 1
                    months = 12  # 12月にする
                else:
                    months -= 1  # 月を一つ減らす
                prev_datetime = datetime.datetime(years, months, 1, 0, 0, 0)
                if select_is_aware:  # select_datetmeがawareな場合
                    prev_datetime = select_timezone.localize(prev_datetime)

            elif freq_str == "Y":  # 年足の場合，元旦始まり
                years = select_datetime.year
                years += 1
                prev_datetime = datetime.datetime(years, 1, 1, 0, 0, 0)
                if select_is_aware:  # select_datetmeがawareな場合
                    prev_datetime = select_timezone.localize(prev_datetime)

            else:  # それ以下の足の場合
                sec_from_freq = get_sec_from_freq(freq_str)
                prev_datetime = select_datetime - timedelta(seconds=sec_from_freq)
        
        return prev_datetime


def get_utc_naive_datetime_from_datetime(select_datetime):
    """
    awareなdatetimeかnaiveなdatetime(local timeを前提)をutcのnaiveなdatetimeに返る
    select_datetime: datetime
    """
    # awareなdatetimeからutc
    select_datetime_unix = select_datetime.timestamp()  #システムローカルのタイムゾーンを考慮したunix時刻
    utc_naive_datetime = datetime.datetime.utcfromtimestamp(select_datetime_unix)  # unix時刻から考慮したutcのnaiveなdatetime
    return utc_naive_datetime


def add_datetime(select_datetime, freq_str, add_number=1, clear_mod=True):
    """
    add_numberのfreq_str分だけ進めたdatetime.datetimeを返す
    select_datetime: datetime.datetime
        指定する日時
    freq_str: str
        frequencyに対応する文字列
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    freq_str = middle_sample_type_with_check(freq_str)
    iter_datetime = select_datetime
    for _ in range(add_number):
        iter_datetime = get_next_datetime(iter_datetime, freq_str, clear_mod=clear_mod)

    return iter_datetime  


def sub_datetime(select_datetime, freq_str, sub_number=1, clear_mod=True):
    """
    sub_numberのfreq_str分だけ減らしたdatetime.datetimeを返す
    select_datetime: datetime.datetime
        指定する日時
    freq_str: str
        frequencyに対応する文字列
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    freq_str = middle_sample_type_with_check(freq_str)
    iter_datetime = select_datetime
    for _ in range(sub_number):
        iter_datetime = get_previous_datetime(iter_datetime, freq_str, clear_mod=clear_mod)

    return iter_datetime  


def get_timezone_datetime_like(select_datetime, like_datetime):
    """
    select_datetimeをlike_datetimeとおなじtimezoneにして返す
    select_datetime: datetime.datetime
    like_datetime: datetime.datetime
    """
    like_datetime_is_aware = like_datetime.tzinfo is not None
    if like_datetime_is_aware:
        like_timezone = timezone(str(like_datetime.tzinfo))
    
    select_datetime_is_aware = select_datetime.tzinfo is not None
    
    if not like_datetime_is_aware and not select_datetime_is_aware:# select_datetime, like_datetimeがどちらもnaiveな場合
        return select_datetime
    elif like_datetime_is_aware and not select_datetime_is_aware:# like_datetimeのみがawareな場合
        return like_timezone.localize(select_datetime)
    elif not like_datetime_is_aware and select_datetime_is_aware:# select_datetimeのみがawareな場合
        return select_datetime.replace(tzinfo=None)
    elif like_datetime_is_aware and select_datetime_is_aware:# select_datetime, like_datetimeがどちらもawareな場合
        if str(like_datetime.tzinfo) != str(select_datetime.tzinfo):
            return select_datetime.astimezone(like_timezone)
        else:
            return select_datetime


def get_next_workday_intraday_datetime(select_datetime, freq_str, clear_mod=False):
    """
    与えられた日時から，frequency_strと営業時間を考慮して次の時間を求めるための関数．余りが出てしまった場合， warningを出す．
    H以下を前提としている
    select_datetime: datetime.datetime(aware or naive)
        与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
    freq_str: str
        frequencyに対応する文字列
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    if get_sec_from_freq(freq_str) > get_sec_from_freq("H"):
        raise Exception("This freq is not suitable for considaring intraday")
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    if clear_mod:
        select_datetime = get_floor_mod_datetime(select_datetime, freq_str, will_warn=True)
    
    raw_next_datetime = get_next_datetime(select_datetime, freq_str, clear_mod=False)
    
    next_border_datetime, next_border_symbol = get_next_border_workday_intraday_jp(select_datetime)
    rest_seconds = get_sec_from_freq(freq_str)
    if check_workday_intraday_jp(select_datetime):  # 指定した日時が営業時間内の場合
        assert next_border_symbol=="border_end"
        if next_border_datetime > raw_next_datetime:  # 同じ営業時間内に入っている場合
            return raw_next_datetime
        else:
            rest_seconds -= (next_border_datetime - select_datetime).total_seconds()  # 終了境界まで引く]
            next_border_datetime, next_border_symbol = get_next_border_workday_intraday_jp(next_border_datetime)

    assert next_border_symbol=="border_start"
    border_start_datetime, _ = next_border_datetime, next_border_symbol
    while True:
        border_end_datetime, border_end_symbol = get_next_border_workday_intraday_jp(border_start_datetime)
        assert border_end_symbol=="border_end"
        # rest_secondsが営業時間内に入った場合
        if rest_seconds < (border_end_datetime-border_start_datetime).total_seconds():
            break

        rest_seconds -= (border_end_datetime-border_start_datetime).total_seconds()
        border_start_datetime, _ = get_next_border_workday_intraday_jp(border_end_datetime)

    out_datetime = border_start_datetime + timedelta(seconds=rest_seconds)
    return out_datetime


def get_previous_workday_intraday_datetime(select_datetime, freq_str, clear_mod=False):
    """
    与えられた日時から，frequency_strと営業時間を考慮して前の時間を求めるための関数．余りが出てしまった場合， warningを出す．
    H以下を前提としている
    select_datetime: datetime.datetime(aware or naive)
        与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
    freq_str: str
        frequencyに対応する文字列
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    if get_sec_from_freq(freq_str) > get_sec_from_freq("H"):
        raise Exception("This freq is not suitable for considaring intraday")
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    if clear_mod:
        select_datetime = get_ceil_mod_datetime(select_datetime, freq_str, will_warn=True)
    
    raw_previous_datetime = get_previous_datetime(select_datetime, freq_str, clear_mod=False)
    
    previous_border_datetime, previous_border_symbol = get_previous_border_workday_intraday_jp(select_datetime)
    rest_seconds = get_sec_from_freq(freq_str)
    if check_workday_intraday_jp(select_datetime):  # 指定した日時が営業時間内の場合
        assert previous_border_symbol=="border_start"
        if previous_border_datetime < raw_previous_datetime:  # 同じ営業時間内に入っている場合
            return raw_previous_datetime
        else:
            rest_seconds -= (select_datetime-previous_border_datetime).total_seconds()  # 終了境界まで引く
            previous_border_datetime, previous_border_symbol = get_previous_border_workday_intraday_jp(previous_border_datetime)

    assert previous_border_symbol=="border_end"
    border_end_datetime, _ = previous_border_datetime, previous_border_symbol
    while True:
        border_start_datetime, border_start_symbol = get_previous_border_workday_intraday_jp(border_end_datetime, force_is_end=True)
        assert border_start_symbol=="border_start"
        # rest_secondsが営業時間内に入った場合
        if rest_seconds < (border_end_datetime-border_start_datetime).total_seconds():
            break

        rest_seconds -= (border_end_datetime-border_start_datetime).total_seconds()
        border_end_datetime, _ = get_previous_border_workday_intraday_jp(border_start_datetime)

    out_datetime = border_end_datetime - timedelta(seconds=rest_seconds)
    return out_datetime


def add_workday_intraday_datetime(select_datetime, freq_str, add_number=1, clear_mod=True):
    """
    営業時間を考慮してadd_numberのfreq_str分だけ進めたdatetime.datetimeを返す
    select_datetime: datetime.datetime
        指定する日時
    freq_str: str
        frequencyに対応する文字列
    clear_mod: bool
        与える日時の余りを削除するかどうか
        
    """
    freq_str = middle_sample_type_with_check(freq_str)
    iter_datetime = select_datetime
    for _ in range(add_number):
        iter_datetime = get_next_workday_intraday_datetime(iter_datetime, freq_str, clear_mod=clear_mod)

    return iter_datetime  


def sub_workday_intraday_datetime(select_datetime, freq_str, sub_number=1, clear_mod=True):
    """
    営業時間を考慮してsub_numberのfreq_str分だけ減らしたdatetime.datetimeを返す
    select_datetime: datetime.datetime
        指定する日時
    freq_str: str
        frequencyに対応する文字列
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    freq_str = middle_sample_type_with_check(freq_str)
    iter_datetime = select_datetime
    for _ in range(sub_number):
        iter_datetime = get_previous_workday_intraday_datetime(iter_datetime, freq_str, clear_mod=clear_mod)

    return iter_datetime

if __name__ == "__main__":
  pass