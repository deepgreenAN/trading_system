import datetime
from datetime import timedelta
from pytz import timezone
import warnings

from py_workdays import add_workday_intraday_datetime, sub_workday_intraday_datetime

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
    指定した日時を指定する周期に応じて切り捨てる. D以下・Y,M,Wに対応している．
    select_datetime: datetime.datetime
        指定日時
    freq_str: freq_str
        サンプリング周期に対応する文字列
    will_warn: bool
        切り捨て時にwarningを出すかどうか
    """
    select_date = select_datetime.date()
    select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか
    if select_is_aware:
        select_timezone = timezone(str(select_datetime.tzinfo))
    
    if freq_str == "Y":  # 年足で求めるとき（元旦始まり）
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second

        if {hour, minute, second} != {0} or {month, day} != {1}:  # 月以下がすべて0でないとき
            warnings.warn("input time is not coresponded to frequency. next datetime may be wrong")
            month, day, hour, minute, second = 1,1,0,0,0
        
        out_datetime = datetime.datetime(year=year,
                                         month=month,
                                         day=day,
                                         hour=hour,
                                         minute=minute,
                                         second=second
                                        )
        if select_is_aware:
            out_datetime = select_timezone.localize(out_datetime)

        return out_datetime
    
    elif freq_str == "M":  # 月足を求めるとき
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second

        if {hour, minute, second} != {0} or day!=1:  # 月の開始時でないとき
            if will_warn:
                warnings.warn("input datetime is floord")
            day, hour, minute, second = 1,0,0,0
            
        out_datetime = datetime.datetime(year=year,
                                         month=month,
                                         day=day,
                                         hour=hour,
                                         minute=minute,
                                         second=second
                                        )
        if select_is_aware:
            out_datetime = select_timezone.localize(out_datetime)

        return out_datetime
        
    elif freq_str == "W":  # 週足のとき
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second

        weekday = select_datetime.weekday()
        if {hour, minute, second} != {0} or weekday != 6:  # 時間以下が全て0でないときあるいは，日曜でなかった場合
            if will_warn:
                warnings.warn("input datetime is floord")
            if {hour, minute, second} != {0}:
                hour, minute, second = 0,0,0
                
            if weekday==6:
                weekday = 0  # 月曜始まりを日曜始まりに強引に変更
            else:
                weekday += 1  # 月曜始まりを日曜始まりに強引に変更
            
            sub_day = weekday
            out_date = select_date - timedelta(days=sub_day)
            
            out_datetime = datetime.datetime(year=out_date.year,
                                             month=out_date.month,
                                             day=out_date.day,
                                             hour=hour,
                                             minute=minute,
                                             second=second
                                            )
            if select_is_aware:
                out_datetime = select_timezone.localize(out_datetime)
                
            return out_datetime
        else:
            return select_datetime

    elif get_sec_from_freq(freq_str) <= get_sec_from_freq("D"):  #日足以下のの足の場合
        sec_from_freq = get_sec_from_freq(freq_str)
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
    
    else:
        raise Exception("This freq({})'s floor is not defined".format(freq_str))


def get_ceil_mod_datetime(select_datetime, freq_str, will_warn=False):
    """
    指定した日時を指定する周期に応じて切り捨てるD以下・Y,M,Wに対応している．
    select_datetime: datetime.datetime
        指定日時
    freq_str: freq_str
        サンプリング周期に対応する文字列
    will_warn: bool
        切り捨て時にwarningを出すかどうか
    """
    select_date = select_datetime.date()
    select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか
    if select_is_aware:
        select_timezone = timezone(str(select_datetime.tzinfo))
        
    if freq_str == "Y":  # 年足で求めるとき（元旦始まり）
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second

        if {hour, minute, second} != {0} or {month, day} != {1}:  # 月以下がすべて0でないとき
            if will_warn:
                warnings.warn("input datetime is ceiled")
            month, day, hour, minute, second = 1,1,0,0,0
        
        year += 1
        
        out_datetime = datetime.datetime(year=year,
                                         month=month,
                                         day=day,
                                         hour=hour,
                                         minute=minute,
                                         second=second
                                        )
        if select_is_aware:
            out_datetime = select_timezone.localize(out_datetime)

        return out_datetime
    
    elif freq_str == "M":  # 月足を求めるとき
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second

        if {hour, minute, second} != {0} or day!=1:  # 月の開始時でないとき
            if will_warn:
                warnings.warn("input datetime is ceiled")
            day, hour, minute, second = 1,0,0,0
            
        if month==12:
            year += 1
            month = 1
        else:
            month += 1
            
        out_datetime = datetime.datetime(year=year,
                                         month=month,
                                         day=day,
                                         hour=hour,
                                         minute=minute,
                                         second=second
                                        )
        if select_is_aware:
            out_datetime = select_timezone.localize(out_datetime)

        return out_datetime
        
    elif freq_str == "W":  # 週足のとき
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second

        weekday = select_datetime.weekday()
        if {hour, minute, second} != {0} or weekday != 6:  # 時間以下が全て0でないときあるいは，日曜でなかった場合
            if will_warn:
                warnings.warn("input datetime is ceiled")
            if {hour, minute, second} != {0}:
                hour, minute, second = 0,0,0
                
            if weekday==6:
                weekday = 0  # 月曜始まりを日曜始まりに強引に変更
            else:
                weekday += 1  # 月曜始まりを日曜始まりに強引に変更
            
            add_day = 6 - weekday  # 日曜からの差分
            out_date = select_date + timedelta(days=add_day)
            
            out_datetime = datetime.datetime(year=out_date.year,
                                             month=out_date.month,
                                             day=out_date.day,
                                             hour=hour,
                                             minute=minute,
                                             second=second
                                            )
            if select_is_aware:
                out_datetime = select_timezone.localize(out_datetime)
                
            return out_datetime
        else:
            return select_datetime

    elif get_sec_from_freq(freq_str) <= get_sec_from_freq("D"):  #日足以下のの足の場合
        sec_from_freq = get_sec_from_freq(freq_str)
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
    
    else:
        raise Exception("This freq({})'s ceil is not defined".format(freq_str))


def get_next_datetime(select_datetime, freq_str, number=1, clear_mod=True, will_warn=False):
    """
    与えられた日時から，frequency_strを考慮して次の時間を求めるための関数．余りが出てしまった場合， warningを出す．
    "D"以下，さらに"W","M","Y"に対応している
    select_datetime: datetime.datetime(aware or naive)
        与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
    freq_str: str
        frequencyに対応する文字列
    number: int
        進めるfreq_strの個数
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    assert number >= 0
    freq_str = middle_sample_type_with_check(freq_str)

    select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか
    if select_is_aware:
        select_timezone = timezone(str(select_datetime.tzinfo))

    if clear_mod:  # 余りを無視する
        select_datetime = get_floor_mod_datetime(select_datetime, freq_str, will_warn=will_warn)
        
    if freq_str == "Y":  # 年足の場合，元旦始まり
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second
        year += int(number)
        next_datetime = datetime.datetime(year, month, day, hour, minute, second)
        if select_is_aware:  # select_datetmeがawareな場合
            next_datetime = select_timezone.localize(next_datetime)
            
    elif freq_str == "M":  # 月足の場合
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second
        def inclement_month(year ,month):
            if month == 12:  # 12月のとき
                year += 1
                month = 1  # １月にする
            else:
                month += 1  # 月を一つ加える
            
            return year, month
                
        for _ in range(number):
            year, month = inclement_month(year, month)
        
        next_datetime = datetime.datetime(year, month, day, hour, minute, second)
        if select_is_aware:  # select_datetmeがawareな場合
            next_datetime = select_timezone.localize(next_datetime)

    elif freq_str=="W":  # 週足あるいは2週足を求めるとき
            next_datetime = select_datetime + timedelta(days=7)*number

    elif get_sec_from_freq(freq_str)<=get_sec_from_freq("D"):  # 日足以下の足の場合
        sec_from_freq = get_sec_from_freq(freq_str)
        next_datetime = select_datetime + timedelta(seconds=sec_from_freq)*number
    
    else:
        raise Exception("This freq({})'s next is not defined".format(freq_str))

    return next_datetime


def get_previous_datetime(select_datetime, freq_str, number=1, clear_mod=True, will_warn=False):
    """
    与えられた日時から，frequency_strを考慮して前の時間を求めるための関数．必要ないだろうが，一応作っておく
    余りが出てしまった場合， warningを出す．
    "D"以下，さらに"W","M","Y"に対応している
    select_datetime: datetime.datetime(aware or naive)
        与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
    freq_str: str
        frequencyに対応する文字列
    number: int
        減らすfreq_strの個数
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    assert number >= 0
    freq_str = middle_sample_type_with_check(freq_str)

    select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか
    if select_is_aware:
        select_timezone = timezone(str(select_datetime.tzinfo))

    if clear_mod:  # 余りを無視する
       select_datetime = get_ceil_mod_datetime(select_datetime, freq_str, will_warn=will_warn)


    if freq_str == "Y":  # 年足の場合，元旦始まり
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second
        year -= int(number)
        prev_datetime = datetime.datetime(year, month, day, hour, minute, second)
        if select_is_aware:  # select_datetmeがawareな場合
            prev_datetime = select_timezone.localize(prev_datetime)
    
    if freq_str == "M":  # 月足の場合
        year, month, day = select_datetime.year, select_datetime.month, select_datetime.day
        hour, minute, second = select_datetime.hour, select_datetime.minute, select_datetime.second
        def decrement_month(year, month):
            if month == 1:  # 1月のとき
                year -= 1
                month = 12  # 12月にする
            else:
                month -= 1  # 月を一つ減らす
                
            return year, month
        
        for _ in range(number):
            year, month = decrement_month(year, month)
        
        prev_datetime = datetime.datetime(year, month, day, hour, minute, second)
        if select_is_aware:  # select_datetmeがawareな場合
            prev_datetime = select_timezone.localize(prev_datetime)
    
    elif freq_str == "W":  # 週足あるいは2週足を求めるとき
        prev_datetime = select_datetime - timedelta(days=7) * number

    elif get_sec_from_freq(freq_str)<=get_sec_from_freq("D"):  # 日足以下の足の場合
        sec_from_freq = get_sec_from_freq(freq_str)
        prev_datetime = select_datetime - timedelta(seconds=sec_from_freq) * number
        
    else:
        raise Exception("This freq({})'s previous is not defined".format(freq_str))

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


def get_naive_datetime_from_datetime(select_datetime):
    """
    awareなdatetimeをnaiveなdatetime(local timeを前提)に変換する．
    select_datetime: datetime
    """
    select_is_aware = select_datetime.tzinfo is not None
    if select_is_aware:
        return select_datetime.replace(tzinfo=None)
    else:
        return select_datetime


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


def get_next_workday_intraday_datetime(select_datetime, freq_str, number=1, clear_mod=True, will_warn=False):
    """
    与えられた日時から，frequency_strと営業時間を考慮して次の時間を求めるための関数．余りが出てしまった場合， warningを出す．
    H以下を前提としている
    select_datetime: datetime.datetime(aware or naive)
        与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
    freq_str: str
        frequencyに対応する文字列
    number: int
        進めるfreq_strの個数
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    assert number >= 0
    if get_sec_from_freq(freq_str) > get_sec_from_freq("H"):
        raise Exception("This freq is not suitable for considaring intraday")
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    if clear_mod:
        select_datetime = get_floor_mod_datetime(select_datetime, freq_str, will_warn=will_warn)
    
    sec_from_freq = get_sec_from_freq(freq_str)
    out_datetime = add_workday_intraday_datetime(select_datetime, timedelta(seconds=sec_from_freq)*number)
    return out_datetime


def get_previous_workday_intraday_datetime(select_datetime, freq_str, number=1, clear_mod=False):
    """
    与えられた日時から，frequency_strと営業時間を考慮して前の時間を求めるための関数．余りが出てしまった場合， warningを出す．
    H以下を前提としている
    select_datetime: datetime.datetime(aware or naive)
        与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
    freq_str: str
        frequencyに対応する文字列
    number: int
        減らすfreq_strの個数
    clear_mod: bool
        与える日時の余りを削除するかどうか
    """
    assert number >= 0
    if get_sec_from_freq(freq_str) > get_sec_from_freq("H"):
        raise Exception("This freq is not suitable for considaring intraday")
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    if clear_mod:
        select_datetime = get_ceil_mod_datetime(select_datetime, freq_str, will_warn=True)
    
    sec_from_freq = get_sec_from_freq(freq_str)
    out_datetime = sub_workday_intraday_datetime(select_datetime, timedelta(seconds=sec_from_freq)*number)
    return out_datetime

if __name__ == "__main__":
  pass