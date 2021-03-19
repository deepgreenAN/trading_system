import jpholiday
import workdays
from pytz import timezone
import datetime

import numpy as np


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
    期間を指定して祝日を取得
    
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


def get_workdays_jp(start_datetime, end_datetime, to_date=True):
    """
    営業日を取得
    
    start_datetime: datetime.datetime
        開始時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    end_datetime: datetime.datetime
        終了時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    to_date: bool
        出力をdatetime.datetimeにするかdatetime.dateにするか
    """
    start_datetime, end_datetime = check_jst_datetimes_to_naive(start_datetime, end_datetime)
    
    def get_workdays_gen_jp(start_datetime, end_datetime):
        start_date = start_datetime.date()
        end_date = end_datetime.date()
        time_delta_days = (end_date - start_date).days
        for i in range(time_delta_days):
            out_datetime = start_datetime + datetime.timedelta(days=i)
            if out_datetime.weekday() not in (5,6):  # 土日ではない
                if not jpholiday.is_holiday(out_datetime):  # 休日ではない
                    if to_date:
                        out_datetime = out_datetime.date()
                    yield out_datetime
        
    # workdayのリストを取得
    workday_array = np.array([workday for workday in get_workdays_gen_jp(start_datetime, end_datetime)])
    return workday_array


def get_not_workdays_jp(start_datetime, end_datetime, to_date=True):
    """
    非営業日を取得(土日＋祝日)
    
    start_datetime: datetime.datetime
        開始時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    end_datetime: datetime.datetime
        囚虜時刻のjstのタイムゾーンが指定されたawareなdatetimeか，naiveなdatetime(ローカルなタイムゾーンとして日本を仮定)
    to_date: bool
        出力をdatetime.datetimeにするかdatetime.dateにするか
    """
    start_datetime, end_datetime = check_jst_datetimes_to_naive(start_datetime, end_datetime)
    
    def get_not_workdays_gen_jp(start_datetime, end_datetime):
        start_date = start_datetime.date()
        end_date = end_datetime.date()
        time_delta_days = (end_date - start_date).days
        for i in range(time_delta_days):
            out_datetime = start_datetime + datetime.timedelta(days=i)
            if out_datetime.weekday() in (5,6) or jpholiday.is_holiday(out_datetime):  # 土日か祝日を取得
                if to_date:
                    out_datetime = out_datetime.date()
                yield out_datetime
        
    # workdayのリストを取得
    not_workday_array = np.array([not_workday for not_workday in get_not_workdays_gen_jp(start_datetime, end_datetime)])
    return not_workday_array


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
    
    