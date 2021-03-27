import jpholiday
from pytz import timezone
import datetime

import numpy as np
import pandas as pd
from pathlib import Path


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


class JPHolidayGetter:
    """
    jpholidayを利用したHolidayGetter
    """
    def get_holidays(self, start_date, end_date, with_name=False):
        """
        期間を指定して祝日を取得．jpholidayを利用して祝日を取得している．

        start_date: datetime.date
            開始時刻のdate
        end_datetime: datetime.date
            終了時刻のdate
        with_name: bool
            休日の名前を出力するかどうか
        to_date: bool
            出力をdatetime.datetimeにするかdatetime.dateにするか
        """
        assert isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date)

        holidays_array = np.array(jpholiday.between(start_date, end_date))

        if not with_name:  # 祝日名がいらない場合
            return holidays_array[:,0].copy()

        return holidays_array


class CSVHolidayGetter:
    """
    CSVのソースファイルを利用したHolidayGetter
    """
    def __init__(self, csv_paths):
        if not isinstance(csv_paths, list):  # リストでないなら，リストにしておく
            csv_paths = [csv_paths]
            
        self.csv_paths = csv_paths
        
    def get_holidays(self, start_date, end_date, with_name=False):
        """
        期間を指定して祝日を取得．csvファイルを利用して祝日を取得している．

        start_date: datetime.date
            開始時刻のdate
        end_datetime: datetime.date
            終了時刻のdate
        with_name: bool
            休日の名前を出力するかどうか
        to_date: bool
            出力をdatetime.datetimeにするかdatetime.dateにするか
        """
        assert isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date)
        
        # datetime.dateをpd.Timestampに変換(datetime.dateは通常pd.DatetimeIndexと比較できないため)
        start_timestamp = pd.Timestamp(start_date)
        end_timestamp = pd.Timestamp(end_date)
        
        for i, csv_path in enumerate(self.csv_paths):
            holiday_df = pd.read_csv(csv_path, 
                                     header=None,
                                     names=["date", "holiday_name"],
                                     index_col="date",
                                     parse_dates=True
                                    )
            if i == 0:
                left_df = holiday_df
            else:
                append_bool = ~holiday_df.index.isin(left_df.index)  # 左Dataframeに存在しない部分を追加
                left_df = left_df.append(holiday_df.loc[append_bool], sort=True)

        
        # 指定範囲内の祝日を取得
        holiday_in_span_index = (start_timestamp<=left_df.index)&(left_df.index<end_timestamp)
        holiday_in_span_df = left_df.loc[holiday_in_span_index]
        
        holiday_in_span_date_array = holiday_in_span_df.index.date
        holiday_in_span_name_array = holiday_in_span_df.loc[:,"holiday_name"].values
        holiday_in_span_array = np.stack([holiday_in_span_date_array,
                                          holiday_in_span_name_array
                                         ],
                                         axis=1
                                        )
        
        if not with_name:  # 祝日名がいらない場合
            return holiday_in_span_date_array
            
        return holiday_in_span_array


class Option():
    """
    オプションの指定のためのクラス
    holiday_start_year: int
        利用する休日の開始年
    holiday_end_year: int
        利用する休日の終了年
    backend: str
        休日取得のバックエンド．csvかjpholidayのいずれかが選べる
    csv_source_paths: list of str or pathlib.Path
        バックエンドをcsvにした場合の休日のソースcsvファイル
    holiday_weekdays: list of int
        休日曜日の整数のリスト
    intraday_borders: list of list of 2 datetime.time
        日中を指定する境界時間のリストのリスト
    """
    def __init__(self):
        self._holiday_start_year = datetime.datetime.now().year-5
        self._holiday_end_year = datetime.datetime.now().year
        
        self._backend = "csv"
        self._csv_source_paths = [Path(__file__).parent / Path("source/holiday_naikaku.csv"),]
        
        self.make_holiday_getter()  # HolidayGetterを作成
        self.make_holidays()  # アトリビュートに追加
        
        self._holiday_weekdays = [5,6]  # 土曜日・日曜日
        self._intraday_borders = [[datetime.time(9,0), datetime.time(11,30)],
                                  [datetime.time(12,30), datetime.time(15,0)]]
        
    
    def make_holiday_getter(self):
        if self.backend == "jp_holiday":
            self._holiday_getter = JPHolidayGetter()
        elif self.backend == "csv":
            self._holiday_getter = CSVHolidayGetter(self.csv_source_paths)
        
    
    def make_holidays(self):
        """
        利用する休日のarrayとDatetimeIndexをアトリビュートとして作成
        """
        self._holidays_date_array = self._holiday_getter.get_holidays(start_date=datetime.date(self.holiday_start_year,1,1),
                                                    end_date=datetime.date(self.holiday_end_year,12,31),
                                                    with_name=False,
                                                   )
        self._holidays_datetimeindex =  pd.DatetimeIndex(self._holidays_date_array)
        
    
    @property
    def holiday_start_year(self):
        return self._holiday_start_year
    
    @holiday_start_year.setter
    def holiday_start_year(self, year):
        assert isinstance(year,int)
        self._holiday_start_year = year
        self.make_holidays()  # アトリビュートに追加
    
    @property
    def holiday_end_year(self):
        return self._holiday_end_year

    @holiday_end_year.setter
    def holiday_end_year(self, year):
        assert isinstance(year,int)
        self._holiday_end_year = year
        self.make_holidays()  # アトリビュートに追加
    
    @property
    def backend(self):
        return self._backend
    
    @backend.setter
    def backend(self, backend_str):
        if backend_str not in ("jp_holiday","csv"):
            raise Exception("backend must be 'jp_holiday' or 'csv'.")
        self._backend = backend_str
        self.make_holiday_getter()  # HolidayGetterを作成
    
    @property
    def csv_source_paths(self):
        #中身の確認
        for csv_source_path in self._csv_source_paths:
            if not isinstance(csv_source_path, str) and not isinstance(csv_source_path, Path):
                raise Exception("csv_source_paths must be list of str or pathlib.Path")
        
        return self._csv_source_paths
    
    @csv_source_paths.setter
    def csv_source_paths(self, path_list):
        self.csv_source_paths = path_list
        
    @property
    def holidays_date_array(self):
        return self._holidays_date_array
    
    @property
    def holidays_datetimeindex(self):
        return self._holidays_datetimeindex
    
    @property
    def holiday_weekdays(self):
        # 中身の確認
        for weekday in self._holiday_weekdays:
            if not isinstance(weekday, int):
                raise Exception("holiday_weekdays must be list of integer(0<=x<=6)")
                
        return self._holiday_weekdays
    
    @holiday_weekdays.setter
    def holiday_weekdays(self, weekdays_list):
        self._holiday_weekdays = weekdays_list
    
    @property
    def intraday_borders(self):
        # 中身の確認
        for border in self._intraday_borders:
            if not isinstance(border, list) or len(border) != 2:
                raise Exception("intraday_borders must be list of list whitch has 2 datetime.time")
                
            for border_time in border:
                if not isinstance(border_time, datetime.time):
                    raise Exception("intraday_borders must be list of list whitch has 2 datetime.time")
             
        return self._intraday_borders
    
    @intraday_borders.setter
    def intraday_borders(self, borders_list):
        self._intraday_borders = borders_list


# Optionの作成
option = Option()


def get_holidays_jp(start_date, end_date, with_name=False, independent=False):
        """
        期間を指定して祝日を取得．

        start_date: datetime.date
            開始時刻のdate
        end_datetime: datetime.date
            終了時刻のdate
        with_name: bool
            休日の名前を出力するかどうか
        to_date: bool
            出力をdatetime.datetimeにするかdatetime.dateにするか
        independent: bool，default:False
            休日をoptionから独立させるかどうか．FalseならばOption内で保持する休日が取得される
        """
        assert isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date)
        
        if not independent:
            # datetime.dateをpd.Timestampに変換(datetime.dateは通常pd.DatetimeIndexと比較できないため)
            start_timestamp = pd.Timestamp(start_date)
            end_timestamp = pd.Timestamp(end_date)

            holidays_in_span_index = (start_timestamp<=option.holidays_datetimeindex)&(option.holidays_datetimeindex<end_timestamp)  # DatetimeIndexを使うことに注意
            holidays_in_span_array = option.holidays_date_array[holidays_in_span_index]  # ndarrayを使うtamp(end_date)

            return holidays_in_span_array
        else:
            if option.backend == "jp_holiday":
                holiday_getter = JPHolidayGetter()
            elif option.backend == "csv":
                holiday_getter = CSVHolidayGetter(option.csv_source_paths)
                
            holidays_array = holiday_getter.get_holidays(start_date=start_date,
                                                         end_date=end_date,
                                                         with_name=with_name
                                                        ) 
            return  holidays_array


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
    holidays_in_span_datetimeindex = option.holidays_datetimeindex[holidays_in_span_index]  # ndarrayを使う

    # 期間中のdatetimeのarrayを取得
    if end_include:
        days_datetimeindex = pd.date_range(start=start_date, end=end_date, freq="D")  # 最終日も含める
    else:
        days_datetimeindex = pd.date_range(start=start_date, end=end_date-datetime.timedelta(days=1), freq="D")  # 最終日は含めない
    
    
    # 休日に含まれないもの，さらに土日に含まれないもののboolインデックスを取得
    holiday_bool_array = days_datetimeindex.isin(holidays_in_span_datetimeindex)  # 休日であるかのブール
    
    days_weekday_array = days_datetimeindex.weekday.values
    holiday_weekday_each_bool_arrays = [days_weekday_array==weekday for weekday in option.holiday_weekdays]  # inを使うのを回避
    holiday_weekday_bool_array = np.logical_or.reduce(holiday_weekday_each_bool_arrays)  # 休日曜日
    
    workdays_bool_array = (~holiday_bool_array)&(~holiday_weekday_bool_array)  # 休日でなく休日曜日でない
    
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
    holidays_in_span_datetimeindex = option.holidays_datetimeindex[holidays_in_span_index]  # pd.DatetimeIndexを使う

    # 期間中のdatetimeのarrayを取得
    if end_include:
        days_datetimeindex = pd.date_range(start=start_date, end=end_date, freq="D")  # 最終日も含める
    else:
        days_datetimeindex = pd.date_range(start=start_date, end=end_date-datetime.timedelta(days=1), freq="D")  # 最終日は含めない
    
    # 休日に含まれないもの，さらに休日曜日に含まれないもののboolインデックスを取得
    holiday_bool_array = days_datetimeindex.isin(holidays_in_span_datetimeindex)  # 休日であるかのブール
    
    days_weekday_array = days_datetimeindex.weekday.values
    holiday_weekday_each_bool_arrays = [days_weekday_array==weekday for weekday in option.holiday_weekdays]  # inを使うのを回避
    holiday_weekday_bool_array = np.logical_or.reduce(holiday_weekday_each_bool_arrays)  # 休日曜日
    
    not_workdays_bool_array = holiday_bool_array | holiday_weekday_bool_array  # 休日あるいは休日曜日
    
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
    # 休日であるかどうか
    is_holiday = (option.holidays_date_array==select_date).sum() > 0
    
    # 休日曜日であるかどうか
    is_holiday_weekday = select_date.weekday() in set(option.holiday_weekdays)
    
    is_workday = not any([is_holiday, is_holiday_weekday])
    
    return is_workday


def get_next_workday_jp(select_date, days=1, select_include=False, return_as="date"):
    """
    指定した日数後の営業日を取得
    select_date: datetime.date
        指定する日時
    days: int
        日数
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
        
    day_counter = 0
    holiday_weekdays_set = set(option.holiday_weekdays)  #setにした方が高速？

    holiday_bigger_select_index = (option.holidays_date_array<=select_date).sum()
    # 祝日イテレータ
    holiday_iter = iter(option.holidays_date_array[holiday_bigger_select_index:])
    def days_gen(select_date):
        add_days = 1  # select_dateを含まない
        while True:
            yield select_date + datetime.timedelta(days=add_days)
            add_days += 1
    # 日にちイテレータ
    days_iter = days_gen(select_date)

    # 以下二つのイテレーターを比較し，one_dayが休日に含まれる場合，カウントしカウントが指定に達した場合終了する
    one_day = next(days_iter)

    one_holiday = next(holiday_iter)

    while True:
        if one_day==one_holiday:  #その日が祝日である
            one_holiday = next(holiday_iter)
        else:
            if not one_day.weekday() in holiday_weekdays_set:  #その日が休日曜日である
                #print(one_day)
                day_counter += 1  # カウンターをインクリメント

        if day_counter >= days:
            break

        one_day = next(days_iter)
        
    if return_as=="date":
        return one_day
    elif return_as=="dt":
        return pd.Timestamp(one_day)


def get_workdays_number_jp(start_date, days, return_as="date"):
    """
    指定した日数分の営業日を取得
    start_date: datetime.date
        開始日時
    days: int
        日数
    return_as: str, defalt: 'dt'
        返り値の形式
        - 'dt':pd.Timstamp
        - 'datetime': datetime.datetime array
    """
    end_date = get_next_workday_jp(start_date, days=days, return_as="date")
    return get_workdays_jp(start_date, end_date, end_include=True, return_as=return_as)
        

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
    holidays_in_span_datetimeindex = option.holidays_datetimeindex[holidays_in_span_index]  # pd.DatetimeIndexを使う
    
    # 休日に含まれないもの，さらに土日に含まれないもののboolインデックスを取得    
    holiday_bool_array = dt_index.tz_localize(None).floor("D").isin(holidays_in_span_datetimeindex)  # 休日
    
    dt_index_weekday = dt_index.weekday
    holiday_weekday_each_bool_arrays = [dt_index_weekday==weekday for weekday in option.holiday_weekdays]  # inを使うのを回避
    holiday_weekday_bool_array = np.logical_or.reduce(holiday_weekday_each_bool_arrays)  # 休日曜日
    
    workdays_bool_array = (~holiday_bool_array)&(~holiday_weekday_bool_array)  # 休日でなく休日曜日でない
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
    pd.DatetimeIndexから，日中のデータのものを抽出．出力データ形式をreturn_asで指定する．
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
  
    bool_array = np.full(len(dt_index), False)
    
    # ボーダー内のboolをTrueにする
    for borders in option.intraday_borders:
        start_time, end_time = borders[0], borders[1]  # 開始時刻と終了時刻
        in_border_indice = dt_index.indexer_between_time(start_time=start_time, end_time=end_time, include_end=False)
        bool_array[in_border_indice] = True
    
    if return_as=="bool":
        return bool_array

    elif return_as=="index":
        intraday_indice = dt_index[bool_array]
        return intraday_indice


def extract_intraday_jp(df, return_as="df"):
    """
    データフレームから，日中のデータのものを抽出．出力データ形式をreturn_asで指定する．
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
    pd.DatetimeIndexから，営業日+日中のデータのものを抽出．出力データ形式をreturn_asで指定する．
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
    データフレームから，営業日+日中のデータのものを抽出．出力データ形式をreturn_asで指定する．
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
    ### get_holidays_jp
    ##########################
    
    start_date = datetime.date(2019, 1, 1)
    end_date = datetime.date(2020, 12, 31)

    holidays = get_holidays_jp(start_date, end_date, with_name=False)
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
