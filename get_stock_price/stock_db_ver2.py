import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import datetime
from pytz import timezone

from contextlib import closing
import re
import warnings

from functools import lru_cache

from utils import ConvertFreqOHLCV, get_sec_from_freq, middle_sample_type_with_check, get_next_datetime, get_previous_datetime
from utils import get_df_freq, get_utc_naive_datetime_from_datetime, add_datetime, get_timezone_datetime_like

stock_str_list = ['Open', 'High', 'Low', 'Close', 'Volume']

def make_ohlcv_column_names(stock_names):
    """
    銘柄コードから，対応するOHLCVカラムを取得．
    stock_names: str or list of str
        検索したい株式，インデックスの銘柄コードあるいはティッカーコード
    """
    if isinstance(stock_names, str):  # 一応一つだけのとき
        stock_names = [stock_names]

    search_column_names = []
    for stock_name in stock_names:
        stock_name_ohlcv = [i+"_"+ stock_name for i in stock_str_list]  # グローバル変数を使うことに注意
        search_column_names.extend(stock_name_ohlcv)
        
    return search_column_names  

class ViewClosier():
    """
    データベースのビュー作成時のクロージャ―．
    """
    # viewのsetとview_closer_list(連動させる)
    view_set = set()  
    view_closier_list = []
    def __init__(self, stock_db, stock_names=None, start_datetime=None, end_datetime=None, view_name=None):
        """
        stock_names: str or list of str
            検索したい株式，インデックスの銘柄コードあるいはティッカーコード
        start_datetime: datatime.datetime
            取得を開始する日時，
        end_datetime: datetime.datetime, default None
            取得を終了する日時，
        view_name: str
            すでに存在するviewの名前
        """
        self.stock_db = stock_db
        self.start_datetime = start_datetime
        self.end_dsatetime = end_datetime
        self.IsClosed = False
        
        # viewを作成
        if stock_names is not None:
            self.IsCreated = True  # 実際に利用できるviewであるかどうか
            created_view_name = self._create_view(tuple(set(stock_names)),
                                                  start_datetime=start_datetime, 
                                                  end_datetime=end_datetime)
            if created_view_name in ViewClosier.view_set:
                raise ValueError("this view is already exists")  # おそらく_create_view内でエラーが出る．
            else:
                ViewClosier.view_set.add(created_view_name)
                ViewClosier.view_closier_list.append(self)

            self._view_name = created_view_name  # StockDBでのviewの呼び出しに利用
        elif view_name is not None:
            self.stock_names = None
            self.IsCreated = False  # これで作成されたViewClosierのviewは利用できない
            if view_name in ViewClosier.view_set:
                raise ValueError("this view is already exists, please close views.")
            else:
                ViewClosier.view_set.add(view_name)
                ViewClosier.view_closier_list.append(self)

            self._view_name = view_name
        else:
            raise ValueError("Set the stock_names or view_name")
        
    def _create_view(self, stock_names, start_datetime=None, end_datetime=None):
        """
        viewを作成
        stock_names: str or list of str
            検索したい株式，インデックスの銘柄コードあるいはティッカーコード
        start_datetime: datatime.datetime
            取得を開始する日時，
        end_datetime: datetime.datetime, default None
            取得を終了する日時，        
        """
        # メモ化して取得(いらないかも)
        table_names_array, column_names_array = self.stock_db._get_table_names_column_names_list_memorize(stock_names)

        # column_names_arrayから対応するstock_namesを取得
        def strip_ohlcv(column_name):
            column_name = column_name.lstrip("Open_")  # Open_を削除
            column_name = column_name.lstrip("High_")  # High_を削除
            column_name = column_name.lstrip("Low_")  # Low_を削除
            column_name = column_name.lstrip("Close_")  # Close_を削除
            column_name = column_name.lstrip("Volume_")  # Volume_を削除
            return column_name
        
        np_strip_ohlcv = np.frompyfunc(strip_ohlcv, 1, 1)  # ユニバーサル関数へ変換
        stock_names_array = np_strip_ohlcv(column_names_array)  # ユニバーサル関数を利用してstrip
        stock_names_array = np.unique(stock_names_array)
        self.stock_names = stock_names_array

        if len(table_names_array) < 1:  # stock_namesに該当するカラムがデータベースに無いとき
            raise ValueError("There is no column of stock_names. cannot create view")

        view_name = self.make_view_name(self.stock_names)
        # 同じviewがないかチェック
        if view_name in ViewClosier.view_set:
            raise ValueError("this view is already exists, please close views.")

        first_table_name = table_names_array[0]  # 最初だけ取得
        table_names_array_rest = np.delete(table_names_array, 0)  # 最初を削除

        def add_timestamp(table_name):
            return table_name+".timestamp"
        
        column_names_str = add_timestamp(first_table_name)+", "+ ", ".join(column_names_array)
        # sqlのjoin部分の作成
        join_str = ""
        for table_name in table_names_array_rest:
            join_str += "\nleft join {} \non \n{} = {}".format(table_name, add_timestamp(first_table_name), add_timestamp(table_name))
        
        # 時間の条件について(viewではなぜかパラメータが使えないので，文字列として扱っている)
        if start_datetime is not None and end_datetime is not None:  # start,endどちらも与えられた場合
            where_str = "\nwhere \n{} >= '{}' and {} < '{}'".format(add_timestamp(first_table_name), 
                                                                    start_datetime,
                                                                    add_timestamp(first_table_name),
                                                                    end_datetime
                                                                   )

        elif start_datetime is not None and end_datetime is None:  # startのみ与えられた場合
            where_str = "\nwhere \n{} >= '{}'".format(add_timestamp(first_table_name), start_datetime)

        elif start_datetime is None and end_datetime is not None:  # endのみ与えられた場合
            where_str = "\nwhere \n{} < '{}'".format(add_timestamp(first_table_name), end_datetime)
        else:  # どちらも与えられない場合
            where_str = ""

        sql = "create view {} as\nselect {} \nfrom {} {} {}".format(view_name, column_names_str, first_table_name, join_str, where_str)

        with closing(self.stock_db.make_connection()) as conn:
            c = conn.cursor()
            c.execute(sql)  # viewの作成(制限付き)
            conn.commit()
        
        return view_name

    def make_view_name(self, stock_names_array):
        """
        viewの名前をテーブル名から作成
        stock_names_array: ndarray
            銘柄名のndarray
        """
        def stock_name_int(stock_name):
            return stock_name, int(stock_name) 
        np_stock_name_int = np.frompyfunc(stock_name_int, 1, 2)
        stock_names_array, stock_name_numbers = np_stock_name_int(stock_names_array)
        stock_name_indice = np.argsort(stock_name_numbers.astype("int"))
        
        stock_names_array = stock_names_array[stock_name_indice].copy()
        return "view_{}".format("_".join(stock_names_array))

    def __enter__(self):
        return self  # withを使わないときと同じ挙動にする．
    
    def close(self):
        """
        自身に対応するviewを削除する．
        """
        # 対応するviewの削除
        if not self.IsClosed:  # すでにClosedされていない場合
            with closing(self.stock_db.make_connection()) as conn:
                c = conn.cursor()
                # viewの存在確認
                select_sql = "select type, name from sqlite_master where type = 'view' and name = '{}'".format(self._view_name)
                c.execute(select_sql)
                fetch_list = c.fetchall()
                if len(fetch_list) > 0:  # 存在するとき
                    drop_sql = "drop view {}".format(self._view_name)
                    c.execute(drop_sql)
                    conn.commit()
        
        self.IsClosed = True  # ClosedフラッグをTrueに
        # view_setから削除
        if self._view_name in ViewClosier.view_set:  # すでに消去されていない場合
            ViewClosier.view_set.remove(self._view_name)
            ViewClosier.view_closier_list.remove(self)

    @classmethod
    def make_view_closier_list(cls, stock_db):
        """
        データベース内に存在するview全てでViewClosierを作成し，(ここで作成されたものは，利用できない)
        そのsetを返す．
        stock_db: StockDatabase
            このStockDatabaseのパスを利用する
        """
        # viewの確認とViewClosierのリストの作成
        with closing(stock_db.make_connection()) as conn:
            c = conn.cursor()
            select_sql = "select name from sqlite_master where type = 'view'"
            c.execute(select_sql)
            fetch_list = c.fetchall()
        
        # 以下で内部にViewClosier.view_closeir_listに追加
        [cls(stock_db=stock_db, view_name=i[0]) for i in fetch_list if i[0] not in ViewClosier.view_set]  # すでに含まれていないもの
        
        return ViewClosier.view_closier_list
    
    @property
    def view_name(self):
        """
        外部から参照する．view_nameを取得する場合は，こちらを利用する．
        """
        if not self.IsClosed and self.IsCreated:  # まだCloseされておらず，さらにviewがcreateされていた場合
            return self._view_name
        elif self.IsClosed:
            raise ValueError("closed view cannot reference")
        elif not self.IsCreated:
            raise ValueError("This view is not one of this time please close this view")
    
    def __repr__(self):
        return_str = "view : name={}, IsCreated={}".format(self._view_name, self.IsCreated)
        return return_str
    
    def __exit__(self, ex_type, ex_value, trace):
        """
        closeを行う
        """
        self.close()

        return False  # falseにすると，通常通りの例外の扱いになるらしい
    

class StockDatabase():
    """
    株価データベースのインターフェース．データベースとしてsqliteを利用している．
    カラム数が膨大になるため，複数のテーブルを連携させて出力している．
    個別のデータの削除には対応していない．
    """
    def __init__(self, db_path, column_upper_limit=1000, table_name_base="table", database_frequency=None):
        """
        db_path: pathlib.Path
            データベースファイルのパス
        column_upper_limit: int
            一つのテーブルにおけるカラムの最大個数．これを変えても，インターフェースは変化しない
        table_name_base: int
            テーブル名がtable_name_base_0, table_name_base_1,...というようになる．
        """
        self.db_path = Path(db_path)
        
        self.column_upper_limit = column_upper_limit
        self.table_name_base = table_name_base
        self.database_frequency = database_frequency
        
        self._make_read_meta_table()  # メタデータの書き出しあるいは読み込み
        self._make_column_name_table()  # column_arrangeとcolumn_countテーブルを作成
        self._check_column_limit()  # データベースの上限値を確認

        # stock_in_memorizeで利用
        self.exists_stock_names_array = None

        # viewの確認と警告
        view_closier_list = ViewClosier.make_view_closier_list(self)
        if len(view_closier_list) > 0:
            str_view_closier_list = [str(i) for i in view_closier_list]
            warnings.warn("following views are exist, please execute close_all_view:\n\t{}".format(",\n\t".join(str_view_closier_list))) 
        
    def make_connection(self):
        """
        データベースのコネクションを返すメソッド
        """
        conn = sqlite3.connect(self.db_path,
                               detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
                              )
        return conn
    
    def _make_column_name_table(self): #カラム名の割り振りに関するテーブルの作成
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # _make_column_name_table
        with closing(self.make_connection()) as conn:
            # tableの存在確認
            c = conn.cursor()
            c.execute("select * from sqlite_master")
            fetch_list = c.fetchall()
            table_name_list = [i[1] for i in fetch_list]  # 二つ目がtable_name
            if "column_arrange" not in table_name_list:
                table_create_sql = "create table column_arrange (column_name text unique, arrange_table text)"
                c.execute(table_create_sql)
                
            if "column_count" not in table_name_list:
                table_create_sql = "create table column_count (table_name text unique, table_number integer, count integer)"  # boolは無いのでテキスト
                c.execute(table_create_sql)
                
                first_row_sql = "insert into column_count (table_name, table_number, count) values('{}_0', 0, 0)".format(self.table_name_base)
                c.execute(first_row_sql)
                
                first_create_sql = "create table {}_0 (timestamp timestamp primary key)".format(self.table_name_base)
                c.execute(first_create_sql)
                
            conn.commit()
    
    def _make_read_meta_table(self):  #メタデータの記録されるテーブの作成あるいは読み込み
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # _make_column_name_table
        with closing(self.make_connection()) as conn:
            # tableの存在確認
            c = conn.cursor()
            c.execute("select * from sqlite_master")
            fetch_list = c.fetchall()
            table_name_list = [i[1] for i in fetch_list]  # 二つ目がble_name
            if "meta_data" not in table_name_list:
                table_create_sql = "create table meta_data (id integer primary key, frequency text)"  # 必要がある場合は，ここに追加していく
                c.execute(table_create_sql)
                
                # self.database_frequencyがNoneだった場合
                if self.database_frequency is None:
                    c.execute("drop table meta_data")  # テーブルの削除
                    raise ValueError("Set the database frequency for initial database")
                    

                insert_0_row_sql = "insert into meta_data (id, frequency) values(0, '{}')".format(self.database_frequency)
                c.execute(insert_0_row_sql)
            else:
                select_sql = "select frequency from meta_data where id = 0"  # idが0のもののみ利用
                c.execute(select_sql)
                fetch_list = c.fetchone()
                database_frequency = fetch_list[0]
                if self.database_frequency != database_frequency and self.database_frequency is not None:
                    raise ValueError("Frequency of StockDatabase object is differ from frequency writtend in db(meta_data table) :'{}'".format(database_frequency))

                self.database_frequency = database_frequency  # データベースのfrequencyを読み取ったものにする．
            conn.commit()
            
    def _check_column_limit(self):
        with closing(self.make_connection()) as conn:
            # tableの存在確認
            c = conn.cursor()
            check_count_sql = "select max(count) from column_count"
            c.execute(check_count_sql)
            max_count = c.fetchone()[0]
                
            if max_count > self.column_upper_limit:
                raise ValueError("may be table column upper limit is different from given database please reload database as new db")
     
    def _arrange_column(self, new_column_names):  # 新しく追加したカラム
        """
        新しいカラムを，テーブル名をキーとしたディクショナリに振り分ける
        new_column_names: list of str or array of str
            新しいカラムのリストあるいはndarray
        Returns
        -------
        table_name_column_names_dict: {"table_name":array([new_column_names,])} 
        """
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # _arrange_column
        new_column_names_array = np.array(new_column_names)  # 集合計算の高速化
            
        with closing(self.make_connection()) as conn:
            c = conn.cursor()
            c.execute("select table_name, table_number, count from column_count where table_number = (select max(table_number) from column_count)")  # サブクエリを利用している
            now_write_table_data = c.fetchone()
            
            now_write_table_name = now_write_table_data[0]
            now_write_table_number = now_write_table_data[1]
            now_write_table_count = now_write_table_data[2]
            
            # 今書き込み済みのテーブルの名前
            self.real_now_write_table_name = now_write_table_name
            self.real_now_write_table_number = now_write_table_number
            self.real_now_write_table_count = now_write_table_count
        
            
            # 計算した既存カラムの数
            exists_column_number = self.column_upper_limit * self.real_now_write_table_number + self.real_now_write_table_count
            # 計算した合計カラムの数
            sum_column_number = exists_column_number + len(new_column_names_array)
            # 新しく必要なテーブルの個数
            create_table_number = sum_column_number // self.column_upper_limit - exists_column_number // self.column_upper_limit
            
            table_name_column_names_dict = {}
            
            if create_table_number < 1:  # 新しいテーブルが必要無いとき
                table_name_column_names_dict[now_write_table_name] = new_column_names_array  # 新しいarrayすべてが入る
                now_write_table_count = now_write_table_count + len(new_column_names_array)
            else:  # 新しいテーブルが必要なとき
                # （制限数×現在のテーブル番号＋１)-現在のカラム数
                add_write_table_count = (now_write_table_number+1)*self.column_upper_limit - exists_column_number
                table_name_column_names_dict[now_write_table_name] = new_column_names_array[:add_write_table_count]
                new_column_names_array = np.setdiff1d(new_column_names_array, table_name_column_names_dict[now_write_table_name])  #加えた部分は削除する
                now_write_table_count = now_write_table_count + add_write_table_count
            
            # カウントのアップデート
            update_sql = "update column_count set count = ? where table_name = ?"  # もしかしたらここでやらなくてもいいかも，例えば実際に値を入れるときとか
            c.execute(update_sql,[now_write_table_count, now_write_table_name])

            for i in range(create_table_number):
                now_write_table_number = now_write_table_number + 1  # 新しいので，追加
                now_write_table_name = "{}_{}".format(self.table_name_base, str(now_write_table_number))
                now_write_table_count = 0
                
                if len(new_column_names_array) < self.column_upper_limit:  # これ以上新しいテーブルが必要無いとき
                    table_name_column_names_dict[now_write_table_name] = new_column_names_array  # 残り全部が入る
                    now_write_table_count = now_write_table_count + len(new_column_names)
                else:
                    table_name_column_names_dict[now_write_table_name] = new_column_names_array[:self.column_upper_limit]  # 制限分が入る
                    new_column_names_array = np.setdiff1d(new_column_names_array, table_name_column_names_dict[now_write_table_name])  # 加えた部分は削除
                    now_write_table_count = now_write_table_count + self.column_upper_limit
                    

                #新しいテーブルの作成
                create_table = "create table {} (timestamp timestamp primary key)".format(now_write_table_name)
                c.execute(create_table)
                
                # column_countのインサート
                insert_new_row_sql = "insert into column_count(table_name, table_number, count) values('{}', {}, {})".format(now_write_table_name,
                                                                                                                             now_write_table_number,
                                                                                                                             now_write_table_count
                                                                                                                            )
                
                c.execute(insert_new_row_sql)
            conn.commit()

        return table_name_column_names_dict

    def _get_table_name_column_names_dict(self, search_column_names, insert_new_column=True):
        """
        引数のカラムを，テーブル名をキーとしたディクショナリに振り分ける
        search_column_names: list of str or array of str
            カラムのリストあるいはndarray
        insert_new_column: bool
            新しいカラムだった場合に，カラムを挿入するかどうか，つまり，upsertのときはTrue, searchのときはfalse
        Returns
        -------
        table_name_column_names_dict: {"table_name":array([new_column_names,])} 
        """
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # _get_table_name_column_names
        #ndarrayにする(高速にするため)
        search_column_names_array = np.array(search_column_names)        
        
        with closing(self.make_connection()) as conn:
            # tableの存在確認
            c = conn.cursor()
            
            column_names_str_list = ["'{}'".format(i) for i in search_column_names]  # 引用符を追加
            all_column_names_str = ", ".join(column_names_str_list)
            
            select_sql = "select column_name, arrange_table from column_arrange where column_name in ({})".format(all_column_names_str)  # columnとtable_nameのリスト
            c.execute(select_sql)
            column_table_df = pd.DataFrame(c.fetchall(), columns=["column_name", "table_name"])
            known_column_names_array = column_table_df["column_name"].values  # データベースに存在していたカラム名のarray
            
            c.execute("select * from column_arrange")
            
            def get_values(df):  # valuesを与えるための関数
                return df.values
            # table名をキーとし，カラム名のarrayをvalueとする辞書,今回はpandasのgroupbyとapplyを利用して求める
            table_name_column_names_dict = dict(column_table_df.groupby("table_name")["column_name"].apply(get_values))  # すでに保存したカラム
            
            unknown_column_names_array = np.setdiff1d(search_column_names_array, known_column_names_array)
        
        if insert_new_column and len(unknown_column_names_array) > 0:  #新しいカラムを追加する場合つまり，upsertする場合
            new_table_name_column_names_dict = self._arrange_column(unknown_column_names_array)
            if self.real_now_write_table_count in table_name_column_names_dict.keys():
                new_table_name_column_names_dict[self.real_now_write_table_name] = np.union1d(table_name_column_names_dict[self.real_now_write_table_name],
                                                                                              new_table_name_column_names_dict[self.real_now_write_table_name]
                                                                                             )
            table_name_column_names_dict.update(new_table_name_column_names_dict)  # 新しく得られた辞書で更新
        
        return table_name_column_names_dict

    def _get_table_names_column_names_list(self, stock_names):
        """
        銘柄コードのタプルから，テーブル名，カラム名のarrayを出力する．search_メソッドで利用する．
        こちらはメモライズをしない方
        """
        search_column_names = make_ohlcv_column_names(stock_names)
        search_column_names_array = np.array(search_column_names)

        with closing(self.make_connection()) as conn:
            # tableの存在確認
            c = conn.cursor()
            
            column_names_str_list = ["'{}'".format(i) for i in search_column_names]  # 引用符を追加
            all_column_names_str = ", ".join(column_names_str_list)

            select_sql = "select column_name, arrange_table from column_arrange where column_name in ({})".format(all_column_names_str)  # columnとtable_nameのリスト
            c.execute(select_sql)
            column_table_df = pd.DataFrame(c.fetchall(), columns=["column_name", "table_name"])

        exists_table_names_array = column_table_df["table_name"].values.copy()
        exists_column_names_array = column_table_df["column_name"].values.copy()

        # search_column_names_arrayからexists_column_names_arrayにあるものを取得
        search_column_bools = np.in1d(search_column_names_array, exists_column_names_array)  # 順番の維持のため
        search_column_names_array = search_column_names_array[search_column_bools]  # 順番が維持される
        table_names_unique = np.unique(exists_table_names_array)  # ユニークにしておく，順番はどうでもよい

        return table_names_unique, search_column_names_array

    @lru_cache(maxsize=None)
    def _get_table_names_column_names_list_memorize(self, stock_names):
        """
        銘柄コードのタプルから，テーブル名，カラム名のarrayを出力する．search_メソッドで利用する．
        こちらでメモライズを行う．いらなかったかも(ほとんど使わない)
        """
        table_names, column_names = self._get_table_names_column_names_list(stock_names)
        return table_names, column_names

    def _upsert_each_table(self, table_name, df, same_table_columns,item_replace_type="replace_null"):
        """
        sqliteの記法を用いてupsertする．
        pandas.DataFrameをリストにする方法を二つ用意
        table_name: str
            テーブル名
        df: pd.DataFrame
            挿入するのに利用するデータフレーム
        item_replace_type: ["replace", "nothing", "replace_null"] defalt, "replace_null"
            すでにレコードがあったときにの挙動
            "replace":更新する
            "nothing":何もしない
            "replace_null":値がヌルの場合，更新する
        same_table_columns: str or list of str or array of list
            テーブル名に対応するカラムの名前，あるいはそのリストかndarray
        """
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # _upsert_each_table
        
        if len(same_table_columns) == 0:
            return None
        
        replace_types = ["replace", "nothing", "replace_null"]  # それぞれ，値を更新，更新しない，nullのとき更新 
        if item_replace_type not in replace_types:
            raise ValueError("item_replace_type is invalid")
        
        same_table_columns_array = np.array(same_table_columns)  #高速演算のためのndarray化
        
        # 部分DataFrameを取得
        sub_df = df.loc[:,same_table_columns_array]
        # DataFrameをリストに変換
        #df_list = self._df_tolist_simple(sub_df)
        sub_df_list = self._df_tolist_numpy(sub_df)
        
        #print("df_list",df_list)   
        # sqlのformatのための辞書
        format_dict = {}
        format_dict["table_name"] = table_name
        
        format_dict["column_names"] = ", ".join(same_table_columns)
        format_dict["questions"] = ", ".join((len(same_table_columns)+1) * ["?"])  #+1 はtimestampの分
        set_excluded_list = [i+"=excluded."+i for i in same_table_columns]  # arrayを使うべき？
        
        set_excluded_use_coalesce = ["{0}=coalesce({0},excluded.{0})".format(i) for i in same_table_columns]  # coalesce関数を用いた場合  # arrayを使うべき？
        format_dict["column_and_value"] = ", ".join(set_excluded_list)
        format_dict["column_and_value_null"] = ", ".join(set_excluded_use_coalesce)
        
        if item_replace_type == "replace":  # すべて更新する
            upsert_sql = """
            insert into {0[table_name]} (timestamp, {0[column_names]})
              values({0[questions]})
              on conflict(timestamp)
              do update set {0[column_and_value]}
            """.format(format_dict)
        elif item_replace_type == "nothing":  # 更新されない
            upsert_sql =  """
            insert into {0[table_name]} (timestamp, {0[column_names]})
              values({0[questions]})
              on conflict(timestamp)
              do nothing
            """.format(format_dict)
        elif item_replace_type == "replace_null":  # nullの値のみ更新
            upsert_sql =  """
            insert into {0[table_name]} (timestamp, {0[column_names]})
              values({0[questions]})
              on conflict(timestamp)
              do update set {0[column_and_value_null]}
            """.format(format_dict)
            
        with closing(self.make_connection()) as conn:
            # columnの存在確認と追加
            c = conn.cursor()
            c.execute("pragma table_info('{}')".format(table_name))
            fetch_list = c.fetchall()
            if len(fetch_list) > 0:
                db_column_names_array = np.array(fetch_list)[:,1]  # 二つ目がcolumn_name
            else:
                db_column_names_array = np.array([]).astype("object")
            add_column_names_array = np.setdiff1d(same_table_columns_array, db_column_names_array)  # 追加するカラム名のarray 
            
            # カラムの追加(sqliteではカラムの複数追加が出来ない)
            def add_column(add_column_name):
                add_column_sql = "alter table {} add column {} real".format(table_name, add_column_name)
                c.execute(add_column_sql)
             
            # カラムの追加
            [add_column(i) for i in add_column_names_array]
            c.executemany(upsert_sql, sub_df_list)
            
            table_names_array = np.array(len(add_column_names_array) * [table_name]).astype("object")
            stacked_add_column_and_table_name = np.concatenate([add_column_names_array[:,None], table_names_array[:,None]], axis=1)  # 行方向にコンカット
            
            # column_arrangeデータベースの書き換え
            c.executemany("insert into column_arrange(column_name, arrange_table) values(?, ?)", stacked_add_column_and_table_name)
            
            conn.commit()
            
    def upsert(self, df, item_replace_type="replace_null"):
        """
        sqliteの記法を用いてupsertする．
        pandas.DataFrameをリストにする方法を二つ用意
        table_name: str
            テーブル名
        df: pd.DataFrame
            挿入するのに利用するデータフレーム
        item_replace_type: ["replace", "nothing", "replace_null"] defalt, "replace_null"
            すでにレコードがあったときにの挙動
            "replace":更新する
            "nothing":何もしない
            "replace_null":値がヌルの場合，更新する
        """
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # upsert
        
        upsert_df = df.copy()  # index等を変更するため，コピーする

        # dfが何も無いとき
        if len(upsert_df.index) < 1:
            return None

        # dfのfrequency判定
        df_freq_str = get_df_freq(upsert_df)
        if df_freq_str != self.database_frequency:
            raise ValueError("trying to upsert db with dataframe witch has different frequency of db")

        # dfのタイムゾーンに関する前処理
        upsert_df = self._df_timezone_convert_for_sqlite(upsert_df)
        
        replace_types = ["replace", "nothing", "replace_null"]  # それぞれ，値を更新，更新しない，nullのとき更新 
        if item_replace_type not in replace_types:
            raise ValueError("item_replace_type is invalid")
        
        #df_columns = list(set(df.columns))  # setにした方が安全
        df_columns = list(upsert_df.columns)  # setにすると勝手にソートされる
        table_name_column_names_dict = self._get_table_name_column_names_dict(df_columns, insert_new_column=True)

        for table_name_key in table_name_column_names_dict.keys():
            column_names_array = table_name_column_names_dict[table_name_key]
            self._upsert_each_table(table_name_key, upsert_df, column_names_array, item_replace_type=item_replace_type)

    def create_view(self, stock_names, start_datetime=None, end_datetime=None):
        """
        指定した銘柄コードに対応するカラムをviewに指定する
        stock_names: str or list of str
            検索したい株式，インデックスの銘柄コードあるいはティッカーコード
        start_datetime: datatime.datetime
            取得を開始する日時．
        end_datetime: datetime.datetime, default None
            取得を終了する日時．
        """
        if isinstance(stock_names, str):  # 一応一つだけのとき
            stock_names = [stock_names]

        # utcのnaiveなdatetimeに変更
        if start_datetime is not None:
            utc_naive_start_datetime = get_utc_naive_datetime_from_datetime(start_datetime)  # utcのnaiveなdatetimeに変更
        else:
            utc_naive_start_datetime = None
        
        if end_datetime is not None:
            utc_naive_end_datetime = get_utc_naive_datetime_from_datetime(end_datetime)  # utcのnaiveなdatetimeに変更
        else:
            utc_naive_end_datetime = None

        view_closier = ViewClosier(self, 
                                   stock_names=stock_names, 
                                   start_datetime=utc_naive_start_datetime,
                                   end_datetime=utc_naive_end_datetime
                                   )
        return view_closier

    def _get_query_df_join(self, 
                           stock_names, 
                           start_datetime=None, 
                           end_datetime=None
                           ):
        """
        銘柄コードのクエリデータを開始時間と終端時間を指定して取得する．joinを用いてtableを連結させている．
        stock_names: str or list of str
            検索したい株式，インデックスの銘柄コードあるいはティッカーコード
        start_datetime: datatime.datetime
            取得を開始する日時．
        end_datetime: datetime.datetime, default None
            取得を終了する日時．
        """
        # テーブル名のリスト,探すカラム名のリスト
        stock_names = sorted(set(stock_names), key=stock_names.index)  # 元の順番でソート

        search_table_names, search_column_names = self._get_table_names_column_names_list(stock_names)
        if len(search_table_names) < 1:  # stock_namesに該当するカラムがデータベースに無いとき
            return None  # Noneを返す．

        first_table_name = search_table_names[0]  # 最初だけ取得
        search_table_names_rest = np.delete(search_table_names, 0)  # 最初を削除

        def add_timestamp(table_name):
            return table_name+".timestamp"
        
        column_names_str = add_timestamp(first_table_name)+", "+ ", ".join(search_column_names)
        # sqlのjoin部分の作成
        join_str = ""
        for table_name in search_table_names_rest:
            join_str += "\nleft join {} \non \n{} = {}".format(table_name, add_timestamp(first_table_name), add_timestamp(table_name))
        
        # 時間の条件について
        if start_datetime is not None and end_datetime is not None:  # start,endどちらも与えられた場合
            where_str = "\nwhere \n{} >= ? and {} < ?".format(add_timestamp(first_table_name), 
                                                            add_timestamp(first_table_name),
                                                            )
            query_param = [start_datetime, end_datetime]

        elif start_datetime is not None and end_datetime is None:  # startのみ与えられた場合
            where_str = "\nwhere \n{} >= ?".format(add_timestamp(first_table_name))
            query_param = [start_datetime,]

        elif start_datetime is None and end_datetime is None:  # endのみ与えられた場合
            where_str = "\nwhere \n{} < ?".format(add_timestamp(first_table_name))
            query_param = [end_datetime,]
        
        else:  # どちらも与えられない場合，エラー
            raise ValueError("datetime is not setted. Set the start datetime or end_datetime")

        sql = "select {} \nfrom {} {} {}".format(column_names_str, first_table_name, join_str, where_str)
        
        with closing(self.make_connection()) as conn:
            query_df = pd.read_sql_query(sql,
                                        con=conn,
                                        index_col="timestamp",
                                        params=query_param
                                        )

        return query_df
    
    def _get_query_df_view(self,
                           stock_names, 
                           view,
                           start_datetime=None, 
                           end_datetime=None,
                           ):
        """
        銘柄コードのクエリデータを開始時間と終端時間を指定して取得する．viewを用いて取得している．
        stock_names: str or list of str
            検索したい株式，インデックスの銘柄コードあるいはティッカーコード
        view: ViewClosier
            検索するviewと対応するViewClosier
        start_datetime: datatime.datetime
            取得を開始する日時，対応している値を含む
        end_datetime: datetime.datetime, default None
            取得を終了する日時，対応している値を含むかどうかはis_end_includeで指定する
        """
        # viewの確認
        if not isinstance(view, ViewClosier):
            raise ValueError("view must be ViewCloseir object")

        # 銘柄のarray
        stock_names = sorted(set(stock_names), key=stock_names.index)  # 順番を維持しつつsetに
        stock_names_array = np.array(list(stock_names))

        # viewのstock_names_arrayとの共通部分を取得
        sub_stock_names_bools = np.in1d(stock_names_array, view.stock_names)  # 順番の維持のため
        sub_stock_names_array = stock_names_array[sub_stock_names_bools]  # 順番を維持した共通部分

        if len(sub_stock_names_array) < 1:  # stock_namesと対応する銘柄がviewに無かった場合
            return None  # Noneを返す．

        search_column_names = make_ohlcv_column_names(sub_stock_names_array)

        column_names_str = "timestamp, "+ ", ".join(search_column_names)

        if start_datetime is not None and end_datetime is not None:  # start,endどちらも与えられた場合
            where_str = "\nwhere \n timestamp >= ? and timestamp < ?"
            query_param = [start_datetime, end_datetime]

        elif start_datetime is not None and end_datetime is None:  # startのみ与えられた場合
            where_str = "\nwhere \n timestamp >= ?"
            query_param = [start_datetime,]

        elif start_datetime is None and end_datetime is None:  # endのみ与えられた場合
            where_str = "\nwhere \n timestamp < ?"
            query_param = [end_datetime,]
        
        else:  # どちらも与えられない場合，エラー
            raise ValueError("datetime is not setted. Set the start_datetime or end_datetime")

        view_name = view.view_name

        sql = "select {} \nfrom {} {}".format(column_names_str, view_name, where_str)

        with closing(self.make_connection()) as conn:
            query_df = pd.read_sql_query(sql,
                                        con=conn,
                                        index_col="timestamp",
                                        params=query_param
                                        )

        return query_df

    def search_span(self, 
                    stock_names,
                    start_datetime=None,
                    end_datetime=None, 
                    freq_str="T", 
                    is_end_include=False, 
                    to_tokyo=True,
                    view=None
                    ):
        """
        期間を指定してデータを取得
        stock_names: str or list of str
            検索したい株式，インデックスの銘柄コードあるいはティッカーコード
        start_datetime: datatime.datetime
            取得を開始する日時，対応している値を含む
        end_datetime: datetime.datetime, default None
            取得を終了する日時，対応している値を含むかどうかはis_end_includeで指定する
        freq_str:
            取得するDataFrameのindexのサンプル周期
        is_end_include:
            end_timeの時の値を含むかどうか
        returns
        -------
        resampled_df: pandas.DataFrame
            取得されたDataFrame
        """ 
        #import pdb; pdb.set_trace()  # search_span

        if isinstance(stock_names, str):  # 一応一つだけのとき
            stock_names = [stock_names]

        transform_dict = {"freq_str":freq_str, "to_tokyo":to_tokyo}  # transformに渡す辞書

        # 一応get_nextdatetimeで比較する
        next_datetime = get_next_datetime(start_datetime, freq_str, has_mod=True)
        if next_datetime > end_datetime:  # サンプリング周期一つ分のスパンにも満たない場合
            warnings.warn("this span less than one frequency")

        # utcのnaiveなdatetimeに変更
        if start_datetime is not None:
            utc_naive_start_datetime = get_utc_naive_datetime_from_datetime(start_datetime)  # utcのnaiveなdatetimeに変更
        else:
            warnings.warn("start_datetime is not setted, attention the memory use")
            utc_naive_start_datetime = None
        
        if end_datetime is not None:
            if is_end_include:  # 最後のサンプリング周期分も含む場合
                # end_datetimeのサンプル分も含める
                next_end_datetime = get_next_datetime(end_datetime, freq_str, has_mod=True)
                utc_naive_end_datetime = get_utc_naive_datetime_from_datetime(next_end_datetime)  # utcのnaiveなdatetimeに変更
            else:
                # 普通に，end_datetimeは含めないようにする
                utc_naive_end_datetime = get_utc_naive_datetime_from_datetime(end_datetime)  # utcのnaiveなdatetimeに変更
        else:
            warnings.warn("end_datetime is not setted, attention the memory use")
            utc_naive_start_datetime = None

        # query_dfの取得
        if view is None:  # viewを用いない場合
            query_df = self._get_query_df_join(stock_names=stock_names,
                                               start_datetime=utc_naive_start_datetime, 
                                               end_datetime=utc_naive_end_datetime)
        else:  # viewを用いる場合
            query_df = self._get_query_df_view(stock_names=stock_names,
                                               view=view,
                                               start_datetime=utc_naive_start_datetime,
                                               end_datetime=utc_naive_end_datetime)

        # 前処理
        query_df = self._df_transform(query_df, transform_dict)

        return query_df

    def search_one(self, stock_names, select_datetime, freq_str="T", to_tokyo=True, view=None):
        """
        ある日時のデータ一つを取得．もしその時間のデータがない場合は，最も値の近い指定の時間より遅いデータが取得される
        stock_names: str or list of str
            検索したい株式の銘柄コード
        select_datetime: datatime.datetime
            取得したい日時
        freq_str:
            取得するDataFrameのindexのサンプル周期
        returns
        -------
        resampled_df: pandas.DataFrame
            取得されたDataFrame
        """
        if isinstance(stock_names, str):  # 一応一つだけのとき
            stock_names = [stock_names]
            
        transform_dict = {"freq_str":freq_str, "to_tokyo":to_tokyo}  # transformに渡す辞書
                    
        # 探索範囲の時刻の取得
        next_datetime = get_next_datetime(select_datetime, freq_str, has_mod=True)  # aware or local naive

        # utcのnaiveなdatetimeに変更
        utc_naive_select_datetime = get_utc_naive_datetime_from_datetime(select_datetime)  # utcのnaiveなdatetimeに変更
        utc_naive_next_datetime = get_utc_naive_datetime_from_datetime(next_datetime)  # utcのnaiveなdatetimeに変更

        # query_dfの取得
        if view is None:  # viewを用いない場合
            query_df = self._get_query_df_join(stock_names=stock_names,
                                               start_datetime=utc_naive_select_datetime, 
                                               end_datetime=utc_naive_next_datetime)
        else:  # viewを用いる場合
            query_df = self._get_query_df_view(stock_names=stock_names,
                                               view=view,
                                               start_datetime=utc_naive_select_datetime,
                                               end_datetime=utc_naive_next_datetime)

        # DataFrameの前処理
        query_df = self._df_transform(query_df, transform_dict)
            
        return query_df 
                
    def search_iter(self, stock_names, from_datetime, freq_str="T", to_tokyo=True, view=None):
        """
        指定した時間を基準として，データフレームを一つ一つ返すジェネレーター．
        毎回サーバーにアクセスすることに注意!

        stock_names: str or list of str
            検索したい株式の銘柄コード
        from_datetime: datatime.datetime
            基準としたい日時
        freq_str:
            取得するDataFrameのindexのサンプル周期
        Returns
        -------
        ジェネレーター
        """          
        if isinstance(stock_names, str):  # 一応一つだけのとき
            stock_names = [stock_names]
            
        transform_dict = {"freq_str":freq_str, "to_tokyo":to_tokyo}  # transformに渡す辞書
                    
        temp_from_datetime = from_datetime        

        while True:
            # freqを考慮した次の時間の計算
            temp_next_datetime = get_next_datetime(temp_from_datetime, freq_str, has_mod=True)

            # utcのnaiveなdatetimeに変更
            utc_naive_from_datetime = get_utc_naive_datetime_from_datetime(temp_from_datetime)  # utcのnaiveなdatetimeに変更
            utc_naive_next_datetime = get_utc_naive_datetime_from_datetime(temp_next_datetime)  # utcのnaiveなdatetimeに変更

            # query_dfの取得
            if view is None:  # viewを用いない場合
                query_df = self._get_query_df_join(stock_names=stock_names,
                                                   start_datetime=utc_naive_from_datetime, 
                                                   end_datetime=utc_naive_next_datetime)
            else:  # viewを用いる場合
                query_df = self._get_query_df_view(stock_names=stock_names,
                                                   view=view,
                                                   start_datetime=utc_naive_from_datetime,
                                                   end_datetime=utc_naive_next_datetime)
            # DataFrameの前処理
            query_df = self._df_transform(query_df, transform_dict)
            yield query_df
            
            #temp_from_datetimeの更新
            temp_from_datetime = temp_next_datetime

    def search_span_all_iter(self, 
                             stock_names,
                             start_datetime=None,
                             end_datetime=None, 
                             row_max=500,
                             column_max=100,
                             is_end_include=False, 
                             to_tokyo=True,
                             view=None
                             ):
        """
        期間を指定してデータを取得するジェネレーター，データを移すときなどに利用する．出力は定めた上限値の行数・列数のデータフレーム
        stock_names: str or list of str
            検索したい株式，インデックスの銘柄コードあるいはティッカーコード
        start_datetime: datatime.datetime
            取得を開始する日時，対応している値を含む
        end_datetime: datetime.datetime, default None
            取得を終了する日時，対応している値を含むかどうかはis_end_includeで指定する
        row_max: int, default:100
            一度に取得するデータフレームの行数
        column_max: int, default:500
            一度に取得するデータフレームの列数
        is_end_include:
            end_timeの時の値を含むかどうか

        returns
        -------
        ジェネレーター
        """ 
        #import pdb; pdb.set_trace()  # search_span_all_iter

        if isinstance(stock_names, str):
            stock_names = [stock_names]

        # 範囲指定にNoneが与えられた場合の範囲指定(タイムゾーンが同じであるように変更)
        if start_datetime is None or end_datetime is None:  # どちらかが与えられなかった場合
            stock_timestamp_df = self.stock_timestamp(stock_names, to_tokyo=False)  # UTCであることに注意
        
        if start_datetime is None and end_datetime is not None:  # start_datetimeのみがNoneの場合
            min_datetimes = stock_timestamp_df.loc[:,"min_datetime"].valus()
            start_datetime = np.amin(min_datetimes)
            start_datetime = get_timezone_datetime_like(start_datetime, end_datetime)
        elif start_datetime is not None and end_datetime is None:  # end_datetimeのみがNoneの場合
            max_datetimes = stock_timestamp_df.loc[:,"max_datetime"].values()
            end_datetime = np.amax(max_datetimes)
            end_datetime = get_timezone_datetime_like(end_datetime, start_datetime)
        elif start_datetime is None and end_datetime is None:  # start_datetime, end_datetimeがどちらもNone
            min_datetimes = stock_timestamp_df.loc[:,"min_datetime"].valus()
            start_datetime = np.amin(min_datetimes)  # UTCであることに注意
            max_datetimes = stock_timestamp_df.loc[:,"max_datetime"].values()
            end_datetime = np.amax(max_datetimes)  # UTCであることに注意


        if is_end_include and end_datetime is not None:  # is_end_includeを指定する場合
            end_datetime = get_next_datetime(end_datetime, freq_str=self.database_frequency)

        # 取得予定の全銘柄コードのndarray．このarrayから利用した分だけ削除していく
        all_column_array = np.array(stock_names)

        # 全て終了したときのフラッグ
        IsEndAll = False

        while True:
            # このcolumnsが終了したときのフラッグ
            IsEndThisColumns = False

            temp_start_datetime = start_datetime

            # 探索するカラムを指定
            if len(all_column_array) >= column_max:  # 長さがcolumn_maxよりも大きな場合 
                temp_use_column_array = all_column_array[:column_max]
                # all_column_arrayから利用する分を削除
                all_column_array = all_column_array[column_max:].copy()
            else:
                temp_use_column_array = all_column_array  # 残り全部
                IsEndAll = True

            while True:
                added_datetime = add_datetime(temp_start_datetime, 
                                              freq_str=self.database_frequency,
                                              add_number=row_max
                                              )
                if added_datetime > end_datetime:  # 範囲を上回ったとき
                    temp_end_datetime = end_datetime
                    IsEndThisColumns = True  # 終了フラッグ
                else:
                    temp_end_datetime = added_datetime  # row_max分追加

                if temp_start_datetime == temp_end_datetime:  # 同じになってしまった場合
                    break
                
                query_df = self.search_span(temp_use_column_array.tolist(),  # リストに変換
                                            start_datetime=temp_start_datetime,
                                            end_datetime=temp_end_datetime,
                                            freq_str=self.database_frequency,
                                            is_end_include=False,
                                            to_tokyo=to_tokyo,
                                            view=view,
                                            )
                if query_df is not None:  # Noneが返ってくることもあるため(空白の期間)
                    yield query_df
                
                # temp_start_datetimeを更新
                temp_start_datetime = temp_end_datetime
                
                if IsEndThisColumns:
                    break
            if IsEndAll:
                break

    def make_db(self,
                db_path,
                stock_names,
                start_datetime=None,
                end_datetime=None, 
                row_max=500,
                column_max=100,
                to_tokyo=False,
                view=None,
                print_progress=True
                ):

        """
        期間を指定してデータを別のデータベースに保存する
        db_path: str or Path
            新しく保存するデータベースのパス
        stock_names: str or list of str
            検索したい株式，インデックスの銘柄コードあるいはティッカーコード
        start_datetime: datatime.datetime
            取得を開始する日時，対応している値を含む
        end_datetime: datetime.datetime, default None
            取得を終了する日時，対応している値を含むかどうかはis_end_includeで指定する
        row_max: int, default:100
            一度に取得するデータフレームの行数
        column_max: int, default:500
            一度に取得するデータフレームの列数
        is_end_include:
            end_timeの時の値を含むかどうか

        returns
        -------
        new_db: StockDataBase
        """
        
        if isinstance(stock_names, str):
            stock_names = [stock_names]

        # データベースがすでに存在していた場合
        if db_path.exists():
            raise Exception("this db_path is already exists, please use another db_path")

        new_stock_db = StockDatabase(db_path, database_frequency=self.database_frequency)

        # 範囲のデータのジェネレーター
        all_stock_data_gen = self.search_span_all_iter(stock_names,
                                                       start_datetime=start_datetime,
                                                       end_datetime=end_datetime,
                                                       row_max=row_max,
                                                       column_max=column_max,
                                                       to_tokyo=to_tokyo,
                                                       view=view
                                                      )

        for i,stock_df in enumerate(all_stock_data_gen):
            if print_progress:
                print("\rcount :{}".format(i),end="")
            new_stock_db.upsert(stock_df, item_replace_type="replace_null")

        return new_stock_db

    def _df_tolist_simple(self, df):
        """
        DataFrameをリストに変えるメソッド
        """
        #df_columns = list(set(df.columns))
        df_columns = list(df.columns)
        
        df_list = []
        # ここが時間がかかる
        for i in df.index:
            column_list = [i.to_pydatetime()]  # 最初はindexをdatetime.datetimeにしていれる
            column_list.extend([df.loc[i,j] for j in df_columns]) # columnのリスト化
            df_list.append(column_list)
            #print("column_list:",column_list)

        return df_list
    
    def _df_tolist_numpy(self, df):
        """
        DataFrameをndarrayに変えるメソッド．こちらの方が高速
        """
        #columns_name = list(set(df.columns))
        # indexをndarrayとして取得
        
        index_array = df.index.to_pydatetime()  # datetime.datetimeのndarrayになる
        
        df_values = df.values
        df_array = np.concatenate([index_array[:,None], df_values], axis=1)
        #df_list = df_array.tolist()
        #return df_list
        return df_array

    def db_len(self, freq_str="T"):
        """
        データベースの長さを出力する．指定したfreq_strに応じて，小数点以下も存在する．
        """
        max_counter = 0
        with closing(self.make_connection()) as conn:
            c = conn.cursor()
            sql = "select table_name from column_count"
            c.execute(sql)
            fetch_list = c.fetchall()
            table_name_array = np.array(fetch_list).squeeze(1)  # データの成形 (1darrayを保つ)
            
            for table_name in table_name_array:
                counter = 0
                sql = "select * from {}".format(table_name)  # 全要素を取得
                c.execute(sql)
                for i, row in enumerate(c):
                    counter += 1
    
                if max_counter < counter:
                    max_counter = counter
            
        # データベースの足から，比率を計算
        ratio_of_counter = get_sec_from_freq(freq_str) / get_sec_from_freq(self.database_frequency) 
        if ratio_of_counter < 1.0:
            raise ValueError("This database is not suitable for frequency {}. This frequency is smaller than database one".format(freq_str))
        return_counter = max_counter / ratio_of_counter 
       
        return return_counter
    
    def table_delete(self, table_name):
        """
        指定したテーブルを削除する．滅多に使うべきではない．
        """
        with closing(self.make_connection()) as conn:
            c = conn.cursor()
            c.execute("drop table {}".format(table_name))
            
    def sammary_column_arrange(self):
        """
        column_arrangeテーブルの内容を出力する．行数はカラムの数なので，多すぎる場合は注意
        column_name, arrange_table
            column_name: カラム名
            arrange_table: そのカラムが存在するテーブル名   
        """
        with closing(self.make_connection()) as conn:
            sql ="select * from column_arrange"
            sammary_df = pd.read_sql_query(sql,
                                     con=conn,
                                    )
            
        return sammary_df
    
    def sammary_column_count(self):
        """
        column_countテーブルの内容を出力する．
        table_name, table_number, count
            table_name: テーブル名
            table_number: そのテーブルの番号 最大値が現在書き込めるテーブル
            count: そのテーブルが含むカラムの数，最大値はself.column_upper_limit
        """
        with closing(self.make_connection()) as conn:
            sql ="select * from column_count"
            sammary_df = pd.read_sql_query(sql,
                                     con=conn,
                                    )
            
        return sammary_df
    
    @staticmethod
    def _get_all_contert_gen_each_table(db_path, table_name, column_max, row_max):
        """
        各テーブルについて，すべてのデータを返すジェネレーターを返す．メモリのためcolumn_maxとrow_maxは慎重に
        データベース全体をロードするのに用いるため，クラスメソッドにしておく(対応していないデータベースでも読み込めるため)
        db_path: pathlib.Path, str
            データベースのパス
        table_name: str
            テーブル名
        column_max: int 
            一度に取り出すカラムの最大量
        row_max: int
            一度に取り出す行の最大量
        """
        db_path = Path(db_path)
        def make_conn():
            conn = sqlite3.connect(db_path,
                                   detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
                                  )
            return conn
        
        IsTableStart = True  # テーブルの開始時間かどうか
        IsFinal = False  # 終了時のフラッグ 
        
        with closing(make_conn()) as conn:
            c = conn.cursor()
            # カラムのリストを取得
            c.execute("pragma table_info('{}')".format(table_name))
            column_array = np.array(c.fetchall())[:,1]  # 二番目がtable_name
            column_array = column_array[column_array != "timestamp"]  # timestamp以外を取得
            
            # 開始時間の取得
            sql = "select timestamp from {} order by timestamp asc limit 1".format(table_name)  # dataを昇順に取得
            c.execute(sql)
            table_start_time = c.fetchone()[0]  # 各テーブルにおける開始時間
            
            # 実際に取り出すカラムのarray
            if len(column_array) <= column_max:  # 同じ長さでもエラー
                search_column_names = column_array  # 最後まで
                IsFinal = True
            else:
                search_column_names = column_array[:row_max]
                column_array = np.setdiff1d(column_array, search_column_names)  # 削除しておく
                
        while True:
            if IsTableStart:  # テーブルの開始時間かどうか
                query_operator = ">="
            else:
                query_operator = ">"
                
            if not IsTableStart:
                start_time = query_df.index[-1].to_pydatetime()  # datetime.datetimeに変換
            else:
                start_time = table_start_time
            
            with closing(make_conn()) as conn:
                sql = """
                select timestamp, {} from {} where timestamp {} ? limit {}
                """.format(", ".join(search_column_names), table_name, query_operator, row_max)
                
                query_df = pd.read_sql_query(sql,
                     con=conn,
                     index_col="timestamp",
                     params= [start_time, ]
                    )
                
                if len(query_df.index) < 1:  # dataが取得できなかった場合
                    if IsFinal:
                        return None  # エラーを出して終了
                    if len(column_array) <= column_max:  # 同じ長さでもエラー
                        search_column_names = column_array  # 最後まで, column_arrayの削除は別にいい
                        IsFinal = True
                    else:
                        search_column_names = column_array[:column_max]  # 取得カラムの修正
                        column_array = np.setdiff1d(column_array, search_column_names)  # 削除しておく
                    IsTableStart = True  # 一応上のif文から出しておく
                else:
                    IsTableStart = False
                
                yield query_df

    
    @classmethod
    def get_all_content_gen(cls, db_path, has_arrange_count=True, column_max=100, row_max=200, table_list=None):
        """
        すべてのデータを返すジェネレーターを返す．メモリのためcolumn_maxとrow_maxは慎重に
        データベース全体をロードするのに用いるため，クラスメソッドにしておく(対応していないデータベースでも読み込めるため)

        db_path: pathlib.Path, str
            データベースのパス
        has_arrange_count: bool
            データベースがこのクラスに対応しているかどうか，対応している場合，table_listは与えなくてよい
        column_max: int 
            一度に取り出すカラムの最大量
        row_max: int
            一度に取り出す行の最大量
        table_list: list of str
            テーブル名のリスト．データベースがこのクラスに対応していない場合の時，指定する．
        """
        def make_conn():
            conn = sqlite3.connect(db_path,
                                   detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
                                  )
            return conn
            
        if not has_arrange_count and table_list is None:
            raise ValueError("table_list is not given")

        # table list の作成
        if has_arrange_count and table_list is None:
            with closing(make_conn()) as conn:
                c = conn.cursor()
                sql = "select table_name from column_count"
                c.execute(sql)
                fetch_list = c.fetchall()
                table_array = np.array(fetch_list).squeeze(1)  # データの成形 (1darrayを保つ)
                
        else:
            with closing(make_conn()) as conn:
                c = conn.cursor()
                # テーブルの存在確認
                c.execute("select * from sqlite_master")
                fetch_list = c.fetchall()
                table_name_list = [i[1] for i in fetch_list]  # 二つ目がtable_name
    
            if not set(table_list) <= set(table_name_list):
                raise ValueError("this table list is invalid")
            table_array = np.array(table_list)
      
        for table_name in table_array:
            each_table_gen = cls._get_all_contert_gen_each_table(db_path, table_name, column_max, row_max)  # ジェネレーターの作成
            
            for return_df in each_table_gen:
                # タイムゾーン指定をする．UTCとする
                if len(return_df.index) > 0:
                    return_df.index = return_df.index.tz_localize("UTC")
                yield return_df
                

    def load_other_db(self, db_path, has_arrange_count=True, column_max=100, row_max=500, table_list=None, item_replace_type="replace_null"):
        """
        他のデータベースからロードする．このクラスに対応していなくてもよい．
        db_path: pathlib.Path, str
            データベースのパス
        has_arrange_count: bool
            データベースがこのクラスに対応しているかどうか，対応している場合，table_listは与えなくてよい
        column_max: int 
            一度に取り出すカラムの最大量
        row_max: int
            一度に取り出す行の最大量
        table_list: list of str
            テーブル名のリスト．データベースがこのクラスに対応していない場合の時，指定する．
        item_replace_type: ["replace", "nothing", "replace_null"] defalt, "replace_null"
            すでにレコードがあったときにの挙動
            "replace":更新する
            "nothing":何もしない
            "replace_null":値がヌルの場合，更新する       
        """
        load_other_gen = StockDatabase.get_all_content_gen(db_path, has_arrange_count, column_max, row_max, table_list)
        for i,return_df in enumerate(load_other_gen):
            print("\rcount :{}, row:{} times column:{}".format(i, row_max, column_max),end="")
            self.upsert(return_df, item_replace_type)
    
    def sammary_timestamp(self, to_tokyo=False):
        """
        各テーブル毎に最近の時間と最も古い時間を出力する
        table_name, min_datetime, max_datetime
        to_tokyo: bool
            timezoneを東京にするかどうか
        """
        with closing(self.make_connection()) as conn:
            c = conn.cursor()
            sql = "select table_name from column_count"
            c.execute(sql)
            fetch_list = c.fetchall()
            table_array = np.array(fetch_list).squeeze(1)  # データの成形 (1darrayを保つ)
        
        max_datetime_list = []
        min_datetime_list = []
        
        for table_name in table_array:
            with closing(self.make_connection()) as conn:
                c = conn.cursor()
                max_sql = "select max(timestamp) from {}".format(table_name)
                c.execute(max_sql)
                max_datetime = c.fetchone()[0]
                
                min_sql = "select min(timestamp) from {}".format(table_name)
                c.execute(min_sql)
                min_datetime = c.fetchone()[0]

                utc_timezone = timezone("UTC")
                jst_timezone = timezone("Asia/Tokyo")

                if max_datetime is not None:  # Noneでなかったらstring
                    max_datetime = datetime.datetime.strptime(max_datetime, "%Y-%m-%d %H:%M:%S")
                    max_datetime = utc_timezone.localize(max_datetime)  # UTCに指定

                    if to_tokyo:
                        max_datetime = max_datetime.astimezone(jst_timezone)  # jstに変換

                if min_datetime is not None:  # Noneでなかったらstring
                    min_datetime = datetime.datetime.strptime(min_datetime, "%Y-%m-%d %H:%M:%S")
                    min_datetime = utc_timezone.localize(min_datetime)  # UTCに指定

                    if to_tokyo:
                        min_datetime = min_datetime.astimezone(jst_timezone)  # jstに変換

                max_datetime_list.append(max_datetime)
                min_datetime_list.append(min_datetime)
                
        return_df = pd.DataFrame({"table_name":table_array,
                                  "min_datetime":min_datetime_list,
                                  "max_datetime":max_datetime_list
                                 })
        
        return return_df

    def sammary_stocklist(self):
        """
        データベースに含まれるすべての銘柄コードのndarrayの出力
        """
        with closing(self.make_connection()) as conn:
            sql ="select column_name from column_arrange"
            c = conn.cursor()
            c.execute(sql)
            fetch_array = np.array(c.fetchall()).squeeze(1)  # 1d-arrayとして保持

            def strip_ohlcv(column_name):
                column_name = column_name.lstrip("Open_")  # Open_を削除
                column_name = column_name.lstrip("High_")  # High_を削除
                column_name = column_name.lstrip("Low_")  # Low_を削除
                column_name = column_name.lstrip("Close_")  # Close_を削除
                column_name = column_name.lstrip("Volume_")  # Volume_を削除
                return column_name
            
            np_strip_ohlcv = np.frompyfunc(strip_ohlcv, 1, 1)  # ユニバーサル関数へ変換
            striped_array = np_strip_ohlcv(fetch_array)  # ユニバーサル関数を利用してstrip
        stock_list_array = np.unique(striped_array)
        return stock_list_array

    def stock_in(self, stock_names, use_memorize=True):
        """
        指定した銘柄コードがデータベースに入っているかどうかをboolで出力．
        use_memorize=Trueとすると
        最初以外はデータベースを検索しないので，高速になるが，最新情報ではない
        """
        if isinstance(stock_names, str):  # 一応一つだけのとき
            stock_names = [stock_names]

        if self.exists_stock_names_array is None or not use_memorize:  # 最初の場合かuse_memorize=Trueの場合
            exists_stock_names_array = self.sammary_stocklist()
            self.exists_stock_names_array = exists_stock_names_array
        
        stock_names_array = np.array(stock_names)
        bool_array = np.in1d(stock_names_array, self.exists_stock_names_array)

        if len(bool_array) == 1:
            return bool_array.item()  # 一つのときはbooleanのみを返す

        return bool_array

    def stock_timestamp(self, stock_names, to_tokyo=False):
        """
        指定した銘柄コードの時間の最近の時間と最も古い時間をpandas.DataFrameとして出力

        stock_names: str or list of str
            検索したい株式の銘柄コード
        to_tokyo: bool
            timezoneを東京にするかどうか
        """
        if isinstance(stock_names, str):  # 一応一つだけのとき
            stock_names = [stock_names]
        
        # 探すカラム名のリスト    
        search_column_names = make_ohlcv_column_names(stock_names)
        # {table_name:column_names_array}
        table_name_column_names_dict = self._get_table_name_column_names_dict(search_column_names, insert_new_column=False)        

        # 返すpandas.DataFrameの初期化
        return_df = pd.DataFrame([[None, None] for i in range(len(stock_names))], columns=["min_datetime", "max_datetime"])
        return_df["column_name"] = pd.Series(stock_names)  # column_name列の追加

        for table_name_key in table_name_column_names_dict.keys():
            # max_datetimeとmin_datetimeを取得
            with closing(self.make_connection()) as conn:
                c = conn.cursor()
                max_sql = "select max(timestamp) from {}".format(table_name_key)
                c.execute(max_sql)
                max_datetime = c.fetchone()[0]
                
                min_sql = "select min(timestamp) from {}".format(table_name_key)
                c.execute(min_sql)
                min_datetime = c.fetchone()[0]

                utc_timezone = timezone("UTC")
                jst_timezone = timezone("Asia/Tokyo")

                if max_datetime is not None:  # Noneでなかったらstring
                    max_datetime = datetime.datetime.strptime(max_datetime, "%Y-%m-%d %H:%M:%S")
                    max_datetime = utc_timezone.localize(max_datetime)  # UTCに指定

                    if to_tokyo:
                        max_datetime = max_datetime.astimezone(jst_timezone)  # jstに変換

                if min_datetime is not None:  # Noneでなかったらstring
                    min_datetime = datetime.datetime.strptime(min_datetime, "%Y-%m-%d %H:%M:%S")
                    min_datetime = utc_timezone.localize(min_datetime)  # UTCに指定

                    if to_tokyo:
                        min_datetime = min_datetime.astimezone(jst_timezone)  # jstに変換

            # return_dfを対応するcolumn_namesを持つカラムだけ変更
            column_names_array = table_name_column_names_dict[table_name_key]
            def strip_ohlcv(column_name):
                column_name = column_name.lstrip("Open_")  # Open_を削除
                column_name = column_name.lstrip("High_")  # High_を削除
                column_name = column_name.lstrip("Low_")  # Low_を削除
                column_name = column_name.lstrip("Close_")  # Close_を削除
                column_name = column_name.lstrip("Volume_")  # Volume_を削除
                return column_name
 
            np_strip_ohlcv = np.frompyfunc(strip_ohlcv, 1, 1)  # ユニバーサル関数へ変換
            striped_column_names_array = np_strip_ohlcv(column_names_array)  # ユニバーサル関数を利用してstrip

            unique_column_names_array = np.unique(striped_column_names_array)  # 集合に変換
            return_df.loc[return_df["column_name"].isin(unique_column_names_array),"min_datetime"] = min_datetime  # min_datetimeの書き換え
            return_df.loc[return_df["column_name"].isin(unique_column_names_array),"max_datetime"] = max_datetime  # max_datetimeの書き換え
    
        return return_df

    @staticmethod
    def _df_timezone_convert_for_sqlite(df):
        """
        pandas.DataFrameのタイムゾーンをsqlite用(UTCのnaiveなdatetime)に整えて出力する．
        与えられていない場合，UTCとみなされる．
        """
        if df.index.tzinfo is None:  # 与えられていない場合
            warnings.warn("This datetime index is not localized. setted automatically UTC")
            df.index = df.index.tz_localize("UTC")  # utcと仮定(apiがutcを前提とするため)
        df.index = df.index.tz_convert(None)  # UTCのnaiveなdatetimeに変換
        return df
    
    def _df_transform(self, df, transform_dict):
        """
        検索結果のpandas.DataFrameを整える．リサンプリングとタイムゾーンの指定を行う
        """
        # Noneだった場合
        if df is None:
            return None  # Noneを返す
        # リサンプル
        if transform_dict["freq_str"] == self.database_frequency or len(df.index) < 2:  # 分足あるいは長さが0の場合はそのまま, 以下でもいいかもただ，一つしか取れなかったときにエラーをだしたい？
            df.index.freq_str = self.database_frequency  # データベース自体のfrequencyの文字列を追加
        else:
            freq_converter = ConvertFreqOHLCV(transform_dict["freq_str"])  # patternに注意!
            df = freq_converter(df)  # リサンプル
        
        if len(df.index) > 0:  # 値が正しく返ってきた場合
        # タイムゾーンを指定 
            df.index = df.index.tz_localize("UTC")   # タイムゾーンをutcと指定 
            if transform_dict["to_tokyo"]:
                df.index = df.index.tz_convert("Asia/Tokyo")    
        return df

    def close_all_view(self):
        """
        全てのviewをcloseする．
        """
        view_closier_list = ViewClosier.make_view_closier_list(self)
        [view_closier.close() for view_closier in view_closier_list]


if __name__ == "__main__":
    
    ##########################
    ### StockDatabase
    ##########################

    from get_stock_price import YahooFinanceStockLoaderMin

    db_path = Path("sample_db2") / Path("sample.db")
    stock_db = StockDatabase(db_path)

    stock_names = ["4755.T","6502.T"]

    stockloader = YahooFinanceStockLoaderMin(stock_names, past_day=5)
    stock_df = stockloader.load()

    stock_db.upsert(stock_df, item_replace_type="replace_null")
    stock_names = ["4755","6502"]

    start_time = datetime.datetime(2020, 10, 27, 0, 0, 0)  # 適当な値に設定
    end_time = datetime.datetime(2020, 10, 27, 5, 0, 0)  # 適当な値に設定

    selected_df = stock_db.search_span(
                                    stock_names=stock_names,
                                    start_datetime=start_time,
                                    end_datetime=None,
                                    )
    print(selected_df.tail(5))

    stock_names = "4755"
    one_time = datetime.datetime(2020, 10, 27, 0, 1, 0)  # 適当な値に設定

    selected_df = stock_db.search_one(
                                    stock_names=stock_names,
                                    select_datetime=one_time,
                                    )

    print(selected_df)

    stock_names = "4755"
    from_time = datetime.datetime(2020, 10, 27, 0, 1, 0)

    df_gen = stock_db.search_iter(
                                  stock_names=stock_names,
                                  from_datetime=from_time
                                 )



    print(next(df_gen))
    print(next(df_gen))
