import csv
from pathlib import Path
import datetime
import schedule
import time

import pandas as pd
import shutil
import numpy as np

from get_stock_price import YahooFinanceStockLoaderMin
from get_stock_price import StockDatabase
from get_stock_price.py_backup import PyBackUp

from utils import py_restart


def mytqdm(an_iter):
    """
    tqdmを模したジェネレータ．イテレーション可能なオブジェクトを引数とする．
    an_iter: any of iterable
        進捗度を出力するイテレータ
    """
    length = len(an_iter)
    my_iter = iter(an_iter)
    counter = 0
    start_time = time.time()
    old_start_time = time.time()
    while True:
        try:
            new_start_time = time.time()
            next_iter = next(my_iter)  # 終了時はここでエラーが出る
            counter += 1
            one_take_time = new_start_time - old_start_time
            print("\r{}/{}, [{:.3f} sec]".format(counter,length ,one_take_time), end="")
            old_start_time = new_start_time
            yield next_iter
            
        except StopIteration:  # StopIterationErrorのみ通す
            counter += 1
            end_time = time.time()
            all_take_time = end_time - start_time
            print("\r{}/{}, mean [{:.3f} sec]".format(length, length, all_take_time/counter))
            return None  # StopIterationErrorを起こす


class CsvKobetsuInsert():
    """
    csvファイルで読み込んだ銘柄のリストをもとに，StockDataBaseにデータをインサートしていく．メモリの点から，一つ一つの銘柄ごとにインサートする．
    """
    def __init__(self, csv_path, stock_loader, stock_db, stock_group="nikkei_255", use_tempfile=False):
        """
        csv_path: pathlib.Path
            csvファイルのパス(内容は銘柄コード，銘柄名)
        stock_loader: YahooFinanceStockLoaderMin
            データローダ．今のところYahooFinanceを利用したもののみ
        stock_db: StockDataBase
            データベース
        stock_group: str
            銘柄グループの名前．csvファイルに対応させる
        use_tempfile: bool
            tempfileを利用するかどうか．tempfileを利用すると，プログラムが途中で終了した場合そこからスタートできる．
        """
        self.csv_path = Path(csv_path)
        self.stock_loader = stock_loader
        self.stock_db = stock_db
        self.stock_codes = pd.read_csv(self.csv_path, header=0)  # 自分で作成
        
        self.stock_group = stock_group
        if len(self.stock_codes) < 1:
            print("csv cannot read")
        self.use_tempfile = use_tempfile


    def __call__(self):
        print("[{}] {}_kobetsu_insert start".format(str(datetime.datetime.now()), self.stock_group))

        stock_codes_array = self.stock_codes.loc[:,"code"].values.astype("str")

        tempfile_path = self.csv_path.parent / Path(self.stock_group+".tmp")
        with py_restart.enable_counter(tempfile_path) as counter:
            for stock_code in counter(mytqdm(stock_codes_array)):
                stock_name = str(stock_code) + ".T"  # これはyahooが前提
                self.stock_loader.set_stock_names(stock_name)
                try:
                    df = stock_loader.load()
                except Exception:
                    #ここでエラーは気にしない
                    df = None

                if df is not None:
                    self.stock_db.upsert(df, item_replace_type="replace_null")
                else:
                    print("\n[{}] cannot get {}".format(str(datetime.datetime.now()), stock_code))

        print("[{}] {}_kobetsu_insert end".format(str(datetime.datetime.now()), self.stock_group))


class FunctionComposer():
    """
    callableなオブジェクトをまとめてcallする．
    """
    def __init__(self, function_list):
        """
        function_list: list of function
            関数のリスト
        """
        self.function_list = function_list
        
    def __call__(self):
        """
        全ての関数の実行
        """
        with py_restart.multi_count():
            for func in self.function_list:
                func()  # 関数の実行


if __name__ == "__main__":
    import argparse
    print("[{}] schedule program start".format(str(datetime.datetime.now())))

    parser = argparse.ArgumentParser(description='insert data to database with scheduling')
    parser.add_argument("--tempfile", action="store_true", help="tempfileを利用するかどうか")

    args = parser.parse_args()

    # 必要なインスタンス
    db_path = Path("db/stock_db") / Path("stock.db")
    stock_db = StockDatabase(db_path, column_upper_limit=1000, table_name_base="table")
    nikkei_code_file_path = Path("get_stock_price") / Path("nikkei225.csv")
    tosho_code_file_path = Path("get_stock_price") / Path("tosho.csv")

    stock_loader = YahooFinanceStockLoaderMin(None, past_day=5, stop_time_span=2.0, is_use_stop=False)  #ストップしない

    nikkei_kobetsu_insert = CsvKobetsuInsert(nikkei_code_file_path, stock_loader, stock_db, stock_group="nikkei_255", use_tempfile=args.tempfile)
    tosho_kobetsu_insert = CsvKobetsuInsert(tosho_code_file_path, stock_loader, stock_db, stock_group="tosho_1", use_tempfile=args.tempfile)

    func_composer = FunctionComposer([nikkei_kobetsu_insert, tosho_kobetsu_insert])

    # schedule
    # schedule.every(5).days.at("18:00").do(func_composer)
    schedule.every(3).days.at("12:00").do(func_composer)

    backup_path = Path("backup")
    db_backup = PyBackUp(db_path, backup_path, back_number=5, to_zip=True)

    #schedule.every().minute.at(":00").do(db_backup.back_up)
    schedule.every().sunday.at("03:00").do(db_backup.back_up)

    func_composer()  # 一度やっておく
    db_backup.back_up()  # 一度やっておく
    while True:
        schedule.run_pending()
        time.sleep(60)  # 一分単位で更新
  