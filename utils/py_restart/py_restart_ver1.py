import csv
from pathlib import Path
import datetime


class Config():
    """
    各クラス・関数が参照する変数をまとめたクラス
    """
    def __init__(self):
        self.parent = None
        self.is_enable = False

config = Config()


class MultiClosier():
    """
    複数のカウンタを利用する時に，このクロージャの終了時にまとえてtempfileを削除するためのクロージャ
    """
    def __init__(self, parent_restart):
        self.parent_restart = parent_restart
        
    def __enter__(self):
        if config.parent is not None:
            raise Exception("MultiCount has already opend. cannot open another MultiCount")
        if config.is_enable:
            raise Exception("Counter has already opened. please close Counter")
        
        config.parent = self.parent_restart
        return self
        
    def __exit__(self, ex_type, ex_value, trace):
        config.parent = None  # 共通して行う
        if ex_type is None:# 正常終了した場合
            self.parent_restart.all_close()
        return False


class CounterClosier():
    """
    イテレーションの進捗をtempfileに保存するクロージャ，自身によってイテレータをラップする．
    途中で例外によって終了した場合と親が存在する場合にファイルを残す．
    """
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                reader = csv.reader(f)
                #dateについて取得, 現在時間との差が一日以内かどうか判定
                datetime_list = next(reader)  # [datetime,実際の日時の文字列]
                tempfile_datetime = datetime.datetime.strptime(datetime_list[1], "%Y-%m-%d %H:%M:%S")
                if datetime.datetime.now() - tempfile_datetime >= datetime.timedelta(days=1):
                    print("tempfile is not recent date, please check tempfile")

                # スタートカウンターの読み込み
                start_counter_list = next(reader)
                self.start_counter = int(start_counter_list[1])
        else:
            self.start_counter = 0
        self.counter = 0  # 一応こちらでも0に初期化
    
    def __call__(self, iterable):  # ジェネレーターを返す
        iterable = iter(iterable)
        self.counter = 0  # カウンタの初期化
        while True:
            if self.counter < self.start_counter:
                self.counter += 1
                try:
                    next(iterable)  # 利用しない．進めるだけ
                except StopIteration:
                    return None  # StopIterationで終了
                continue
            
            try:
                yield_item = next(iterable)  # iterableから一つ取得
                yield yield_item
            except:
                return None  # StopIteration
            
            self.counter += 1  # すべてが終了したら+1
    
    def __enter__(self):
        config.is_enable = True
        return self
    
    def write(self):
        with open(self.file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["datetime", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["start_count",self.counter])
            
    def __exit__(self, ex_type, ex_value, trace):
        if ex_type is not None:  # エラーで終了した場合 
            self.write()
        else:  # 正常に終了した場合
            if config.parent is None:  # ペアレントが無い場合
                if self.file_path.exists():
                    self.file_path.unlink()
            else:  # ペアレントが存在する場合
                self.write()
        
        config.is_enable = False
        return False


class CounterClosierThrough():
    """
    CounterClosierを模した何もしないクロージャ．実装を変えずにtempfileの使用・不使用を切り替えるために利用する
    """
    def __init__(self, file_path):
        self.file_path = Path(file_path)
    
    def __call__(self, iterable):
        return iterable
    
    def __enter__(self):
        config.is_enable = True
        return self
    

    def __exit__(self, ex_type, ex_value, trace):   
        #ファイルが存在したら削除
        if self.file_path.exists():
            self.file_path.unlink()
        
        config.is_enable = False
        return False


class RestartChild():
    """
    子供としてCounterClosierを渡すためのクラス
    """
    def __init__(self, file_path, use_tempfile=True):
        self.file_path = Path(file_path)
        self.use_tempfile = use_tempfile
        
    def enable(self):
        if self.use_tempfile:
            return CounterClosier(self.file_path)
        else:
            return CounterClosierThrough(self.file_path)


class RestartParent():
    """
    config.parentを変更し，親から子供を作るためにクラス
    """
    def __init__(self):
        self.child_list = []
        
    def create_child(self, file_path, use_tempfile=True):
        """
        親から子供を作成する
        """
        restart_child = RestartChild(file_path, use_tempfile)
        self.child_list.append(restart_child)
        return restart_child
        
    def multi_child(self):
        return MultiClosier(self)
    
    def all_close(self):
        """
        子のtempfileを削除する．
        """
        [child.file_path.unlink() for child in self.child_list if child.file_path.exists()]  # フォイルが存在している場合は，削除


def multi_count():
    """
    複数カウンタを作成するときに展開することで，tempfileの削除をすべての終了タイミングに変更できる．
    """
    parent = RestartParent()
    return parent.multi_child()

def enable_counter(file_path, use_tempfile=True):
    if config.parent is None:  #ペアレントが存在しない場合
        one_child_parent = RestartParent()  # 一つのchildを持つペアレントを作成
        child = one_child_parent.create_child(file_path, use_tempfile)
        return child.enable()
    
    else:
        child = config.parent.create_child(file_path)
        return child.enable()


if __name__ == "__main__":
    pass