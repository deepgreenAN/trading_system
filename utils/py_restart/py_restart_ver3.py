import csv
from pathlib import Path
import datetime

import pickle
import shutil
import warnings

class Config():
    """
    各クラス・関数が参照する変数をまとめたクラス
    """
    def __init__(self):
        self.parent = None

config = Config()


class MultiClosier():
    """
    複数のカウンタを利用する時に，このクロージャの終了時にまとめてtempfileを削除するためのクロージャ
    """
    def __init__(self, parent):
        self.parent = parent
        
    def __enter__(self):
        if config.parent is not None:
            raise Exception("MultiCount has already opend. cannot open another MultiCount")
        
        config.parent = self.parent
        return self
        
    def __exit__(self, ex_type, ex_value, trace):
        config.parent = None  # 共通して行う
        if ex_type is None:# 正常終了した場合
            self.parent.all_close()
        return False


class CounterClosier():
    """
    イテレーションの進捗をtempfileに保存するクロージャ，自身によってイテレータをラップする．
    途中で例外によって終了した場合と親が存在する場合にファイルを残す．
    """
    def __init__(self, file_path, parent, each_save=False, save_span=1):
        """
        file_path: pathlib.Path
            一時ファイルのパス
        parent: ParentCounter
            自身の親を意味するクラス
        each_save: bool
            指定回数ごとに保存するかどうか
        save_span: int, default:1
            指定価数ごとに保存する場合の，指定回数
        """
        self.file_path = file_path
        self.parent = parent
        self.each_save = each_save
        self.save_span = save_span
        
        # 保存オブジェクト・保存関数の初期化
        self.object = None
        self.object_path = None
        self.load_funcs = []
        self.save_funcs = []
        self.func_paths = []
        self.src_dsts = []
        
        
        # 一時ファイルの読み込み
        self.start_counter = self._read_tempfile()
        self.counter = 0  # 一応こちらでも0に初期化
        
    def _read_tempfile(self):
        """
        一時ファイルの読み込み
        """
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                reader = csv.reader(f)
                #dateについて取得, 現在時間との差が一日以内かどうか判定
                datetime_list = next(reader)  # [datetime,実際の日時の文字列]
                tempfile_datetime = datetime.datetime.strptime(datetime_list[1], "%Y-%m-%d %H:%M:%S")
                if datetime.datetime.now() - tempfile_datetime >= datetime.timedelta(days=1):
                    warnings.warn("tempfile is not recent date, please check tempfile")

                # スタートカウンターの読み込み
                start_counter_list = next(reader)
                start_counter = int(start_counter_list[1])
        else:
            start_counter = 0
        
        return start_counter
    
    def _load_object(self):
        """
        オブジェクト保存ファイルの読み込み
        """
        new_object = self.object  # とりあえず現在のオブジェクトとする
        
        if self.object_path is not None:
            if self.object_path.exists():
                with open(self.object_path, "rb") as f:
                    new_object = pickle.load(f)
                    
        return new_object
    
    def _save_object(self):
        """
        オブジェクト保存ファイルの書き出し
        """
        if self.object is not None:
            with open(self.object_path, "wb") as f:
                pickle.dump(self.object, f)
                
    def save_load_object(self, obj, obj_path):
        """
        オブジェクトの保存についての設定と読み込み．保存ファイルが存在しなかった場合は，引数のオブジェクトがそのまま返る．
        obj: any
            保存するオブジェクト
        obj_path: path
            保存するパス
        """
        self.object = obj
        self.object_path = obj_path
        
        # オブジェクト保存ファイルの読み込み
        self.object = self._load_object()
        
        return self.object
    
    def _load_funcs(self):
        """
        ロード関数による読み込み
        """
        for func_path, load_func in zip(self.func_paths, self.load_funcs):           
            if func_path.exists():
                load_func(func_path)  # load関数の実行
                
    def _save_funcs(self):
        """
        セーブ関数による保存
        """
        for func_path, save_func in zip(self.func_paths, self.save_funcs):
            save_func(func_path)  # save関数の実行
    
    def save_load_funcs(self, save_funcs, load_funcs, func_paths):
        """
        関数による保存についての設定と読み込み
        save_funcs: list of function
            保存する関数のリスト．各関数の引数はfunc_pathsの対応するパスとする．
        load_funcs: list of function
            ロードする関数のリスト．各関数の引数はfunc_pathsの対応するパスとする．
        func_paths: list of pathlib.Path
            保存・ロードするパスのリスト
        """
        arg_lists =[save_funcs, load_funcs, func_paths]
        
        assert all(list(map(lambda arg_list: isinstance(arg_list, list), arg_lists)))  # 皆list
        assert len(set(map(len , arg_lists)))==1  # 皆長さが一致
        
        self.save_funcs = save_funcs
        self.load_funcs = load_funcs
        self.func_paths = func_paths
        
        # 関数保存ファイルの読み込み
        self._load_funcs()
        
    def _write_tempfile(self):
        """
        一時ファイルの書き出し
        """
        with open(self.file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["datetime", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["start_count",self.counter])
            
    def save(self):
        """
        一時ファイル・保存ファイルを保存
        """
        self._write_tempfile()
        self._save_object()
        self._save_funcs()
        
    def remove_files(self):
        """
        利用した一時ファイル・保存ファイルをすべて削除する
        """
        # 一時ファイルの削除
        if self.file_path.exists():
            self.file_path.unlink()
        # 保存ファイルの削除
        if self.object_path is not None:
            if self.object_path.exists():
                self.object_path.unlink()
        for func_path in self.func_paths:
            if func_path.exists():
                # ディレクトリかファイルか判定
                if func_path.is_file():
                    func_path.unlink()
                elif func_path.is_dir():
                    shutil.rmtree(func_path)
    
    def _iter_finish(self):
        """
        イテレーションが終了したときの処理．親が存在するならばファイルを削除せず保存する．
        """
        if self.parent is None:  # ペアレントが無い場合
            self.remove_files()  # 一時ファイル・保存ファイルの削除
        else:  # ペアレントが存在する場合
            self.save()  # 一時ファイル・保存ファイルの保存
    
    def __call__(self, iterable):  # ジェネレーターを返す
        """
        for文にラップするための関数
        iterable: イテラブルなオブジェクト
        
        return
        ------
        ジェネレーター
        """
        iterable = iter(iterable)
        self.counter = 0  # カウンタの初期化
        while True:
            if self.counter < self.start_counter:
                self.counter += 1
                try:
                    next(iterable)  # 利用しない．進めるだけ
                except StopIteration:
                    self._iter_finish()
                    return None  # StopIterationで終了
                continue
            
            try:
                yield_item = next(iterable)  # iterableから一つ取得
                yield yield_item
            except StopIteration:
                self._iter_finish()
                return None  # StopIterationで終了
            
            self.counter += 1  # すべてが終了したら+1
            if self.each_save:  # 一時ファイルを指定回数ごとに保存
                if self.counter%self.save_span==0:
                    self.save()  # 一時ファイル・保存ファイルの保存
    
    def __enter__(self):
        return self
                
    def __exit__(self, ex_type, ex_value, trace):
        """
        with文が終了した際に，異常終了なら各ファイルを保存し，正常終了ならすでに処理してあるため何もしない
        """
        if ex_type is not None:  # エラーで終了した場合 
            self.save()
        return False


class CounterClosierThrough(CounterClosier):
    """
    CounterClosierを模した何もしないクロージャ．実装を変えずにtempfileの使用・不使用を切り替えるために利用する
    """
    def __call__(self, iterable):
        """
        ファイルを削除して，イテレータをそのまま返す
        """
        # このタイミングで，一時ファイル・保存ファイルを削除
        self.remove_files()
        # そのままイテレータを返す
        return iterable
    
    def __enter__(self):
        """
        何もしない
        """
        return self

    def __exit__(self, ex_type, ex_value, trace):
        """
        何もしない
        """
        return False


class ParentCounter():
    """
    CounterClosierをまとめる親となるクラス
    """
    def __init__(self):
        self.child_counter_list = []
        
    def create_child(self, file_path, each_save=False, save_span=1, use_tempfile=True):
        """
        自身から子供を作成する
        file_path: pathlib.Path
            一時ファイルのパス
        each_save: bool
            指定回数ごとに保存するかどうか
        save_span: int, default:1
            指定価数ごとに保存する場合の，指定回数
        use_tempfile: bool
            一時ファイルを利用するかどうか
        """
        if use_tempfile:
            counter = CounterClosier(file_path, parent=self, each_save=each_save, save_span=save_span)
        else:
            counter = CounterClosierThrough(file_path, parent=self)
        
        self.child_counter_list.append(counter)
        return counter
        
    @staticmethod
    def create_non_parent_child(file_path, each_save=False, save_span=1, use_tempfile=True):
        """
        file_path: pathlib.Path
            一時ファイルのパス
        each_save: bool
            指定回数ごとに保存するかどうか
        save_span: int, default:1
            指定価数ごとに保存する場合の，指定回数
        use_tempfile: bool
            一時ファイルを利用するかどうか
        """
        if use_tempfile:
            counter = CounterClosier(file_path, parent=None, each_save=each_save, save_span=save_span)
        else:
            counter = CounterClosierThrough(file_path, parent=None)
        return counter
        
    def multi_child(self):
        """
        子供を複数まとめる場合にwith文で展開する
        """
        return MultiClosier(self)
    
    def all_close(self):
        """
        子の一時ファイル・保存ファイルを全て削除する．
        """
        [counter.remove_files() for counter in self.child_counter_list]


def multi_count():
    """
    複数カウンタを作成するときに展開することで，一時ファイル・保存ファイルの削除をすべてが終了したタイミングで行うことができる．．
    """
    parent = ParentCounter()
    return parent.multi_child()


def enable_counter(file_path, use_tempfile=True, each_save=True, save_span=1):
    """
    for文をラップするCounterClosierオブジェクトを返す．with文で展開することで，エラーによる終了時に進捗状況(一時ファイル・保存ファイル)を保存する.
    with文に展開しなくても，each_saveをTrueにすることで，指定したイテレーション回数ごとに保存することもできる．
    
    file_path: pathlib.Path
        一時ファイルのパス
    use_tempfile: bool
        一時ファイル・保存ファイルを利用するかどうか．つまりこれをTrueにすると，利用しないのと全く同じになる
    each_save: bool
        指定回数ごとに保存するかどうか．
    save_span: int
        指定回数ごとに保存する場合の指定回数
    """
    if config.parent is None:  # グローバルのペアレントが存在しない場合
        counter = ParentCounter.create_non_parent_child(file_path, each_save=each_save, save_span=save_span, use_tempfile=use_tempfile)
        return counter
    else:  # グローバルのペアレントが存在する場合
        counter = config.parent.create_child(file_path, each_save=each_save, save_span=save_span, use_tempfile=use_tempfile)
        return counter
    
    
def simple_counter(file_path, iterable, use_tempfile=True, save_span=1):
    """
    for文をラップするジェネレータを直接返す．イテレーションの毎回で保存される．
    file_path: pathlib.Path
        一時ファイルのパス
    iterable: any of itrable
        イテラブルなオブジェクト
    use_tempfile: bool
        一時ファイル・保存ファイルを利用するかどうか．つまりこれをTrueにすると，利用しないのと全く同じになる
    save_span: int
        指定回数ごとに保存する場合の指定回数
    """
    return enable_counter(file_path, use_tempfile, each_save=True, save_span=save_span)(iterable)


if __name__ == "__main__":
    pass