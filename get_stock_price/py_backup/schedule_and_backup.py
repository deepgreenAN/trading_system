import csv
from pathlib import Path
import datetime
import shutil
import time

import zipfile


def make_zip(source_path, zip_path):
    """
    指定したパスのファイル・ディレクトリをzipファイルにして保存する関数
    source_path: pathlib.Path
        ソースとなるファイル・ディレクトリのパス
    zip_path: pathlib.Path
        保存するzipファイルのパス
    """
    source_path = Path(source_path)
    zip_path =zip_path

    if zip_path.suffix != ".zip":
        raise ValueError("zip_path must be zipfile")
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_LZMA) as new_zip:
        if source_path.is_file():  # 一つのファイルの場合
            new_zip.write(filename=source_path, arcname=source_path.name)
            
        elif source_path.is_dir():
            only_source_root_path = Path(source_path.name)  # 対象ディレクトリをzipファイルに含めたい
            def nest_zip(search_path):
                if search_path.is_file():  # ファイルの場合
                    arc_path = only_source_root_path / search_path.relative_to(source_path)  # 対象ディレクトリ + 相対パス
                    new_zip.write(filename=search_path, arcname=arc_path)
                elif search_path.is_dir():
                    for search_path_one in search_path.iterdir():
                        nest_zip(search_path_one)  # ネスト
            
            nest_zip(source_path)


class PyBackUp():
    """
    指定したファイル・フォルダをバックアップする．保存形式は元と同じあるいはzip形式．
    zip形式にはLZMA形式を利用する
    """
    def __init__(self, source_path, backup_path, back_number=6, is_use_text=True, to_zip=False):
        """
        source_path: str or pathlib.Path
            バックアップしたいソースのパス．ファイルでもディレクトリでも良い．
        backup_path: str or pathlib.Path
            バックアップ先のディレクトリのパス．そのディレクトリにsource_pathに対応したフォルダを作成する．
        back_number: int
            バックアップファイルの個数．
        is_use_text: bool
            バックアップファイルの管理にcsvを使うかどうか
        to_zip: bool
            zip形式で保存するかどうか
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise ValueError("This path does not exists")
        self.source_path = source_path
        self.source_name = source_path.name  # ファイル名
        self.source_stem = source_path.stem  # 拡張子を除いたファイル名

        backup_path = Path(backup_path)
        if not backup_path.exists():
            backup_path.mkdir()

        self.backup_path = backup_path

        self.back_number = back_number
        self.backup_counter = -1  # 0からスタートするように
        
        self.to_zip = to_zip
        if is_use_text:
            self.read_backup_data()

    def back_up(self):
        print("[{}] backup start.".format(str(datetime.datetime.now())))
        self.backup_counter += 1
        
        backup_number = int((self.backup_counter)%self.back_number)  # 保存するディレクトリに対応
        backup_dir_name = "back_up_" + str(backup_number)

        backup_dir_path = self.backup_path / Path(backup_dir_name)
        backup_dst_path = backup_dir_path / Path(self.source_name)  # 実際に保存するパス

        if not backup_dir_path.exists():  # バックアップファイルのディレクトリが存在しない場合
            backup_dir_path.mkdir(parents=True)  # ディレクトリを作成
            
        if backup_dst_path.exists():  # すでにバックアップファイルが存在する場合
            if backup_dst_path.is_file():  # ファイルの場合
                backup_dst_path.unlink()  # 削除
            elif backup_dst_path.is_dir():  # ディレクトリの場合
                shutil.rmtree(backup_dst_path)

        # バックアップファイルのコピー
        if self.to_zip:  # zip
            backup_dst_path = backup_dst_path.with_suffix(".zip")  # zipとつける
            make_zip(source_path=self.source_path, zip_path=backup_dst_path)
            
        else: #zipでなくコピー
            if self.source_path.is_file():
                shutil.copyfile(src=self.source_path, dst=backup_dst_path)
            elif self.source_path.is_dir():
                shutil.copytree(src=self.source_path, dst=backup_dst_path)
        
        backup_data_text_path = backup_dir_path / Path("data.csv")
        if not backup_data_text_path.exists():  # バックアップデータの詳細を書いたテキストファイル
            backup_data_text_path.touch(exist_ok=True)

        # バックアップデータの書き込み・書き換え
        with open(backup_data_text_path, "w", newline="") as f:
            writer = csv.writer(f)
            backup_time = datetime.datetime.now()
            writer.writerow(["date", backup_time.strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["dir_number", backup_number])
            writer.writerow(["backup_count", self.backup_counter])


        print("[{}] back up db_file {}".format(str(backup_time),str(backup_number)))
        print("[{}] backup end.".format(str(datetime.datetime.now())))
        return backup_dst_path

    def read_backup_data(self):
        backup_datetime_list = []
        backup_counter_list = []

        for backup_dir in self.backup_path.iterdir():
            # バックアップファイルの存在確認
            if self.to_zip: #Zipの場合
                backup_file_path = backup_dir / Path(self.source_stem).with_suffix(".zip")  # xz.tarを前提
            else:
                backup_file_path = backup_dir / Path(self.source_name)  
                
            if backup_file_path.exists():  # バックアップファイルが存在する場合
                backup_data_text_path = backup_dir / Path("data.csv")
                if backup_data_text_path.exists():
                    # バックアップデータの読み込み
                    with open(backup_data_text_path, "r") as f:
                        reader = csv.reader(f)
                        
                        datetime_list = next(reader)
                        backup_datetime = datetime.datetime.strptime(datetime_list[1],"%Y-%m-%d %H:%M:%S")
                        
                        next(reader)  # この行はいらなかったかも

                        counter_list = next(reader)
                        backup_counter_list.append(counter_list[1])

                        backup_datetime_list.append(backup_datetime)

        # バックアップデータが存在する場合
        if len(backup_datetime_list) > 0:
            # 最近のインデックスを求める, maxカウンターでもいいけど念のため
            def get_timestamp(datetime):
                return datetime.timestamp()
            max_date = max(backup_datetime_list, key=get_timestamp)
            max_date_index = backup_datetime_list.index(max_date)
            
            self.backup_counter = int(backup_counter_list[max_date_index])


if __name__ == "__main__":
    import argparse
    import schedule
    
    print("[{}] schedule program start".format(str(datetime.datetime.now())))
    parser = argparse.ArgumentParser(description='back up by shchedule')

    parser.add_argument("source_path", help="ソースファイル・ディレクトリ")
    parser.add_argument("backup_path", help="バックアップディレクトリ")
    parser.add_argument("--zip", action="store_true", help="zipファイルにするかどうか")
    parser.add_argument("--number", help="バックアップファイルの個数", type=int, default=5)
    parser.add_argument("--days", help="バックアップの間隔日数", type=int, default=7)

    args = parser.parse_args()

    source_path = Path(args.source_path)
    backup_path = Path(args.backup_path)

    pybackup = PyBackUp(source_path=source_path,
                        backup_path=backup_path,
                        back_number=args.number,
                        to_zip=args.zip
                        )
    
    schedule.every(5).minutes.do(pybackup.back_up)

    #if args.days != 1:
    #    schedule.every(args.days).days.at("12:00").do(pybackup.back_up)
    #else:
    #    schedule.every().day.at("12:00").do(pybackup.back_up)

    while True:
        schedule.run_pending()
        time.sleep(1)  # 一秒単位で更新
