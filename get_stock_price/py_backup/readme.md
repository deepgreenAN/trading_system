# ファイル・ディレクトリバックアップ

## バックアッププログラムを利用する場合

requirement:
- schedule

args:
- source_path: バックアップしたいファイル・ディレクトリ  
- backup_path: バックアップ先のディレクトリ  
- --zip : zipにするかどうかのフラッグ  
- --number: バックアップファイルの個数  
- --days: バックアップを行う日数の間隔  

使い方例
```
$ python schedule_and_backup.py backup_source/source backup/dir_backup --number 5
```

## 他のスケジューリングプログラムで利用する場合

使い方例
```python
from schedule_and_backup import PyBackUp
from pathlib import Path

source_path = Path("backup_source/source")
backup_path = Path("backup/dir_backup")

pybackup = PyBackUp(source_path=source_path,
                    backup_path=backup_path,
                    back_number=5,
                    to_zip=True
                    )

# バックアップを行うメソッド
pybackup.back_up()
```