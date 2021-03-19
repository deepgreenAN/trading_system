# イテレーション時に進捗を保存
イテレーションが何らかの例外によって途中で終了した場合，その進捗状況を一時ファイルに保存し，再度実行時にそこから開始できる．

### 使い方

```python
from pathlib import Path
from py_restart import enable_counter, multi_count, simple_counter
```

#### 一つの場合 

以下のように，`enable_counter`をwith文に添えた返り値(`CounterClosier`オブジェクト)でイテレーターをラップする．イテレーション内でエラーが生じた場合に，一時ファイルを保存し，次回はエラーが起きたイテレーションから再開できる．
この例では，iが4のときにKeybordInterruptを行った後，もう一度実行した結果である．


```python
tempfile_path = Path("temp1.tmp")

with enable_counter(tempfile_path) as counter:
    for i in counter(range(10)):
        print(i)
        time.sleep(3)
```

    4
    5
    6
    7
    8
    9
    

#### 一つの場合(毎回保存する場合) 

with文を利用したくない場合，`enable_counter`の引数`each_save`をTrueにするか，`simple_couonter`が利用できる．どちらも異常終了時に一時ファイルを保存するわけではなく，イテレーションの指定回数ごとに保存する．また，`simple_counter`は直接ジェネレータを出力する．


```python
tempfile_path = Path("temp2.tmp")

for i in simple_counter(tempfile_path, range(10)):
    print(i)
    time.sleep(3)
```

    4
    5
    6
    7
    8
    9
    

#### 二つ以上の場合 

`enable_counter`あるいは`simple_counter`のみでは，一つのfor文が終了したときに一時ファイルが削除されてしまうため，二つ以上for文が連続する場合に進捗を保存できない．`multi_count`を利用すればそのインデントブロックが終了するまで一時ファイルを残すことができる．以下の例では，一つ目のfor文が終了したのちにiが2の時点でKeybordInterruptを行い，再度実行した結果である


```python
tempfile_path1 = Path("temp3.tmp")
tempfile_path2 = Path("temp4.tmp")

with multi_count():
    with enable_counter(tempfile_path1) as counter:
        for i in counter(range(10)):
            print("1:",i)
            time.sleep(3)
            
    print("1 is finished")
    for i in simple_counter(tempfile_path2, range(5)):
            print("2:",i)
            time.sleep(3)
```

    1 is finished
    2: 2
    2: 3
    2: 4
    

#### 再帰的に使う場合 

以下の例では，iが1,jが2の時にKeybordInterruptを行ったのち，再度実行したものである．


```python
tempfile_path3 = Path("temp5.tmp")
tempfile_path4 = Path("temp6.tmp")

with enable_counter(tempfile_path3) as outer_counter:
    for i in outer_counter(range(3)):
        print("outer:",i)
        for j in simple_counter(tempfile_path4 ,range(5)):
                print("\tinner:",j)
                time.sleep(3)
```

    outer: 1
    	inner: 2
    	inner: 3
    	inner: 4
    outer: 2
    	inner: 0
    	inner: 1
    	inner: 2
    	inner: 3
    	inner: 4
    

#### オブジェクトの一時保存

イテレーションの進捗保存だけでなく，特定のオブジェクトも一時的に保存できる．その場合，以下のように`enable_counter`の返り値`CounterClosier`の`save_load_object`メソッドを利用できる．もちろん`save_load_object`はイテレーション内に記述するべきではないが，withブロック内に記述する必要がある．登録したオブジェクトがイミュータブルな場合，イテレーション途中で`CounterClosier`の`object`プロパティを明示的に変更する．  

この例では，iが4のときにKeybordInterruptを行った後，もう一度実行した結果である．


```python
tempfile_path = Path("temp7.tmp")

# 保存したいオブジェクト
save_object = {"sum":0}

with enable_counter(tempfile_path) as counter:
    # オブジェクトの登録(保存ファイルがある場合の読み込み)
    save_object = counter.save_load_object(save_object, Path("temp_sum.pickle"))
    print(save_object)
    for i in counter(range(10)):
        print("i:",i)
        time.sleep(3)
        
        save_object["sum"] += i
        # 変更を明示する場合，以下のようにする
        #counter.object = save_object
        
        print("sum:",save_object["sum"])
```

    {'sum': 6}
    i: 4
    sum: 10
    i: 5
    sum: 15
    i: 6
    sum: 21
    i: 7
    sum: 28
    i: 8
    sum: 36
    i: 9
    sum: 45
    

毎回保存する場合，つまり`enable_counter`の引数`each_save`を`True`にした場合，with文を用いなくても保存できる．しかしイテレーション毎にpickleで保存するため，データの読み込み・書き出しのオーバーヘッドが加わることに注意する．`enable_counter`の引数`save_span`を指定することで，保存間隔を指定できる．

指定回数ごとに保存することによって，エラーで検知できないような終了(例えば，Google colabの接続切れなど)をしてしまっても，一次ファイルを保存できるメリットがある．


```python
tempfile_path = Path("temp8.tmp")

# 保存したいオブジェクト
save_object = {"sum":0}

counter = enable_counter(tempfile_path, each_save=True, save_span=1)
# オブジェクトの登録(保存ファイルがある場合の読み込み)
save_object = counter.save_load_object(save_object, Path("temp_sum.pickle"))
print(save_object)

for i in counter(range(10)):
    print("i:",i)
    time.sleep(3)

    save_object["sum"] += i
    counter.object = save_object  # 一応明示的に変更

    print("sum:",save_object["sum"])
```

    {'sum': 6}
    i: 4
    sum: 10
    i: 5
    sum: 15
    i: 6
    sum: 21
    i: 7
    sum: 28
    i: 8
    sum: 36
    i: 9
    sum: 45
    

#### 任意の保存・ロード関数の利用 

機械学習における重みファイルの保存など，オブジェクトの保存に外部の関数を利用したい場合がある．その場合は`CounterClosier`の`save_load_funcs`メソッドを利用できる．`save_load_funcs`の引数は`save_funcs`(保存用の関数のリスト),`load_funcs`(読み込み用の関数のリスト)，`func_paths`(二つの関数の引数となるパスのリスト)の3つのリストを対応するように渡す必要がある．保存用の関数・読み込み用の関数，はどちらもパスのみを引数とするため,任意の関数を利用する場合は無名関数などを用いて調節する必要がある．なお，`load_funcs`に与える関数は，保存したいオブジェクトをグローバル変数にして変更する必要があることに注意する．


```python
import torch
import torch.nn as nn
import numpy as np
```


```python
linear_model = nn.Linear(5, 10)
temp_array = np.zeros((2,2))
temp_tensor = torch.zeros((3,3))

# pickleでtensorを書き出す用の関数
def save_temp_tensor_as_pickle(save_path):
    with open(save_path, "wb") as f:
        pickle.dump(temp_tensor, f)
    
# 保存関数のリスト
save_funcs = [lambda save_path: torch.save(linear_model.state_dict(), save_path),
              lambda save_path: np.save(save_path, temp_array),
              save_temp_tensor_as_pickle
             ]

# pytorchのモデルを読み込む用の関数
def load_linear_model(load_path):
    global linear_model  # こちらの宣言は必要ない
    linear_model.load_state_dict(torch.load(load_path))
# ndarrayを読み込む用の関数
def load_temp_array(load_path):
    global temp_array  # 書き換えるため，グローバル変数宣言
    temp_array = np.load(load_path)
# pickleでtensorを読み込む用の関数
def load_temp_tensor(load_path):
    global temp_tensor  # 書き換えるため，グローバル変数宣言
    with open(load_path, "rb") as f:
        temp_tensor = pickle.load(f)

# ロード関数のリスト
load_funcs = [load_linear_model,
              load_temp_array,
              load_temp_tensor,
             ]

#　パスのリスト
func_paths = [Path("temp_linear_model.pth"),
              Path("temp_array.npy"),
              Path("temp_tensor.pickle")]
```

この例では，iが4のときにKeybordInterruptを行った後，もう一度実行した結果である．


```python
tempfile_path = Path("temp9.tmp")

with enable_counter(tempfile_path) as counter:
    # 保存・ロード関数の登録(保存ファイルがある場合は読み込み)
    counter.save_load_funcs(save_funcs=save_funcs,
                            load_funcs=load_funcs,
                            func_paths=func_paths)
    
    print("load temp_array:", temp_array)
    print("load temp_tensor:", temp_tensor)
    for i in counter(range(10)):
        print("i:",i)
        time.sleep(3)
        
        temp_array += i * np.ones((2,2))
        temp_tensor += i * torch.ones((3,3))
        print("temp_array:", temp_array)
        print("temp_tensr:", temp_tensor)
```

    load temp_array: [[6. 6.]
     [6. 6.]]
    load temp_tensor: tensor([[6., 6., 6.],
            [6., 6., 6.],
            [6., 6., 6.]])
    i: 4
    temp_array: [[10. 10.]
     [10. 10.]]
    temp_tensr: tensor([[10., 10., 10.],
            [10., 10., 10.],
            [10., 10., 10.]])
    i: 5
    temp_array: [[15. 15.]
     [15. 15.]]
    temp_tensr: tensor([[15., 15., 15.],
            [15., 15., 15.],
            [15., 15., 15.]])
    i: 6
    temp_array: [[21. 21.]
     [21. 21.]]
    temp_tensr: tensor([[21., 21., 21.],
            [21., 21., 21.],
            [21., 21., 21.]])
    i: 7
    temp_array: [[28. 28.]
     [28. 28.]]
    temp_tensr: tensor([[28., 28., 28.],
            [28., 28., 28.],
            [28., 28., 28.]])
    i: 8
    temp_array: [[36. 36.]
     [36. 36.]]
    temp_tensr: tensor([[36., 36., 36.],
            [36., 36., 36.],
            [36., 36., 36.]])
    i: 9
    temp_array: [[45. 45.]
     [45. 45.]]
    temp_tensr: tensor([[45., 45., 45.],
            [45., 45., 45.],
            [45., 45., 45.]])
    

###  エラーとなる処理

以下のように，`multi_count`は再帰的に利用できない


```python
with multi_count():
    with multi_count():
        pass
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-56-3b3e455f2f62> in <module>
          1 with multi_count():
    ----> 2     with multi_count():
          3         pass
    

    <ipython-input-32-06c8301f69c3> in __enter__(self)
          8     def __enter__(self):
          9         if config.parent is not None:
    ---> 10             raise Exception("MultiCount has already opend. cannot open another MultiCount")
         11 
         12         config.parent = self.parent_restart
    

    Exception: MultiCount has already opend. cannot open another MultiCount