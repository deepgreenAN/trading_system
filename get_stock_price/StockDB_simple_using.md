```python
%cd ..
```

    E:\システムトレード入門\predict_git_workspace
    


```python
from pathlib import Path
import datetime
from pytz import timezone
```


```python
from get_stock_price import YahooFinanceStockLoaderMin
```


```python
from get_stock_price import StockDatabase
```

## インスタンスの作成 


```python
db_path = Path("db/test_db") / Path("stock.db")
```


```python
stock_db = StockDatabase(db_path)
```

## upsert 


```python
stock_names  = ["4755.T","6502.T","2802.T","6954.T"]

stockloader = YahooFinanceStockLoaderMin(stock_names, stop_time_span=2.0, is_use_stop=False, to_tokyo=True)
stock_df = stockloader.load()
stock_df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_4755</th>
      <th>High_4755</th>
      <th>Low_4755</th>
      <th>Close_4755</th>
      <th>Volume_4755</th>
      <th>Open_6502</th>
      <th>High_6502</th>
      <th>Low_6502</th>
      <th>Close_6502</th>
      <th>Volume_6502</th>
      <th>Open_2802</th>
      <th>High_2802</th>
      <th>Low_2802</th>
      <th>Close_2802</th>
      <th>Volume_2802</th>
      <th>Open_6954</th>
      <th>High_6954</th>
      <th>Low_6954</th>
      <th>Close_6954</th>
      <th>Volume_6954</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-20 14:55:00+09:00</th>
      <td>1127.0</td>
      <td>1128.0</td>
      <td>1126.0</td>
      <td>1128.0</td>
      <td>41000.0</td>
      <td>2902.0</td>
      <td>2902.0</td>
      <td>2900.0</td>
      <td>2902.0</td>
      <td>24200.0</td>
      <td>2289.5</td>
      <td>2289.5</td>
      <td>2289.0</td>
      <td>2289.5</td>
      <td>6000.0</td>
      <td>24515.0</td>
      <td>24520.0</td>
      <td>24515.0</td>
      <td>24520.0</td>
      <td>1300.0</td>
    </tr>
    <tr>
      <th>2020-11-20 14:56:00+09:00</th>
      <td>1127.0</td>
      <td>1128.0</td>
      <td>1126.0</td>
      <td>1128.0</td>
      <td>53300.0</td>
      <td>2902.0</td>
      <td>2902.0</td>
      <td>2900.0</td>
      <td>2902.0</td>
      <td>15700.0</td>
      <td>2289.0</td>
      <td>2289.5</td>
      <td>2288.0</td>
      <td>2289.5</td>
      <td>7300.0</td>
      <td>24515.0</td>
      <td>24515.0</td>
      <td>24510.0</td>
      <td>24515.0</td>
      <td>2500.0</td>
    </tr>
    <tr>
      <th>2020-11-20 14:57:00+09:00</th>
      <td>1128.0</td>
      <td>1129.0</td>
      <td>1127.0</td>
      <td>1128.0</td>
      <td>20000.0</td>
      <td>2902.0</td>
      <td>2904.0</td>
      <td>2900.0</td>
      <td>2904.0</td>
      <td>23300.0</td>
      <td>2289.5</td>
      <td>2289.5</td>
      <td>2288.5</td>
      <td>2289.0</td>
      <td>4500.0</td>
      <td>24515.0</td>
      <td>24520.0</td>
      <td>24510.0</td>
      <td>24520.0</td>
      <td>1800.0</td>
    </tr>
    <tr>
      <th>2020-11-20 14:58:00+09:00</th>
      <td>1128.0</td>
      <td>1130.0</td>
      <td>1127.0</td>
      <td>1130.0</td>
      <td>76000.0</td>
      <td>2902.0</td>
      <td>2905.0</td>
      <td>2901.0</td>
      <td>2905.0</td>
      <td>27000.0</td>
      <td>2288.5</td>
      <td>2289.0</td>
      <td>2287.5</td>
      <td>2289.0</td>
      <td>8400.0</td>
      <td>24520.0</td>
      <td>24520.0</td>
      <td>24510.0</td>
      <td>24520.0</td>
      <td>4000.0</td>
    </tr>
    <tr>
      <th>2020-11-20 14:59:00+09:00</th>
      <td>1129.0</td>
      <td>1130.0</td>
      <td>1128.0</td>
      <td>1129.0</td>
      <td>94000.0</td>
      <td>2904.0</td>
      <td>2910.0</td>
      <td>2902.0</td>
      <td>2910.0</td>
      <td>56100.0</td>
      <td>2289.0</td>
      <td>2289.0</td>
      <td>2285.5</td>
      <td>2286.0</td>
      <td>31200.0</td>
      <td>24520.0</td>
      <td>24535.0</td>
      <td>24505.0</td>
      <td>24505.0</td>
      <td>13200.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock_db.upsert(stock_df, item_replace_type="replace_null")
```

## search 

データの存在確認


```python
stock_names = ["6502", "4755"]
stock_db.stock_in(stock_names)
```




    array([ True,  True])



データの時間範囲確認．あくまでもテーブルの時間を取得しているので，実際に値が入っているかどうかは不明


```python
stock_db.stock_timestamp(stock_names, to_tokyo=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min_datetime</th>
      <th>max_datetime</th>
      <th>column_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-11-02 09:00:00+09:00</td>
      <td>2020-11-20 14:59:00+09:00</td>
      <td>6502</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-11-02 09:00:00+09:00</td>
      <td>2020-11-20 14:59:00+09:00</td>
      <td>4755</td>
    </tr>
  </tbody>
</table>
</div>



### search_span 

期間を指定してデータを取得．datetimeにはtimezoneを指定する．指定しなかった場合，ローカルのタイムゾーンと判定される．出力データのタイムゾーンはutcであり，to_tokyo=Trueとすると東京時間となる．


```python
stock_names = ["6502", "4755"]

jst_timezone = timezone("Asia/Tokyo")

start_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,9,0,0))
end_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,15,0,0))

query_df = stock_db.search_span(stock_names=stock_names,
                                start_datetime=start_datetime,
                                end_datetime=end_datetime,
                                freq_str="10T",
                                to_tokyo=True
                               )

query_df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_6502</th>
      <th>High_6502</th>
      <th>Low_6502</th>
      <th>Close_6502</th>
      <th>Volume_6502</th>
      <th>Open_4755</th>
      <th>High_4755</th>
      <th>Low_4755</th>
      <th>Close_4755</th>
      <th>Volume_4755</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-18 14:10:00+09:00</th>
      <td>2840.0</td>
      <td>2845.0</td>
      <td>2835.0</td>
      <td>2836.0</td>
      <td>54900.0</td>
      <td>1102.0</td>
      <td>1103.0</td>
      <td>1101.0</td>
      <td>1102.0</td>
      <td>129400.0</td>
    </tr>
    <tr>
      <th>2020-11-18 14:20:00+09:00</th>
      <td>2837.0</td>
      <td>2838.0</td>
      <td>2817.0</td>
      <td>2818.0</td>
      <td>77700.0</td>
      <td>1101.0</td>
      <td>1102.0</td>
      <td>1095.0</td>
      <td>1096.0</td>
      <td>237900.0</td>
    </tr>
    <tr>
      <th>2020-11-18 14:30:00+09:00</th>
      <td>2818.0</td>
      <td>2822.0</td>
      <td>2810.0</td>
      <td>2822.0</td>
      <td>80600.0</td>
      <td>1096.0</td>
      <td>1097.0</td>
      <td>1094.0</td>
      <td>1096.0</td>
      <td>246700.0</td>
    </tr>
    <tr>
      <th>2020-11-18 14:40:00+09:00</th>
      <td>2823.0</td>
      <td>2826.0</td>
      <td>2821.0</td>
      <td>2824.0</td>
      <td>71500.0</td>
      <td>1096.0</td>
      <td>1097.0</td>
      <td>1095.0</td>
      <td>1097.0</td>
      <td>185100.0</td>
    </tr>
    <tr>
      <th>2020-11-18 14:50:00+09:00</th>
      <td>2823.0</td>
      <td>2835.0</td>
      <td>2823.0</td>
      <td>2835.0</td>
      <td>156600.0</td>
      <td>1097.0</td>
      <td>1099.0</td>
      <td>1093.0</td>
      <td>1094.0</td>
      <td>638600.0</td>
    </tr>
  </tbody>
</table>
</div>



### search_one 

与えられた時間から一つ分だけデータを取得．


```python
stock_names = ["4755"]

jst_timezone = timezone("Asia/Tokyo")

select_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,9,0,0))

query_df = stock_db.search_one(stock_names=stock_names,
                               select_datetime=select_datetime,
                               freq_str="10T",
                               to_tokyo=True
                              )

query_df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_4755</th>
      <th>High_4755</th>
      <th>Low_4755</th>
      <th>Close_4755</th>
      <th>Volume_4755</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-18 09:00:00+09:00</th>
      <td>1107.0</td>
      <td>1122.0</td>
      <td>1107.0</td>
      <td>1109.0</td>
      <td>532900.0</td>
    </tr>
  </tbody>
</table>
</div>



### search_itter 

与えた時間から一つ一つ取り出すジェネレータ


```python
stock_names = ["6502", "4755"]

jst_timezone = timezone("Asia/Tokyo")

from_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,9,0,0))

query_gen = stock_db.search_iter(stock_names=stock_names,
                                 from_datetime=from_datetime,
                                 freq_str="10T",
                                 to_tokyo=True
                                )
```


```python
query_df = next(query_gen)
query_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_6502</th>
      <th>High_6502</th>
      <th>Low_6502</th>
      <th>Close_6502</th>
      <th>Volume_6502</th>
      <th>Open_4755</th>
      <th>High_4755</th>
      <th>Low_4755</th>
      <th>Close_4755</th>
      <th>Volume_4755</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-18 09:00:00+09:00</th>
      <td>2819.0</td>
      <td>2823.0</td>
      <td>2788.0</td>
      <td>2796.0</td>
      <td>227800.0</td>
      <td>1107.0</td>
      <td>1122.0</td>
      <td>1107.0</td>
      <td>1109.0</td>
      <td>532900.0</td>
    </tr>
  </tbody>
</table>
</div>



## search (viewを利用する) 

sqliteのviewを利用して，範囲を指定しておくことでデータの増加による検索速度の低下を防ぐ


```python
stock_names = ["4755"]

jst_timezone = timezone("Asia/Tokyo")

start_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,9,0,0))
end_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,15,0,0))


with stock_db.create_view(stock_names, start_datetime=start_datetime, end_datetime=end_datetime) as view:
    query_df = stock_db.search_span(stock_names=stock_names,
                                    start_datetime=start_datetime,
                                    end_datetime=end_datetime,
                                    freq_str="10T",
                                    to_tokyo=True,
                                    view=view
                                   )

query_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_4755</th>
      <th>High_4755</th>
      <th>Low_4755</th>
      <th>Close_4755</th>
      <th>Volume_4755</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-18 09:00:00+09:00</th>
      <td>1107.0</td>
      <td>1122.0</td>
      <td>1107.0</td>
      <td>1109.0</td>
      <td>532900.0</td>
    </tr>
    <tr>
      <th>2020-11-18 09:10:00+09:00</th>
      <td>1110.0</td>
      <td>1114.0</td>
      <td>1104.0</td>
      <td>1106.0</td>
      <td>333900.0</td>
    </tr>
    <tr>
      <th>2020-11-18 09:20:00+09:00</th>
      <td>1105.0</td>
      <td>1107.0</td>
      <td>1094.0</td>
      <td>1095.0</td>
      <td>402000.0</td>
    </tr>
    <tr>
      <th>2020-11-18 09:30:00+09:00</th>
      <td>1095.0</td>
      <td>1097.0</td>
      <td>1092.0</td>
      <td>1096.0</td>
      <td>242100.0</td>
    </tr>
    <tr>
      <th>2020-11-18 09:40:00+09:00</th>
      <td>1095.0</td>
      <td>1102.0</td>
      <td>1095.0</td>
      <td>1099.0</td>
      <td>255100.0</td>
    </tr>
  </tbody>
</table>
</div>



以下はviewの速度の恩恵を受けられる例，検索時間が半分近くに減少している．以下のようにcreate_viewメソッドをwith文を用いずに利用する場合は，view(ViewClosierオブジェクト)をcloseしないといけない．(データベースにviewが残ってしまう)ただ，致命的なエラーが出るわけではないので，後回しでも構わない


```python
stock_names = ["6502", "4755"]
jst_timezone = timezone("Asia/Tokyo")
start_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,9,0,0))
end_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,12,0,0))


from_datetime = jst_timezone.localize(datetime.datetime(2020,11,18,9,0,0))

view = stock_db.create_view(stock_names, start_datetime=start_datetime, end_datetime=end_datetime)
query_gen = stock_db.search_iter(stock_names, 
                                 from_datetime=from_datetime,
                                 freq_str="10T",
                                 to_tokyo=True,
                                 view=view)
```


```python
query_df = next(query_gen)
query_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open_6502</th>
      <th>High_6502</th>
      <th>Low_6502</th>
      <th>Close_6502</th>
      <th>Volume_6502</th>
      <th>Open_4755</th>
      <th>High_4755</th>
      <th>Low_4755</th>
      <th>Close_4755</th>
      <th>Volume_4755</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-18 09:00:00+09:00</th>
      <td>2819.0</td>
      <td>2823.0</td>
      <td>2788.0</td>
      <td>2796.0</td>
      <td>227800.0</td>
      <td>1107.0</td>
      <td>1122.0</td>
      <td>1107.0</td>
      <td>1109.0</td>
      <td>532900.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
view.close()
```
