# 構造の厳密なリスト 

プロパティとしてリストを採用する際などで，変更時に型チェックできる．セッターなどで利用するときも簡単に型チェックできるようにした．基本的に構造はコンストラクタで確定される．(構造から空リストを作ることもできる．)


## 使い方
ワーキングディレクトリにクローンするかパスを通せば使える． 

```python
from py_strict_list import StructureStrictList, TypeStrictList, LengthStrictList
```

## 型・長さ構造が厳密なリスト


```python
a = StructureStrictList([1,2],["a", "b"])
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-2-905093049281> in <module>
    ----> 1 a = StructureStrictList([1,2],["a", "b"])
    

    E:\py_strict_list\py_strict_list_ver1.py in __init__(self, *args)
          2     def __init__(self, *args):
          3         super(StructureStrictList, self).__init__(args)
    ----> 4         self._get_structure()  # 構造を取得
          5 
          6     def _get_structure(self):
    

    E:\py_strict_list\py_strict_list_ver1.py in _get_structure(self)
          5 
          6     def _get_structure(self):
    ----> 7         self._type_structure = self._get_type_structure(self)
          8         self._length_structure = self._get_length_structure(self)
          9 
    

    E:\py_strict_list\py_strict_list_ver1.py in _get_type_structure(list_like)
         24             raise Exception("list like object is enmty")
         25         if len(set(type_list)) > 1:  # リストの型の種類が一つ以上の場合
    ---> 26             raise Exception("list like object have to have same type items")
         27 
         28         # それぞれの型が全て等しいかチェック
    

    Exception: list like object have to have same type items



```python
a = StructureStrictList([1,2],[3])
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-3-337e9e91693a> in <module>
    ----> 1 a = StructureStrictList([1,2],[3])
    

    E:\py_strict_list\py_strict_list_ver1.py in __init__(self, *args)
          2     def __init__(self, *args):
          3         super(StructureStrictList, self).__init__(args)
    ----> 4         self._get_structure()  # 構造を取得
          5 
          6     def _get_structure(self):
    

    E:\py_strict_list\py_strict_list_ver1.py in _get_structure(self)
          6     def _get_structure(self):
          7         self._type_structure = self._get_type_structure(self)
    ----> 8         self._length_structure = self._get_length_structure(self)
          9 
         10     @staticmethod
    

    E:\py_strict_list\py_strict_list_ver1.py in _get_length_structure(list_like)
         59                     raise Exception("list like object have to have same length recursive")
         60 
    ---> 61         item_have_same_length(list_like)  # 長さが違う場合エラーがでる．
         62 
         63         def length_dicision(item, structure_dict):
    

    E:\py_strict_list\py_strict_list_ver1.py in item_have_same_length(item)
         57                         item_have_same_length(item_child)
         58                 if len(length_list)!=0 and len(set(length_list))!=1:
    ---> 59                     raise Exception("list like object have to have same length recursive")
         60 
         61         item_have_same_length(list_like)  # 長さが違う場合エラーがでる．
    

    Exception: list like object have to have same length recursive



```python
a = StructureStrictList([1,2],[3,4])
a
```




    [[1, 2], [3, 4]]




```python
a.length_structure
```




    {2: {2: None}}




```python
a.type_structure
```




    [[int]]



### 他のSSLとの比較


```python
b = StructureStrictList(3,4)
a.check_same_structure_with(b)
```




    False



### 他のリストとの比較


```python
c = [[5,6],[7,8]]
a.check_same_structure_with(c)
```




    True



### 要素との比較 

appendとかの型判定で利用


```python
a.check_item_structure([1,2])
```




    True




```python
a.check_item_structure([3])
```




    False



### append 


```python
a.append(1)
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-12-da0a5ad497c3> in <module>
    ----> 1 a.append(1)
    

    E:\py_strict_list\py_strict_list_ver1.py in append(self, item)
        157     def append(self, item):
        158         if not self.check_item_structure(item):
    --> 159             raise Exception("this item is restricted for append")
        160         super(StructureStrictList, self).append(item)
        161         self._get_structure()
    

    Exception: this item is restricted for append



```python
a.append([5,6])
a
```




    [[1, 2], [3, 4], [5, 6]]



###  extend


```python
a.extend([[7,8],[9,10],[11,12]])
a
```




    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]



### from_list 

リストから作成する場合


```python
d = StructureStrictList.from_list([1,2,3])
d
```




    [1, 2, 3]




```python
d.length_structure
```




    {3: None}




```python
d.type_structure
```




    [int]



## 型が厳密なリスト 


```python
a = TypeStrictList(["a","b"],[1,2])
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-19-e265aa80b45c> in <module>
    ----> 1 a = TypeStrictList(["a","b"],[1,2])
    

    E:\py_strict_list\py_strict_list_ver1.py in __init__(self, *args)
          2     def __init__(self, *args):
          3         super(StructureStrictList, self).__init__(args)
    ----> 4         self._get_structure()  # 構造を取得
          5 
          6     def _get_structure(self):
    

    <ipython-input-18-7d3d3a17c8e2> in _get_structure(self)
          1 class TypeStrictList(StructureStrictList):
          2     def _get_structure(self):
    ----> 3         self._type_structure = self._get_type_structure(self)
          4 
          5     def check_same_structure_with(self, list_like, include_outer_length=True):
    

    E:\py_strict_list\py_strict_list_ver1.py in _get_type_structure(list_like)
         24             raise Exception("list like object is enmty")
         25         if len(set(type_list)) > 1:  # リストの型の種類が一つ以上の場合
    ---> 26             raise Exception("list like object have to have same type items")
         27 
         28         # それぞれの型が全て等しいかチェック
    

    Exception: list like object have to have same type items



```python
a = TypeStrictList(["a","b"],"c")
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-20-2ef8de3b788c> in <module>
    ----> 1 a = TypeStrictList(["a","b"],"c")
    

    E:\py_strict_list\py_strict_list_ver1.py in __init__(self, *args)
          2     def __init__(self, *args):
          3         super(StructureStrictList, self).__init__(args)
    ----> 4         self._get_structure()  # 構造を取得
          5 
          6     def _get_structure(self):
    

    <ipython-input-18-7d3d3a17c8e2> in _get_structure(self)
          1 class TypeStrictList(StructureStrictList):
          2     def _get_structure(self):
    ----> 3         self._type_structure = self._get_type_structure(self)
          4 
          5     def check_same_structure_with(self, list_like, include_outer_length=True):
    

    E:\py_strict_list\py_strict_list_ver1.py in _get_type_structure(list_like)
         37                     raise Exception("list like object have to have same type recursive")
         38 
    ---> 39         item_have_same_type(list_like)  # 長さが違う場合エラーがでる．
         40 
         41         def type_dicision(item):
    

    E:\py_strict_list\py_strict_list_ver1.py in item_have_same_type(item)
         35                         item_have_same_type(item_child)
         36                 if len(type_list)!=0 and len(set(type_list))!=1:
    ---> 37                     raise Exception("list like object have to have same type recursive")
         38 
         39         item_have_same_type(list_like)  # 長さが違う場合エラーがでる．
    

    Exception: list like object have to have same type recursive



```python
a = TypeStrictList(["a","b"],["d"])
```


```python
a.type_structure
```




    [[str]]



### append 


```python
a.append("a")
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-25-be817fbe9230> in <module>
    ----> 1 a.append("a")
    

    E:\py_strict_list\py_strict_list_ver1.py in append(self, item)
        157     def append(self, item):
        158         if not self.check_item_structure(item):
    --> 159             raise Exception("this item is restricted for append")
        160         super(StructureStrictList, self).append(item)
        161         self._get_structure()
    

    Exception: this item is restricted for append



```python
a.append(["e"])
a
```




    [['a', 'b'], ['d'], ['e']]



### structureから空のリストを作成する場合 


```python
b = TypeStrictList.from_type_structure([str])
```


```python
b.type_structure
```




    [str]




```python
b.append(["c"])
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-446-ea8bf38fa590> in <module>
    ----> 1 b.append(["c"])
    

    <ipython-input-419-c7e8eca190d4> in append(self, item)
        148     def append(self, item):
        149         if not self.check_item_structure(item):
    --> 150             raise Exception("this item is restricted for append")
        151         super(StructureStrictList, self).append(item)
        152         self._get_structure()
    

    Exception: this item is restricted for append



```python
b.append("a")
b
```




    ['a']



## 長さが厳密なリスト

使用用途は不明


```python
a = LengthStrictList([1,2,3],[1,2])
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-30-2fd35d765fe9> in <module>
    ----> 1 a = LengthStrictList([1,2,3],[1,2])
    

    E:\py_strict_list\py_strict_list_ver1.py in __init__(self, *args)
          2     def __init__(self, *args):
          3         super(StructureStrictList, self).__init__(args)
    ----> 4         self._get_structure()  # 構造を取得
          5 
          6     def _get_structure(self):
    

    E:\py_strict_list\py_strict_list_ver1.py in _get_structure(self)
        213 class LengthStrictList(StructureStrictList):
        214     def _get_structure(self):
    --> 215         self._length_structure = self._get_length_structure(self)
        216 
        217     def check_same_structure_with(self, list_like, include_outer_length=True):
    

    E:\py_strict_list\py_strict_list_ver1.py in _get_length_structure(list_like)
         59                     raise Exception("list like object have to have same length recursive")
         60 
    ---> 61         item_have_same_length(list_like)  # 長さが違う場合エラーがでる．
         62 
         63         def length_dicision(item, structure_dict):
    

    E:\py_strict_list\py_strict_list_ver1.py in item_have_same_length(item)
         57                         item_have_same_length(item_child)
         58                 if len(length_list)!=0 and len(set(length_list))!=1:
    ---> 59                     raise Exception("list like object have to have same length recursive")
         60 
         61         item_have_same_length(list_like)  # 長さが違う場合エラーがでる．
    

    Exception: list like object have to have same length recursive



```python
a = LengthStrictList([2,3],[1,2])
a
```




    [[2, 3], [1, 2]]


