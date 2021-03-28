class StructureStrictList(list):
    def __init__(self, *args):
        super(StructureStrictList, self).__init__(args)
        self._get_structure()  # 構造を取得
        
    def _get_structure(self):
        self._type_structure = self._get_type_structure(self)
        self._length_structure = self._get_length_structure(self)
        
    @staticmethod
    def _get_type_structure(list_like):
        
        # 型が全て等しいかチェック
        # flattenジェネレーター
        def flatten_type_gen(_list):
            for item in _list:
                if isinstance(item, list):
                    yield from flatten_type_gen(item)
                else:
                    yield type(item)
                    
        type_list = list(flatten_type_gen(list_like))
        if len(set(type_list)) == 0:  # リストがからの場合
            raise Exception("list like object is enmty")
        if len(set(type_list)) > 1:  # リストの型の種類が一つ以上の場合
            raise Exception("list like object have to have same type items")
            
        # それぞれの型が全て等しいかチェック
        def item_have_same_type(item):
            if isinstance(item, list):
                type_list = []
                for item_child in item:
                    type_list.append(type(item_child))
                    if isinstance(item_child, list):
                        item_have_same_type(item_child)
                if len(type_list)!=0 and len(set(type_list))!=1:
                    raise Exception("list like object have to have same type recursive")
          
        item_have_same_type(list_like)  # 長さが違う場合エラーがでる．
        
        def type_dicision(item):
            if isinstance(item, list):  # リストの場合
                return [type_dicision(item[0])]
            else:
                return type(item)
        return type_dicision(list_like)
    
    @staticmethod
    def _get_length_structure(list_like):
        # 長さが全て等しいかチェック
        def item_have_same_length(item):
            if isinstance(item, list):
                length_list = []
                for item_child in item:
                    if isinstance(item_child, list):
                        length_list.append(len(item_child))
                        item_have_same_length(item_child)
                if len(length_list)!=0 and len(set(length_list))!=1:
                    raise Exception("list like object have to have same length recursive")
          
        item_have_same_length(list_like)  # 長さが違う場合エラーがでる．
        
        def length_dicision(item, structure_dict):
            if isinstance(item, list):  # リストの場合
                inner_structure_dict = {}
                inner_structure_dict[len(item)]=length_dicision(item[0],{})
                return inner_structure_dict
            else:
                return None
                
        
        all_structure_dict = length_dicision(list_like, {})
        return all_structure_dict
        
    @staticmethod
    def _check_same_type_structure(structure1, structure2):
        def same_structure_dicision(item1, item2):
            if isinstance(item1, list) and isinstance(item2, list):  #どちらもリスト
                return same_structure_dicision(item1[0], item2[0])
            elif (not isinstance(item1, list)) and (not isinstance(item2, list)):  # どちらもリストでない
                return isinstance(item1,type(item2))  # 型が同一か判定
            else:  # どちらかがリストでない
                return False
        return same_structure_dicision(structure1, structure2)
    
    @staticmethod
    def _check_same_length_structure(structure1, structure2):
        def same_structure_dicision(dict1, dict2):
            if (dict1 is not None) and (dict2 is not None):  # どちらもNoneでなかった場合     
                one_key1 = list(dict1.keys())[0]
                value1 = dict1[one_key1]

                one_key2 = list(dict2.keys())[0]
                value2 = dict2[one_key2]

                #キーが一致しない場合Falseを返す
                if one_key1 != one_key2:
                    return False
                return same_structure_dicision(value1, value2)
            elif (dict1 is None) and (dict2 is None):  # どちらもNone
                return True  # いままでFalseでなかったのでTrue
            else:  # どちらかのバリューがNone
                return False  # 構造が異なるため
        return same_structure_dicision(structure1, structure2)
    
    def check_same_structure_with(self, list_like, include_outer_length=False):
        is_same_type_structure = self._check_same_type_structure(self._type_structure, self._get_type_structure(list_like))
        list_like_length_structure = self._get_length_structure(list_like)
        if include_outer_length: # 一番外側の比較も行う
            is_same_length_structure = self._check_same_length_structure(self._length_structure,
                                                                         list_like_length_structure
                                                                        ) 

        else:  # 一番外側の比較は行わない
            is_same_length_structure = self._check_same_length_structure(self._length_structure[list(self._length_structure.keys())[0]],
                                                                         list_like_length_structure[list(list_like_length_structure.keys())[0]]
                                                                        )

        return is_same_type_structure and is_same_length_structure
        
    def check_item_structure(self, item):
        # itemがリストの場合
        if isinstance(item, list):
            is_same_type_structure = self._check_same_type_structure(self._type_structure[0], self._get_type_structure(item))
            is_same_length_structure = self._check_same_length_structure(self._length_structure[list(self._length_structure.keys())[0]],
                                                                         self._get_length_structure(item))
        else:
            if not isinstance(self._type_structure[0], list):
                is_same_type_structure = isinstance(item, self._type_structure[0])
            else:
                is_same_type_structure = False
            is_same_length_structure = True
        
        return is_same_type_structure and is_same_length_structure
    
    @classmethod
    def from_list(cls, _list):
        return cls(*_list)
    
    @classmethod
    def from_structures(cls, _type_structure, _length_structure):
        # 空の自身を作成
        instance = cls(None)
        super(StructureStrictList, instance).remove(None)  # 親のスーパークラスのメソッド呼び出し
        instance._type_structure = _type_structure
        instance._length_structure = _length_structure
        return instance
    
    @property
    def type_structure(self):
        return self._type_structure
        
    @property
    def length_structure(self):
        return self._length_structure
    
    def append(self, item):
        if not self.check_item_structure(item):
            raise Exception("this item is restricted for append")
        super(StructureStrictList, self).append(item)
        self._get_structure()
        
    def extend(self, iterable):
        if not self.check_same_structure_with(list(iterable), include_outer_length=False):  # 外側の長さの比較は行わない
            raise Exception("this iterable is restricted for extend")
        super(StructureStrictList, self).extend(iterable)
        self._get_structure()
        
    def insert(self, i, item):
        if not self.check_item_structure(item):
            raise Exception("this item is restricted for insert")
        super(StructureStrictList, self).insert(i, item)
        self._get_structure()
        
    def remove(self, *args, **kwargs):
        super(StructureStrictList, self).remove(*args, **kwargs)
        self._get_structure()
        
    def pop(self, *args, **kwargs):
        super(StructureStrictList, self).pop(*args, **kwargs)
        self._get_structure() 



class TypeStrictList(StructureStrictList):
    def _get_structure(self):
        self._type_structure = self._get_type_structure(self)
    
    def check_same_structure_with(self, list_like, include_outer_length=True):
        is_same_type_structure = self._check_same_type_structure(self._type_structure, self._get_type_structure(list_like))
        return is_same_type_structure
    
    def check_item_structure(self, item):
        # itemがリストの場合
        if isinstance(item, list):
            is_same_type_structure = self._check_same_type_structure(self._type_structure[0], self._get_type_structure(item))
        else:
            if not isinstance(self._type_structure[0], list):
                is_same_type_structure = isinstance(item,self._type_structure[0])
            else:
                is_same_type_structure = False
        
        return is_same_type_structure
    
    @classmethod
    def from_type_structure(cls, _type_structure):
        instance = cls(None)
        super(StructureStrictList, instance).remove(None)  # 親のスーパークラスのメソッド呼び出し
        instance._type_structure = _type_structure
        return instance


class LengthStrictList(StructureStrictList):
    def _get_structure(self):
        self._length_structure = self._get_length_structure(self)
       
    def check_same_structure_with(self, list_like, include_outer_length=True):
        list_like_length_structure = self._get_length_structure(list_like)
        if include_outer_length: # 一番外側の比較も行う
            is_same_length_structure = self._check_same_length_structure(self._length_structure,
                                                                         list_like_length_structure
                                                                        ) 

        else:  # 一番外側の比較は行わない
            is_same_length_structure = self._check_same_length_structure(self._length_structure[list(self._length_structure.keys())[0]],
                                                                         list_like_length_structure[list(list_like_length_structure.keys())[0]]
                                                                        )

        return is_same_length_structure
    
    def check_item_structure(self, item):
        # itemがリストの場合
        if isinstance(item, list):
            is_same_length_structure = self._check_same_length_structure(self._length_structure[list(self._length_structure.keys())[0]],
                                                                         self._get_length_structure(item))
        else:
            is_same_length_structure = True
        
        return is_same_length_structure
    
    @classmethod
    def from_length_structure(cls, _length_structure):
        instance = cls(None)
        super(StructureStrictList, instance).remove(None)  # 親のスーパークラスのメソッド呼び出し
        instance._length_structure = _length_structure
        return instance



if __name__ == "__main__":
    pass