import datetime
from pytz import timezone
import pandas as pd
import numpy as np
import unittest


from utils import get_utc_naive_datetime_from_datetime, get_floor_mod_datetime, get_ceil_mod_datetime, get_next_datetime, get_previous_datetime
from utils import get_timezone_datetime_like, get_next_workday_intraday_datetime, get_previous_workday_intraday_datetime

class TestDatetimeFreq(unittest.TestCase):
    def test_utc_naive(self):
        jst_timezone = timezone("Asia/Tokyo")
        jst_datetime = jst_timezone.localize(datetime.datetime(2021, 1, 1, 0, 0, 0))
        utc_naive_datetime = get_utc_naive_datetime_from_datetime(jst_datetime)
        self.assertEqual(utc_naive_datetime, datetime.datetime(2020, 12, 31, 15, 0))
        
        hk_timezone = timezone("Asia/Hong_Kong")
        hk_datetime = hk_timezone.localize(datetime.datetime(2021, 1, 1, 0, 0, 0))
        utc_naive_datetime = get_utc_naive_datetime_from_datetime(hk_datetime)
        self.assertEqual(utc_naive_datetime, datetime.datetime(2020, 12, 31, 16, 0))
        
    def test_floor_ceil(self):
        # get_floor_mod_datetime
        jst = timezone("Asia/Tokyo")
        
        select_datetime = datetime.datetime(2020, 11, 4, 12, 30, 0)
        freq_str = "H"
        floor_datetime = get_floor_mod_datetime(select_datetime, freq_str)
        self.assertEqual(floor_datetime, datetime.datetime(2020, 11, 4, 12, 0))
        
        args = [(datetime.datetime(2020, 11, 4, 12, 30, 0), "H"),
                (jst.localize(datetime.datetime(2020, 11, 4, 12, 0, 0)), "H"),
                (datetime.datetime(2020, 11, 4, 12, 3, 0), "5T"),
                (jst.localize(datetime.datetime(2020, 11, 4, 12, 15, 0)), "30T"),
                (datetime.datetime(2020, 11, 4, 12, 40, 0), "30T"),
               ]
        
        results = [datetime.datetime(2020, 11, 4, 12, 0),
                   jst.localize(datetime.datetime(2020, 11, 4, 12, 0)),
                   datetime.datetime(2020, 11, 4, 12, 0),
                   jst.localize(datetime.datetime(2020, 11, 4, 12, 0)),
                   datetime.datetime(2020, 11, 4, 12, 30)
                  ]
        
        for one_args, one_result in zip(args, results):
            with self.subTest(one_args=one_args, one_result=one_result):
                self.assertEqual(get_floor_mod_datetime(*one_args), one_result)
        
        # get_ceil_mod_datetime
        select_datetime = datetime.datetime(2020, 11, 4, 12, 30, 0)
        freq_str = "H"
        ceil_datetime = get_ceil_mod_datetime(select_datetime, freq_str)
        self.assertEqual(ceil_datetime, datetime.datetime(2020, 11, 4, 13, 0))
        
        args = [(datetime.datetime(2020, 11, 4, 12, 30, 0), "H"),
                (jst.localize(datetime.datetime(2020, 11, 4, 13, 0, 0)), "H"),
                (datetime.datetime(2020, 11, 4, 12, 57, 0), "5T"),
                (jst.localize(datetime.datetime(2020, 11, 4, 12, 45, 0)), "30T"),
                (datetime.datetime(2020, 11, 4, 13, 10, 0), "30T")
               ]
        
        results = [datetime.datetime(2020, 11, 4, 13, 0),
                   jst.localize(datetime.datetime(2020, 11, 4, 13, 0)),
                   datetime.datetime(2020, 11, 4, 13, 0),
                   jst.localize(datetime.datetime(2020, 11, 4, 13, 0)),
                   datetime.datetime(2020, 11, 4, 13, 30)
                  ] 
        
        for one_args, one_result in zip(args, results):
            with self.subTest(one_args=one_args, one_result=one_result):
                self.assertEqual(get_ceil_mod_datetime(*one_args), one_result)
                
    def test_next_previous(self):
        def get_next_datetime_check(freq_str):
            start_datetime = datetime.datetime(2021,1,1,0,0,0)
            end_datetime = datetime.datetime(2021,2,1,0,0,0)
            true_datetime_array = pd.date_range(start_datetime, end_datetime, freq=freq_str).to_pydatetime()

            datetime_list = []
            iter_datetime = datetime.datetime(2021,1,1,0,0,0)
            datetime_list.append(iter_datetime)
            for _ in range(len(true_datetime_array)-1):
                iter_datetime = get_next_datetime(iter_datetime, freq_str)
                datetime_list.append(iter_datetime)

            result_datetime_array = np.array(datetime_list) 
            return np.array_equal(result_datetime_array, true_datetime_array)
        
        freq_strs = ["D", "12H", "4H", "H", "30T", "10T", "5T", "T"]
        for freq_str in freq_strs:
            with self.subTest(freq_str=freq_str):
                self.assertTrue(get_next_datetime_check(freq_str))
                
                
        def get_previous_datetime_check(freq_str):
            start_datetime = datetime.datetime(2021,1,1,0,0,0)
            end_datetime = datetime.datetime(2021,2,1,0,0,0)
            true_datetime_array = pd.date_range(start_datetime, end_datetime, freq=freq_str).to_pydatetime()

            datetime_list = []
            iter_datetime = datetime.datetime(2021,2,1,0,0,0)
            datetime_list.append(iter_datetime)
            for _ in range(len(true_datetime_array)-1):
                iter_datetime = get_previous_datetime(iter_datetime, freq_str)
                datetime_list.append(iter_datetime)

            result_datetime_array = np.array(sorted(datetime_list)) 
            return np.array_equal(result_datetime_array, true_datetime_array)
        
        freq_strs = ["D", "12H", "4H", "H", "30T", "10T", "5T", "T"]
        for freq_str in freq_strs:
            with self.subTest(freq_str=freq_str):
                self.assertTrue(get_previous_datetime_check(freq_str))
                
    def test_next_previous_multi(self):
        def add_datetime_check(freq_str):
            start_datetime = datetime.datetime(2021,1,1,0,0,0)
            end_datetime = datetime.datetime(2021,2,1,0,0,0)
            true_datetime_array = pd.date_range(start_datetime, end_datetime, freq=freq_str).to_pydatetime()

            datetime_list = [get_next_datetime(datetime.datetime(2021,1,1,0,0,0),freq_str, number=i) for i  in range(len(true_datetime_array))]
            result_datetime_array = np.array(datetime_list)
            return np.array_equal(result_datetime_array, true_datetime_array)
        
        freq_strs = ["D", "12H", "4H", "H", "30T", "10T", "5T", "T"]
        for freq_str in freq_strs:
            with self.subTest(freq_str=freq_str):
                self.assertTrue(add_datetime_check(freq_str))
                
                
        def sub_datetime_check(freq_str):
            start_datetime = datetime.datetime(2021,1,1,0,0,0)
            end_datetime = datetime.datetime(2021,2,1,0,0,0)
            true_datetime_array = pd.date_range(start_datetime, end_datetime, freq=freq_str).to_pydatetime()

            datetime_list = [get_previous_datetime(datetime.datetime(2021,2,1,0,0,0),freq_str, number=i) for i  in range(len(true_datetime_array))]
            result_datetime_array = np.array(sorted(datetime_list))
            return np.array_equal(result_datetime_array, true_datetime_array)
        
        freq_strs = ["D", "12H", "4H", "H", "30T", "10T", "5T", "T"]
        for freq_str in freq_strs:
            with self.subTest(freq_str=freq_str):
                self.assertTrue(sub_datetime_check(freq_str))

if __name__ == "__main__":
  unittest.main()
