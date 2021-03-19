import pandas as pd

class SimpleMovingAverage():
    def __init__(self, window_size, use_nan=True):
        self.window_size = window_size
        if self.window_size < 1 or not isinstance(self.window_size, int):
            raise ValueError("window size must be over 1")
        
        self.use_nan = use_nan
        
        self.column_suffix = "_ma{}".format(str(self.window_size))
        
    def __call__(self, df):
        new_df = df.rolling(self.window_size).mean()
        new_df = new_df.add_suffix(self.column_suffix)
        if not self.use_nan:
            new_df = new_df.loc[new_df.index[self.window_size-1:],:].copy()
        
        return new_df


def movingaverage(df, window_size, use_nan=True):
    return SimpleMovingAverage(window_size,
                               use_nan
                               )(df)