from utils import ConvertFreqOHLCV

def convert_freq_ohlcv(df, freq_str):
    converter = ConvertFreqOHLCV(freq_str)
    converted_df = converter(df.copy())  # 一応コピーしておく
    return converted_df