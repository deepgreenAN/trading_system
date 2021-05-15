from pytz import timezone
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from get_stock_price import StockDatabase
from portfolio.trade_transformer import PortfolioTransformer, PortfolioRestrictorIdentity, FeeCalculatorFree
from portfolio.price_supply import StockDBPriceSupplier

from portfolio.rl_base.envs import TradeEnv, TickerSampler, DatetimeSampler, SamplerManager, PortfolioVectorSampler, ConstSamper
from portfolio.rl_base.basis_func import ComposeFunction, PriceNormalizeConst, MeanCostPriceNormalizeConst, State2Feature

jst = timezone("Asia/Tokyo")
start_datetime = jst.localize(datetime.datetime(2020,11,10,0,0,0))
end_datetime = jst.localize(datetime.datetime(2020,11,20,0,0,0))
ticker_number = 19
window = np.arange(0,50)
episode_length = 300
freq_str = "5T"

db_path = Path("db/sub_stock_db/nikkei_255_stock_v2.db")

ticker_codes_df = pd.read_csv(Path("portfolio/rl_base/nikkei225_modified.csv"), header=0)  # 自分で作成
ticker_codes = ticker_codes_df["code"].values.astype(str).tolist()

def make_env():
    # stock_db
    stock_db = StockDatabase(db_path)

    # sampler
    #ticker_names_sampler = TickerSampler(all_ticker_names=ticker_codes,
    #                                     sampling_ticker_number=ticker_number)


    ticker_names_sampler = ConstSamper(TickerSampler(ticker_codes, ticker_number).sample())  # 固定する



    start_datetime_sampler = DatetimeSampler(start_datetime=start_datetime,
                                            end_datetime=end_datetime,
                                            episode_length=episode_length,
                                            freq_str=freq_str
                                            )

    portfolio_sampler = PortfolioVectorSampler(ticker_number+1)


    sampler_manager = SamplerManager(ticker_names_sampler=ticker_names_sampler,
                                    datetime_sampler=start_datetime_sampler,
                                    portfolio_vector_sampler=portfolio_sampler,
                                    )


    # PriceSupplierの設定
    price_supplier = StockDBPriceSupplier(stock_db,
                                        [],  # 最初は何の銘柄コードも指定しない
                                        episode_length,
                                        freq_str,
                                        interpolate=True
                                        )

    # PortfolioTransformerの設定
    portfolio_transformer = PortfolioTransformer(price_supplier,
                                                portfolio_restrictor=PortfolioRestrictorIdentity(),
                                                use_ohlc="Close",
                                                initial_all_assets=1e6,  # 学習には関係ない
                                                fee_calculator=FeeCalculatorFree()
                                                )

    # TradeEnvの設定
    trade_env = TradeEnv(portfolio_transformer,
                        sampler_manager,
                        window=window,
                        fee_const=0.0025
                        )

    return trade_env
