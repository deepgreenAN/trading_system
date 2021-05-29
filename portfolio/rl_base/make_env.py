from pytz import timezone
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from get_stock_price import StockDatabase
from portfolio.trade_transformer import PortfolioTransformer, PortfolioRestrictorIdentity, FeeCalculatorFree
from portfolio.price_supply import StockDBPriceSupplier

from portfolio.rl_base.envs import TradeEnv, TickerSampler, DatetimeSampler, SamplerManager, PortfolioVectorSampler, ConstSamper, MeanCostPriceSamplerNormal
from portfolio.rl_base.basis_func import ComposeFunction, PriceNormalizeConst, MeanCostPriceNormalizeConst, State2Feature

jst = timezone("Asia/Tokyo")
freq_str = "5T"

def make_env(db_path,
             csv_path,
             is_ticker_sample=True,
             start_datetime=jst.localize(datetime.datetime(2020,11,10,0,0,0)),
             end_datetime=jst.localize(datetime.datetime(2020,11,20,0,0,0)),
             episode_length=300,
             window=np.arange(0,50),
             ticker_number=19,
             fee_const=0.0025,
             ):
    ticker_codes_df = pd.read_csv(csv_path, header=0)  # 自分で作成
    ticker_codes = ticker_codes_df["code"].values.astype(str).tolist()
    # stock_db
    stock_db = StockDatabase(db_path)

    # sampler
    if is_ticker_sample:
        ticker_names_sampler = TickerSampler(all_ticker_names=ticker_codes,
                                            sampling_ticker_number=ticker_number)
    else:
        ticker_names_sampler = ConstSamper(TickerSampler(ticker_codes, ticker_number).sample())  # 固定する



    start_datetime_sampler = DatetimeSampler(start_datetime=start_datetime,
                                            end_datetime=end_datetime,
                                            episode_length=episode_length,
                                            freq_str=freq_str
                                            )

    portfolio_sampler = PortfolioVectorSampler(ticker_number+1)

    mean_cost_price_sampler = MeanCostPriceSamplerNormal(mean=None, var=None)  # 意味なし


    sampler_manager = SamplerManager(ticker_names_sampler=ticker_names_sampler,
                                    datetime_sampler=start_datetime_sampler,
                                    portfolio_vector_sampler=portfolio_sampler,
                                    mean_cost_price_array_sampler=mean_cost_price_sampler
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
                        fee_const=fee_const
                        )

    return trade_env
