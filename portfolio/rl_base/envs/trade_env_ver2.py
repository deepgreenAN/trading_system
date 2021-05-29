import numpy as np

class TradeEnv:
    """
    PortfolioStateを利用して基本的な売買を行う強化学習用の環境
    """
    def __init__(self, 
                 portfolio_transformer,
                 sampler_manager,
                 window=[0],
                 fee_const=0.0025,
                ):
        """
        portfolio_transformer: PortfolioTransformer
             ポートフォリオを遷移させるTransformer
        sampler_manager: SamplerManager
            各種サンプリングを行うためのマネージャー
        window: list
            PortfolioStateのウィンドウサイズ
        fee_const: float
            単位報酬・取引量当たりの手数料
        """
        self.portfolio_transformer = portfolio_transformer
        self.sampler_manager = sampler_manager
        self.window = window
        self.fee_const = fee_const
        
    def reset(self, seed=None):
        """
        Parameters
        ----------
        seed: int
            乱数シード
            
        Returns
        -------
        PortfolioState
            遷移したPortfolioStateのインスタンス
        float
            報酬．resetなので0
        bool
            終了を示すフラッグ
        dict
            その他の情報
        """
        random_ticker_names = self.sampler_manager.ticker_names_sampler.sample(seed=seed)  # 銘柄名のサンプリング
        random_datetime = self.sampler_manager.datetime_sampler.sample(seed=seed, window=self.window)  # 開始日時のサンプリング
        random_portfolio_vector = self.sampler_manager.portfolio_vector_sampler.sample(seed=seed)  # ポートフォリオベクトルのサンプリング
        random_mean_cost_price_array = self.sampler_manager.mean_cost_price_array_sampler.sample(seed=seed)  # 平均取得価格のサンプリング
        
        self.portfolio_transformer.price_supplier.ticker_names = list(random_ticker_names)  # 銘柄名の変更
        self.portfolio_transformer.initial_portfolio_vector = random_portfolio_vector  # 初期ポートフォリオを変更(Noneの場合デフォルト)
        self.portfolio_transformer.initial_mean_cost_price_array = random_mean_cost_price_array  #初期平均取得価格を変更(Noneの場合デフォルト) 
        
        portfolio_state, done = self.portfolio_transformer.reset(random_datetime, window=self.window)
        
        self.portfolio_state = portfolio_state
        
        return self.portfolio_state.copy(), 0, done, None
    
    def step(self, portfolio_vector):
        """
        Parameters
        ----------
        portfolio_vector: ndarray
            actionを意味するポートフォリオベクトル
            
        Returns
        -------
        PortfolioState
            遷移したPortfolioStateのインスタンス
        float
            報酬．
        bool
            終了を示すフラッグ
        dict
            その他の情報
        """        

        previous_portfolio_state = self.portfolio_state
        
        #状態遷移
        portfolio_state, done = self.portfolio_transformer.step(portfolio_vector)
        
        #報酬の計算
        portfolio_vector = portfolio_state.portfolio_vector
        
        price_change_ratio = portfolio_state.now_price_array / previous_portfolio_state.now_price_array  # y
        raw_reward_ratio = np.dot(portfolio_vector, price_change_ratio)  # r
        
        portfolio_change_vector = portfolio_vector - previous_portfolio_state.portfolio_vector #W_{t}-w_{t-1}
        reward = np.log(raw_reward_ratio*(1-self.fee_const*np.dot(portfolio_change_vector, portfolio_change_vector)))
        
        return portfolio_state.copy(), reward, done, None


if __name__ == "__main__":
    pass