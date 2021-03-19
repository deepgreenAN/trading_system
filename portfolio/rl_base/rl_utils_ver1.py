import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RansacGradLearningDecider():
    def __init__(self, line_number=10, point_number=2, grad_space_distance_lim=50, decision_rate=0.7):
        self.line_number = line_number
        self.point_number = point_number
        self.grad_space_distance_lim = grad_space_distance_lim
        self.decision_rate = decision_rate
        
    def decide(self, state_list, info_list, env):
        """
        うまく学習できているかどうか判定
        """
        # stock_number の arrayを取得
        stock_number_tensor = torch.from_numpy(np.array(list(map(lambda state: state.unit_number*env.one_unit_stocks, state_list)))).float()
        x_tensor = torch.from_numpy(np.arange(0, len(stock_number_tensor))).float()

        # line_number の数だけサンプリング(重複なし)
        random_index_all = np.random.permutation(len(stock_number_tensor))
        random_index = random_index_all[:int(self.line_number*self.point_number)]  # 重複なく(.line_number, point_number)のインデックスを取得

        sampled_stock_number_tensor = stock_number_tensor[random_index]
        sampled_stock_number_tensor_reshaped = torch.reshape(sampled_stock_number_tensor, (self.line_number, self.point_number))

        sampled_x_tensor = x_tensor[random_index]
        sampled_x_tensor_reshaped = torch.reshape(sampled_x_tensor, (self.line_number, self.point_number))

        # 全データの最小二乗解
        X = np.stack([sampled_x_tensor.numpy(), np.ones_like(sampled_x_tensor.numpy())], axis=1)
        inv_XtX = np.linalg.inv(np.dot(X.T,X))
        Xty = np.dot(X.T, sampled_stock_number_tensor.numpy())
        beta_hat = np.dot(inv_XtX, Xty)

        # バッチの最小二乗解
        b_X = torch.stack([sampled_x_tensor_reshaped, torch.ones_like(sampled_x_tensor_reshaped).float()],dim=2)
        b_XtX = torch.bmm(b_X.transpose(1,2),b_X)
        b_inv_XtX = torch.inverse(b_XtX)
        b_Xty = torch.bmm(b_X.transpose(1,2), sampled_stock_number_tensor_reshaped[:,:,None])

        b_beta_hat = torch.bmm(b_inv_XtX, b_Xty)  # (batch,2,1) であることに注意
        b_beta_hat_squeeze = b_beta_hat.squeeze().numpy()  # (batch,2)
        
        # 全データの最小二乗解に近いバッチの最小二乗解の個数の割合によって判定
        near_beta_hat_bool = ((beta_hat[None,:] - b_beta_hat_squeeze)**2).sum(axis=1) < self.grad_space_distance_lim
        if near_beta_hat_bool.sum() / self.line_number < self.decision_rate:  # 近い割合がdecision_rateより低い場合
            return_bool = True
        else:
            return_bool = False
        
        return return_bool


class QFunction(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.fc2 = nn.Linear(32, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, n_actions)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        
        x = self.fc4(x)
        out = pfrl.action_value.DiscreteActionValue(x)
        return out


def episode(env, agent, state_transform=None, reward_transform=None, print_span=None, is_observe=True):
    state_list = []
    info_list = []
    action_list = []
    
    obs,_,_,info = env.reset()

    state_list.append(obs)
    info_list.append(info)
    R = 0
    t = 1
    if print_span is not None:
        print("\tt:{},all_property:{}, unit_number:{}, price:{}, penalty:{}, cash:{}".format(t,
                                                                                             info["all_property"],
                                                                                             obs.unit_number,
                                                                                             obs.now_price,
                                                                                             info["penalty"],
                                                                                             obs.cash
                                                                                            ))
    
    if state_transform is not None:
        normalized_obs = state_transform(obs)
    else:
        normalized_obs = obs

    while True:
        action = agent.act(normalized_obs)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        R += reward
        t += 1
        reset = False


        # state, rewardの前処理
        if state_transform is not None:
            normalized_obs = state_transform(obs)
        else:
            normalized_obs = obs
        if reward_transform is not None:
            normalized_reward = reward_transform(reward)
        else:
            normalized_reward = reward

        if is_observe:  # 観測(学習)する場合
            agent.observe(normalized_obs, normalized_reward, done, reset)

        state_list.append(obs)
        info_list.append(info)

        if done or reset:
            break
        if print_span is not None:
            if t%print_span==0:
                print("\tt:{},all_property:{}, unit_number:{}, price:{}, penalty:{}, cash:{}".format(t,
                                                                                                     info["all_property"],
                                                                                                     obs.unit_number,
                                                                                                     obs.now_price,
                                                                                                     info["penalty"],
                                                                                                     obs.cash
                                                                                                    ))
                print("\taction_counter:",collections.Counter(action_list))
    
    if print_span is not None:
        print("\tt:{},all_property:{}, unit_number:{}, price:{}, penalty:{}, cash:{}".format(t,
                                                                                             info["all_property"],
                                                                                             obs.unit_number,
                                                                                             obs.now_price,
                                                                                             info["penalty"],
                                                                                             obs.cash
                                                                                            ))
        print("\taction_counter:",collections.Counter(action_list))
        print("finished. episode length: {}".format(t))
    return state_list, info_list, action_list


if __name__ == "__main__":
    import sys
    sys.path.append(r"E:\システムトレード入門\tutorials\rl\pfrl")
    sys.path.append(r"E:\システムトレード入門\trade_system_git_workspace")

    import datetime
    from pytz import timezone
    from pathlib import Path

    import pfrl
    from get_stock_price import StockDatabase
    from envs_ver2 import OneStockEnv, NormalizeState, NormalizeReward

    db_path = Path("E:/システムトレード入門/trade_system_git_workspace/db/stock_db") / Path("stock.db")
    stock_db = StockDatabase(db_path)

    jst_timezone = timezone("Asia/Tokyo")
    start_datetime = jst_timezone.localize(datetime.datetime(2020,11,1,0,0,0))
    end_datetime = jst_timezone.localize(datetime.datetime(2020,12,1,0,0,0))
    #end_datetime = get_next_workday_jp(start_datetime, days=11)  # 営業日で一週間(5日間)

    #stock_names = "4755"
    #stock_names = "9984"
    stock_names = "6502"
    #stock_names = ["6502","4755"]
    #stock_list = ["4755","9984","6701","7203","7267"]

    stock_df = stock_db.search_span(stock_names=stock_names, 
                                    start_datetime=start_datetime,
                                    end_datetime=end_datetime,
                                    freq_str="T",
                                    to_tokyo=True
                                )

    use_ohlc="Close"

    initial_cash = 1.e6
    initial_unit = 50

    freq_str = "5T"
    episode_length = 12*5*7  # 1週間

    #state_time_list = [0,1,12,12*3,12*5,12*5*3],  # [現在，次時刻，一時間後，3時間後，5時間後(1日後), 15時間後(3日後)]
    state_time_list = [0,
                    1,
                    2,
                    6,
                    12,
                    12*2,
                    12*3,
                    12*4,
                    12*5*1,
                    12*5*2,
                    12*5*3,
                    12*5*4,
                    12*5*5,
                    ]  # 現在，5分後, 10分後, 30分後, 1時間後, 2時間後, 3時間後, 4時間後, 1日後, 2日後, 3日後, 4日後, 5日後, 6日後, 7日後

    one_unit_stocks = 20
    max_units_number = 5
    stay_penalty_unit_bound=30


    env = OneStockEnv(stock_db,
                    stock_names=stock_names,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    freq_str="5T",
                    episode_length=episode_length,  # 一週間
                    state_time_list=state_time_list,
                    use_ohlc=use_ohlc,  # 終値を使う
                    initial_cash=initial_cash,  # 種銭：100万円,
                    initial_unit=initial_unit,
                    use_view=False,
                    one_unit_stocks=one_unit_stocks,  # 独自単元数
                    max_units_number=max_units_number,  # 一度に売買できる独自単元数
                    low_limmit=1.e4,  # 全財産がこの値以下になれば終了
                    interpolate=True,
                    stay_penalty_unit_bound=stay_penalty_unit_bound  # このunit数以下の場合のstayはペナルティ
                    )

    state_transform = NormalizeState(cash_const=initial_cash,
                                    unit_const=100,
                                    price_const=1.e4,
                                    )

    reward_transform = NormalizeReward(reward_const=5.e3,
                                    )

    obs_size = env.observation_space.low.size
    n_actions = env.action_space.n
    q_func = QFunction(obs_size, n_actions)

    optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-4)

    gamma = 0.95

    init_episilon = 0.3
    init_explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=init_episilon,
                                                    random_action_func=env.action_space.sample
                                                )

    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10**6)


    def phi_func(observe):
        observe_array = observe.to_numpy()
        return observe_array.astype(np.float32, copy=False)


    phi = phi_func

    gpu = -1 # -1 is cpu

    good_agent = pfrl.agents.DoubleDQN(
        q_function=q_func,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=gamma,
        explorer=init_explorer,
        replay_start_size=500,
        update_interval=1,
        target_update_interval=100,
        phi=phi,
        gpu=gpu
    )

    bad_agent = pfrl.agents.DoubleDQN(
        q_function=q_func,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=gamma,
        explorer=init_explorer,
        replay_start_size=500,
        update_interval=1,
        target_update_interval=100,
        phi=phi,
        gpu=gpu
    )


    folder_name = "2020_12_27__03_31_48"
    load_path = Path("agents") / Path(folder_name)
    good_agent.load(load_path)


    folder_name = "2020_12_23__05_38_05"
    load_path = Path("agents") / Path(folder_name)
    bad_agent.load(load_path)

    with good_agent.eval_mode():
        #env.seed()
        env.seed(0,0)
        #env.seed(0)
        good_state_list, good_info_list, _ = episode(env, good_agent, state_transform=state_transform, reward_transform=None, print_span=None, is_observe=False)


    with bad_agent.eval_mode():
        #env.seed()
        env.seed(0,0)
        #env.seed(0)
        bad_state_list, bad_info_list, _ = episode(env, bad_agent, state_transform=state_transform, reward_transform=None, print_span=None, is_observe=False)


    lr_eval_decider = RansacGradLearningDecider(line_number=10, point_number=2, grad_space_distance_lim=50, decision_rate=0.7)
    print("good lr:",lr_eval_decider.decide(good_state_list, good_info_list, env))
    print("bad lr:",lr_eval_decider.decide(bad_state_list, bad_info_list, env))