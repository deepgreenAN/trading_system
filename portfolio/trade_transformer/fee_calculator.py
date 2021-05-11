import numpy as np
from abc import ABCMeta, abstractmethod

class FeeCalculator(metaclass=ABCMeta):
    """
    手数料を計算する抽象基底クラス
    """
    @abstractmethod
    def calculate(self, pre_portfolio_state, new_portfolio_state):
        pass


class FeeCalculatorPerNumber(FeeCalculator):
    """
    取引個数に応じて手数料を計算するFeeCalculator
    """
    def __init__(self, fee_per_number):
        self.fee_per_number = fee_per_number
    def calculate(self, pre_portfolio_state, new_portfolio_state):
        not_key_currency_indices_list = list(range(len(pre_portfolio_state.names)))
        not_key_currency_indices_list.remove(pre_portfolio_state.key_currency_index)
        
        not_key_currency_indices = np.array(not_key_currency_indices_list) 
        commition_fee = self.fee_per_number*np.abs((new_portfolio_state.numbers[not_key_currency_indices] - pre_portfolio_state.numbers[not_key_currency_indices])).sum()
        return commition_fee


if __name__ == "__main__":
    pass