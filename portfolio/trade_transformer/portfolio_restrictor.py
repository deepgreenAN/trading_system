from abc import ABCMeta, abstractmethod

class PortfilioRestrictor(metaclass=ABCMeta):
    """
    ポートフォリオの制限を行う抽象基底クラス
    restrictメソッドをオーバーライドする必要がある
    """
    @abstractmethod
    def restrict(self, portfolio_state, supplied_data_unit, portfolio_vector):
        pass


class PortfolioRestrictorSingleKey(PortfilioRestrictor):
    """
    """
    def __init__(self, unit_number, key_name):
        pass
        
    def restrict(self, portfolio_state, supplied_data_unit, portfolio_vector):
        pass


class PortfolioRestrictorIdentity(PortfilioRestrictor):
    """
    portfolioの恒等写像を行うPortfolioRestrictor
    """
    def restrict(self, portfolio_state, supplied_data_unit, portfolio_vector):
        return portfolio_vector


if __name__ == "__main__":
    pass