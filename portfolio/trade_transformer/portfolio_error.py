from utils import TradeSystemBaseError

class PortfolioVectorInvalidError(TradeSystemBaseError):
    """
    portfolio vector が有効かどうかチェックする
    """
    def __init__(self, err_str=None):
        """
        err_str:
            エラーメッセージ
        """
        self.err_str = err_str
    def __str__(self):
        if self.err_str is None:
            return "This portfolio is invalid"
        else:
            return self.err_str

if __name__ == "__main__":
    pass