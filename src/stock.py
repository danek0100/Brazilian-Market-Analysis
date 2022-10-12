from hashlib import md5


# Класс для удобной работы с акциями.
class Stock:
    def __init__(self, id_, key_, close_price_, volume_):
        try:
            self.id = int(id_)
            self.key = key_
            self.name = ''
            self.close_price = close_price_
            self.volume = volume_
            self.profitability = []
            self.profitability_sorted = []
            self.risk = None
            self.E = None
            self.VaR = {}

        except TypeError:
            raise TypeError

    def __eq__(self, other):
        if self.id == other.id \
                and self.key == other.key:
            return True
        else:
            return False

    def __hash__(self):
        enc_str = self.key + str(self.id)
        return int(md5(enc_str.encode()).hexdigest(), 16)