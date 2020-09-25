
class PolynomialDecay:
    def __init__(self, maxEpochs=100, initAlpha=2.5e-3, power=0.9):
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay

        return float(alpha)
