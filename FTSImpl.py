import warnings
import matplotlib.pylab as plt
from pyFTS.partitioners import Grid
from pyFTS.common import FLR
from pyFTS.models import chen


class FuzzyTS:
    def __init__(self, trainData, parts, fuzzyMethod, fuzzyMode):

        self.trainData = trainData
        self.fs = Grid.GridPartitioner(data=self.trainData,npart=parts)
        self.fuzzyfied = self.fs.fuzzyfy(self.trainData, method=fuzzyMethod, mode=fuzzyMode)
        self.patterns = FLR.generate_non_recurrent_flrs(self.fuzzyfied)
        self.model = chen.ConventionalFTS(partitioner=self.fs)
        self.model.fit(self.trainData)
    
    def predict(self, data, steps_ahead = 1):
        self.forecasts = self.model.predict(data, steps_ahead)
        return self.forecasts