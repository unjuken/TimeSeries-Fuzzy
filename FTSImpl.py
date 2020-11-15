import warnings
import matplotlib.pylab as plt
from pyFTS.partitioners import Grid
from pyFTS.common import FLR
from pyFTS.models import chen
import pyFTS.models.hofts


class FuzzyTS:
    def __init__(self, trainData, parts, fuzzyMethod, fuzzyMode, order = 1):
        self.order = order
        self.trainData = trainData
        self.fs = Grid.GridPartitioner(data=self.trainData,npart=parts)
        self.fuzzyfied = self.fs.fuzzyfy(self.trainData, method=fuzzyMethod, mode=fuzzyMode)
        self.patterns = FLR.generate_non_recurrent_flrs(self.fuzzyfied)
        if self.order > 1:
            self.modelHO = pyFTS.models.hofts.HighOrderFTS(order = self.order, partitioner=self.fs)
            self.modelHO.fit(self.trainData)
        else:
            self.model = chen.ConventionalFTS(partitioner=self.fs)
            self.model.fit(self.trainData)
    
    def predict(self, data, steps_ahead = 1):
        if self.order > 1 :
            self.forecasts = self.modelHO.predict(data, steps_ahead) 
        else : 
            self.forecasts = self.model.predict(data, steps_ahead) 
        return self.forecasts
    
    def getModel(self):
        if self.order > 1 :
            return self.modelHO
        else :
            return self.model
