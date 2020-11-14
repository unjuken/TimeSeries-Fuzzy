import warnings
import matplotlib.pylab as plt
from pyFTS.partitioners import Grid
from pyFTS.common import FLR
from pyFTS.models import chen

class FuzzyTS:
    def __init__(self, trainData, parts, fuzzyMethod, fuzzyMode):

        self.data = trainData
        self.fs = Grid.GridPartitioner(data=data,npart=parts)
        self.fuzzyfied = fs.fuzzyfy(data, method=fuzzyMethod, mode=fuzzyMode)
        self.patterns = FLR.generate_non_recurrent_flrs(fuzzyfied)
        self.model = chen.ConventionalFTS(partitioner=fs)
        self.model.fit(data)