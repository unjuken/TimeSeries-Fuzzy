import timeseries as ts
import FTSImpl as FTS
import numpy
import numpy as np
import matplotlib.pylab as plt

'''sarima1000007 = ts.TimeSeries((1, 0, 0), (0, 0, 0, 7))
sarima0100007 = ts.TimeSeries((0, 1, 0), (0, 0, 0, 7))
sarima0010007 = ts.TimeSeries((0, 0, 1), (0, 0, 0, 7))
sarima0001007 = ts.TimeSeries((0, 0, 0), (1, 0, 0, 7))'''
sarima0000107 = ts.TimeSeries((0, 0, 0), (0, 1, 0, 7))
'''sarima0000017 = ts.TimeSeries((0, 0, 0), (0, 0, 1, 7))
test_data = sarima1000007.test_data


print("(1, 0, 0), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima1000007.RMSE, mape=sarima1000007.MAPE))
print("(0, 1, 0), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0100007.RMSE, mape=sarima0100007.MAPE))
print("(0, 0, 1), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0010007.RMSE, mape=sarima0010007.MAPE))
print("(0, 0, 0), (1, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0001007.RMSE, mape=sarima0001007.MAPE))
print("(0, 0, 0), (0, 1, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0000107.RMSE, mape=sarima0000107.MAPE))
print("(0, 0, 0), (0, 0, 1, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0000017.RMSE, mape=sarima0000017.MAPE))'''


trainMtx = numpy.matrix(sarima0000107.train_data).transpose()
testMtx = numpy.matrix(sarima0000107.test_data).transpose()
train = numpy.array(trainMtx.flatten())[0]
test = numpy.array(testMtx.flatten())[0]

fuzzyTS  = FTS.FuzzyTS(train, 10, 'maximum', 'sets')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])
fuzzyTS.fs.plot(ax)
fuzzyTS.fs