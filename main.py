import timeseries as ts
import FTSImpl as FTS
import numpy

sarima1000007 = ts.TimeSeries((1, 0, 0), (0, 0, 0, 7))
'''sarima0100007 = ts.TimeSeries((0, 1, 0), (0, 0, 0, 7))
sarima0010007 = ts.TimeSeries((0, 0, 1), (0, 0, 0, 7))
sarima0001007 = ts.TimeSeries((0, 0, 0), (1, 0, 0, 7))
sarima0000107 = ts.TimeSeries((0, 0, 0), (0, 1, 0, 7))
sarima0000017 = ts.TimeSeries((0, 0, 0), (0, 0, 1, 7))
test_data = sarima1000007.test_data


print("(1, 0, 0), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima1000007.RMSE, mape=sarima1000007.MAPE))
print("(0, 1, 0), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0100007.RMSE, mape=sarima0100007.MAPE))
print("(0, 0, 1), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0010007.RMSE, mape=sarima0010007.MAPE))
print("(0, 0, 0), (1, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0001007.RMSE, mape=sarima0001007.MAPE))
print("(0, 0, 0), (0, 1, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0000107.RMSE, mape=sarima0000107.MAPE))
print("(0, 0, 0), (0, 0, 1, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0000017.RMSE, mape=sarima0000017.MAPE))'''


FTS.FuzzyTS(sarima1000007.train_data, 10, 'maximum', 'sets')