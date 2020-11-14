from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time


class TimeSeries:
    def parser(self, s):
        return datetime.strptime(s, '%d/%m/%Y')

    def __init__(self, my_order, my_seasonal_order):
        df = pd.read_csv("./archive/datos.csv",delimiter=";", parse_dates=[0], index_col=0, squeeze=True, date_parser=self.parser)
        print(df.head())
        #df.info()
        df = df.asfreq(pd.infer_freq(df.index))

        start_date = datetime(2017,1,1)
        end_date = datetime(2020,1,1)

        #remove 2020
        lim_catfish_sales = df[start_date:end_date]

        #normalizing
        lim_catfish_sales = (lim_catfish_sales - lim_catfish_sales.min()) / (lim_catfish_sales.max() - lim_catfish_sales.min())

        train_end = datetime(2019,7,1)
        test_end = datetime(2020,1,1)

        #train-test split
        self.train_data = lim_catfish_sales[:train_end]
        self.test_data = lim_catfish_sales[train_end + timedelta(days=1):test_end]

        #my_order = (0,1,0)
        #my_seasonal_order = (1, 0, 1, 7)

        # define model
        model = SARIMAX(self.train_data, order=my_order, seasonal_order=my_seasonal_order)

        #fit the model
        start = time()
        model_fit = model.fit()
        end = time()
        print('Model Fitting Time:', end - start)

        ## summary of the model
        print(model_fit.summary())

        #get the predictions and residuals
        self.predictions = model_fit.forecast(len(self.test_data))
        self.predictions = pd.Series(self.predictions, index=self.test_data.index)
        residuals = self.test_data - self.predictions
        self.MAPE = round(np.mean(abs(residuals/self.test_data)),4)
        self.RMSE = np.sqrt(np.mean(residuals**2))

        #print('Mean Absolute Percent Error:', )
        #print('Root Mean Squared Error:', )
    #endInit
#endClass 





