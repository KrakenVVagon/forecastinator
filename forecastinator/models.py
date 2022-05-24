'''
05/18/2022

Andrew Younger

Model definitions for forecasting DAU and players
'''
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf

class TimeSeries(pd.DataFrame):
    '''
    TimeSeries inherits from pandas dataframes and simply adds a few functions to assist model creation
    '''
    @property
    def _constructor(self):
        return TimeSeries

    def check_stationarity(self,window,plot=False):
        r_mean = self.rolling(window=window).mean()
        r_std  = self.rolling(window=window).std()

        if plot:
            print('Plotting stationarity not currently supported')

        r = adfuller(self)
        adf_statistic = r[0]
        p_value = r[1]
        critical_values = r[4]

        return None

    def mean_shift(self,window):
        log_df = np.log(self)
        r_mean = log_df.rolling(window=window).mean()

        mean_df = log_df - r_mean
        mean_df.dropna(inplace=True)
        return TimeSeries(mean_df)

    def log_shift(self):
        log_df = np.log(self)

        shift_df = log_df - log_df.shift()
        shift_df.dropna(inplace=True)

        return TimeSeries(shift_df)

    def get_acf(self,plot=False,nlags=20):
        if plot:
            print("Plotting not currently supported")

        return acf(self,fft=False,alpha=0.05,nlags=nlags)

    def get_pacf(self,plot=False,nlags=20):
        if plot:
            print("Plotting not currently supported")

        return pacf(self,alpha=0.05,nlags=nlags)

    def get_p_value(self,n=3,nlags=20):
        '''
        Get the p-value for ARIMA models. Comes from the PACF
        '''
        cf, conf = TimeSeries.get_pacf(self,nlags=nlags)

        mconf = np.mean(conf,axis=0)
        lower_bound = mconf[0]
        upper_bound = mconf[1]

        cf = np.array((lower_bound < cf) & (cf < upper_bound))

        p = np.where( (cf[:-(n-1)] == cf[(n-2):-1]) & (cf[(n-2):-1] == cf[(n-1):]) )[0][0]

        return p

    def get_q_value(self,n=3,nlags=20):
        '''
        Get the q-value for ARIMA models. Comes from the ACF
        Same as p value but ACF instead of PACF
        '''
        cf, conf = TimeSeries.get_acf(self,nlags=nlags)

        mconf = np.mean(conf,axis=0)
        lower_bound = mconf[0]
        upper_bound = mconf[1]

        cf = np.array((lower_bound < cf) & (cf < upper_bound))

        q = np.where( (cf[:-(n-1)] == cf[(n-2):-1]) & (cf[(n-2):-1] == cf[(n-1):]) )[0][0]

        return q

    def create_arima_model(self,p=None,d=1,q=None,transform=None,orig=None,frequency='D',seasonal=False,s=None):
        p = p or TimeSeries.get_p_value(self)
        q = q or TimeSeries.get_q_value(self)

        if not transform:
            if orig is None:
                raise Exception('Using no transformation requires original data set')
            return ArimaModel(self,order=(p,d,q),orig=orig,frequency=frequency)
        elif transform == 'logshift':
            df = TimeSeries.log_shift(self)
        elif transform == 'meanshift':
            df = TimeSeries.mean_shift(self,window=7)
        elif transform is not None:
            df = TimeSeries(transform(self))

        orig = self

        if seasonal:
            D = max(2-d,0)
            s = s or 7
            P = p
            Q = q
            return ArimaModel(df,order=(p,d,q),orig=orig,frequency=frequency,seasonal_order=(P,D,Q,s))

        return ArimaModel(df,order=(p,d,q),orig=orig,frequency=frequency)

class ArimaModel(ARIMA):

    def __init__(self,df,order,orig,frequency,seasonal_order=None):
        if seasonal_order is not None:
            super().__init__(df,order=order,freq=frequency,seasonal_order=seasonal_order)
        else:
            super().__init__(df,order=order,freq=frequency)
        
        self.fdata = self.fit()
        self.df = df
        self.values = self.fdata.fittedvalues
        self.orig = orig
        self.order = order
        self.frequency = frequency
        self.seasonal_order = seasonal_order

        return None

    def _undo_transform(self,transform='meanshift',window=7,origlog=False,tdf=None):
        '''
        Reverses the transforms that may have been done to create the ARIMA model
        Accepted transforms are "meanshift" and "logshift"
        '''
        if tdf is None:
            tdf = TimeSeries(self.values)
            tdf = tdf.rename(columns={0:self.orig.columns[0]})

        if not origlog:
            orig = np.log(self.orig)
        else:
            orig = self.orig
        
        # if transform is not a string assume it is some special function
        if not isinstance(transform,str):
            return TimeSeries(transform(df))
        
        if transform == "meanshift":
            tdf = tdf + orig.rolling(window=window).mean()
            return TimeSeries(np.exp(tdf))
        elif transform == "logshift":
            return TimeSeries(tdf + orig.shift())
        else:
            raise Exception('{} not an accepted transform. Options are "meanshift" or "logshift"'.format(transform))

        return None

    def predict(self,start_date,end_date,do_transform=True,k=7,origlog=False,transform='meanshift',T=4):

        if not do_transform:
            return TimeSeries(self.fdata.predict(start=start_date,end=end_date))

        x = TimeSeries(self.fdata.predict(start=start_date,end=end_date))

        if not origlog:
            orig = np.log(self.orig)
        else:
            orig = self.orig

        if transform == "meanshift":
            frac = -k/(1-k)
            u = np.zeros(len(x))

            rolling_mean = orig.rolling(window=k).mean()

            for i in range(len(x)):
                j = min(np.count_nonzero(u),k-1)
                if j < k-1:
                    u[i] = ( x.values[i] + np.sum(rolling_mean[int((-1)*(k-1-i)):])/k + np.sum(u[:j])/k )*frac
                else:
                    u[i] = ( x.values[i] + np.sum(u[i-j:i])/k )*frac
            
            return TimeSeries(np.exp(u),columns=['dau'],index=x.index)

        elif transform == 'logshift':
            d1 = orig.iloc[orig.index.get_loc(start_date)-1].values[0]

            return TimeSeries(np.exp(x.cumsum()*-1 + d1))

        elif transform == 'lineartrend':
            trend_func = ArimaModel.get_trend(self,start_date,k=k,T=T)

            from datetime import datetime as dt
            start = dt.strptime(start_date,'%Y-%m-%d')
            end = dt.strptime(end_date,'%Y-%m-%d')

            days = (end-start).days

            line = trend_func(np.arange(k*T,k*T+days+1))

            result = [ float(x.values[i]+line[i]) for i in range(len(x)) ]

            return TimeSeries(np.exp(result),columns=['dau'],index=x.index)

        else:
            raise ValueError('{} not an accepted transformation. Options are "meanshift" or "logshift"'.format(transform))

    def get_trend(self,end_date,origlog=False,k=7,T=4):

        if not origlog:
            orig = np.log(self.orig)
        else:
            orig = self.orig

        d1 = orig.index.get_loc(end_date)

        locs = orig.iloc[d1-k*T:d1+1]
        lin_fit = np.polyfit(range(len(locs)),locs.values,1)
            
        m = lin_fit[0][0]
        b = lin_fit[1][0]

        def f(x):
            return m*x + b
        return f
