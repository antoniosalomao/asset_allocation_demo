import pandas as pd
import numpy as np
import yfinance as yf
import math
from math import floor

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------#
# Functions #
#-----------#

def get_sector_equity_dict(df_holdings):
    '''
    Create Dictionary --> Key: Sector, Values: Equities
    '''
    sector_equity_dict = {k: [] for k in df_holdings['sector'].unique() if k not in [None, np.nan, 'Cash and/or Derivatives']}
    for i, row in df_holdings.iterrows():
        if row['sector'] in sector_equity_dict:
            sector_equity_dict[row['sector']].append(row['ticker'])

    return sector_equity_dict

def get_returns(yf_dict):
    '''
    Returns
    '''
    yf_dict['tickers'].append(yf_dict['benchmark'])
    yf_dict['tickers'] = sorted([k + '.SA' for k in yf_dict['tickers']])
    df_main = yf.download(**yf_dict, )
    df_main.index = pd.to_datetime(df_main.index)

    # Slicing and cleaning DataFrame --> Price Series
    p_options = ['Adj Close', 'Close']
    df_ps = df_main.loc[:, [p_options[1]]].ffill(axis=0)
    df_ps.columns = df_ps.columns.droplevel()
    df_ps.columns = [col.replace('.SA', '') for col in list(df_ps.columns)]

    # Returns
    df_ret = np.log(df_ps).diff(1).fillna(method='ffill').dropna(axis=1, thresh=floor((0.75)*len(df_ps)))

    return df_ret

def get_excess_returns(yf_dict, df_ret):
    '''
    Excess Returns (over benchmark)
    '''
    df_excess_ret = df_ret.sub(df_ret[yf_dict['benchmark']], axis=0)
    df_excess_ret.drop(yf_dict['benchmark'], axis=1, inplace=True)

    return df_excess_ret

def get_expected_returns(yf_dict, df_ret):
    '''
    Expected Returns (Historical Mean)
    '''
    df_exp_ret = pd.DataFrame(data=df_ret.mean(), columns = ['Expected Returns'])
    ann_dict = {'1d': 252}
    if (yf_dict['annualize'] == True):
        df_exp_ret = df_exp_ret*ann_dict[yf_dict['interval']]

    return df_exp_ret

def get_covar(yf_dict, df_ret):
    '''
    Sample Variance-Covariance Matrix of Asset Returns
    '''
    df_covar = df_ret.cov()
    ann_dict = {'1d': 252}
    if (yf_dict['annualize'] == True):
        df_covar = df_covar*ann_dict[yf_dict['interval']]

    return df_covar

def get_df_ir(yf_dict, df_ret):
    '''
    Information Ratio DataFrame
    '''
    df_exp_ret = get_expected_returns(yf_dict=yf_dict, df_ret=df_ret)
    df_covar = get_covar(yf_dict=yf_dict, df_ret=df_ret)
    exp_ret_arr, var_arr, stdev_arr = np.array(df_exp_ret['Expected Returns']), np.diag(df_covar), np.power(np.diag(df_covar), 0.5)
    ir_arr = (exp_ret_arr/stdev_arr)
    ir_df = pd.DataFrame([exp_ret_arr, stdev_arr, ir_arr], columns=df_exp_ret.index, index=['Expected Excess Returns', 'Standard Deviation', 'Information Ratio']).T

    return ir_df

def get_max_ir_arr(yf_dict, df_holdings):
    '''
    Array with max_ir of a given sector
    '''
    sector_equity_dict = get_sector_equity_dict(df_holdings=df_holdings)
    max_ir_arr = []
    for k, v in sector_equity_dict.items():
        yf_dict['tickers'] = v
        df_ret = get_returns(yf_dict=yf_dict)
        df_excess_ret = get_excess_returns(yf_dict=yf_dict, df_ret=df_ret)
        df_ir = get_df_ir(yf_dict=yf_dict, df_ret=df_excess_ret)
        max_ir = df_ir['Information Ratio'].idxmax()
        max_ir_arr.append(max_ir)
        print(df_ir)

    return max_ir_arr

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
#------#
# Main #
#------#

read_excel_dict = {'io': "/Users/antonioelias/Desktop/risk parity bolsa balcao/small_11_holdings.xlsx",
           'sheet_name': 'SMAL11_holdings',
              'usecols': 'A:H',
                'names': ['ticker', 'name', 'weight', 'price', 'quotas', 'market value', 'face value', 'sector']}

df_smal11_holdings = pd.read_excel(**read_excel_dict)

yf_dict = {'benchmark': 'SMAL11',
             'tickers': None,
               'start': '2019-11-01',
                 'end': '2019-11-30',
            'interval': '1d',
           'annualize': True}

max_ir = get_max_ir_arr(yf_dict=yf_dict, df_holdings=df_smal11_holdings)


print(max_ir)



