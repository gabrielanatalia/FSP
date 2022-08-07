from datetime import date
from datetime import datetime
from copy import copy
from math import sqrt
from decimal import Decimal
import pandas as pd
import numpy as np
from copy import copy
from missingno import matrix as msmx

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from  matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

import seaborn as sns

import pickle
import pyfolio as pf


ANN_FACTOR = 252
SEM = 126
WEEK = 5
MONTH = 21

G10_FX_M = ["AUDUSD", "USDCAD", "EURUSD", "GBPUSD", "USDJPY", "NZDUSD", "USDSEK", "USDNOK", "USDCHF"]
G10_FX = ["USDAUD", "USDCAD", "USDEUR", "USDGBP", "USDJPY", "USDNZD", "USDSEK", "USDNOK", "USDCHF"]
ASIA_FX = ['USDSGD', 'USDTHB', 'USDTWD', 'USDKRW', 'USDPHP', 'USDINR', 'USDIDR', 'USDMYR', 'USDCNH']
ALL_FX = G10_FX + ASIA_FX

plt.rcParams["figure.figsize"] = (16,9)

PALETTE = ['#0000FF', '#FF7D40', '#008000', '#CD2626', '#68228B',
           '#FF1493', '#98FB98', '#CDCD00', '#00FFFF', '#808080',
           '#93AA00', '#593315', '#232C16', '#4E8CA1', '#F9C1E7',
           '#595697', '#2AED9E', '#ADA662']

c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
v = [0,.15,.4,.5,0.6,.9,1.]
l = list(zip(v,c))
RdGn = LinearSegmentedColormap.from_list('rg',l, N=256)

def read_prices(path: str):
    return pd.read_csv(path, index_col = 0, header = [0, 1], parse_dates = True)

# def omit_leading_zeros(series):
#   return np.trim_zeros(series, "f")

def omit_leading_zeros(data):
  if type(data) == pd.Series:
    return np.trim_zeros(data, "f")
  elif type(data) == pd.DataFrame:
    return pd.concat([np.trim_zeros(data.loc[:,x], "f") for x in data.columns], axis = 1)

def omit_trailing_zeros(series):
  return np.trim_zeros(series, "b")

def omit_leading_trailing_zeros(series):
  return np.trim_zeros(series, "fb")

def omit_leading_na(data, how="all"):
  if type(data) == pd.Series:
    return data.loc[data.first_valid_index():]
  elif type(data) == pd.DataFrame:
    if how == "all":
      first_index = max([list(data.index).index(data.loc[:,col].first_valid_index()) for col in data])
    elif how == "any":
      first_index = min([list(data.index).index(data.loc[:,col].first_valid_index()) for col in data])
  return data.iloc[first_index:]

def omit_trailing_na(data, how = "all"):
  if type(data) == pd.Series:
    return data.loc[:data.last_valid_index()]
  elif type(data) == pd.DataFrame:
    if how == "all":
      last_index = min([list(data.index).index(data.loc[:,col].last_valid_index()) for col in data])
    elif how == "any":
      last_index = max([list(data.index).index(data.loc[:,col].last_valid_index()) for col in data])
  return data.iloc[:last_index+1]

##############################################################
###################### DATA PREPARATION ######################
##############################################################

DATE_RANGE = pd.date_range(date(2000,1,3), date(2022,1,1), freq='B')

def fill_date(data, fill_forward = False, limit = None):
  date_range_df = pd.DataFrame(DATE_RANGE).set_index(0)
  date_range_df.index.names = ["Dates"]
  data.index.names = ["Dates"]
  filled_data = pd.merge(date_range_df, data, how="outer", on="Dates")
  if fill_forward:
    filled_data = filled_data.ffill(limit=limit)
  filled_data = filled_data.loc[:data.index.max()]
  if type(data) == pd.Series:
    return filled_data[filled_data.index.isin(date_range_df.index)].sort_index().iloc[:,0]
  return filled_data[filled_data.index.isin(date_range_df.index)].sort_index()

def standardize_series(adf, typ):
  df = adf.copy()
  problems = [list(df).index(x) for x in list(df) if x[:3] != "USD"]
  tmp_names = list(map(lambda x: df.columns[x] if x not in problems else (df.columns[x][3:] + df.columns[x][:3]) , range(len(df.columns))))
  for i in problems:
    if typ == "prices":
      df.iloc[:,i] = 1/df.iloc[:,i]
    elif typ == "returns":
      df.iloc[:,i] = df.iloc[:,i] * -1
    else:
      raise Exception("Choose `prices` or `returns`")
  df.columns = tmp_names
  return df

# Load SPOT Prices
raw_data = pd.read_csv("../../Dymon/Code Data/NUS_Data.csv", index_col=0, header=1, dtype=str)
new_raw_data = raw_data.iloc[2:].astype(float)
new_raw_data.index = pd.to_datetime(new_raw_data.index)

fx = raw_data[[x for x in list(raw_data) if ("USD" in x or "+1" in x)]]
spot_ndf = fx[[x for x in fx.columns if fx.loc["field", x] == "PX_LAST"]]
spot_ndf = spot_ndf.iloc[2:,:]
spot_ndf.columns = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDSEK", "USDNOK",
                    "USDCNY", "USDCNH", "USDSGD", "USDCHF", "USDTHB", "USDIDR", "USDINR", "USDMYR",
                    "USDTWD", "USDKRW", "USDPHP"]
spot_ndf.index.name = "date"
spot_ndf.index = pd.to_datetime(spot_ndf.index)
spot_ndf = spot_ndf.astype(float)
spot = spot_ndf.copy()

# Load NDF
ndf = raw_data[[x for x in list(raw_data) if x in ["IHN1M CMPN Curncy", "IRN1M CMPN Curncy", "MRN1M CMPN Curncy", "NTN1M CMPN Curncy", "KWN1M CMPN Curncy", "PPN1M CMPN Curncy"]]].iloc[2:]
ndf.columns = ["USDIDR", "USDINR", "USDMYR", "USDTWD", "USDKRW", "USDPHP"]
ndf = ndf.astype(float)
ndf.index = pd.to_datetime(ndf.index)

# Forward Scale
fwd_scale = {"EURUSD": 10000,
             "GBPUSD" : 10000,
             "AUDUSD" : 10000,
             "NZDUSD" : 10000,
             "USDJPY" : 100,
             "USDCAD" : 10000,
             "USDSEK" : 10000,
             "USDNOK" : 10000,
             "USDCNY" : 10000,
             "USDCNH" : 10000,
             "USDSGD" : 10000,
             "USDCHF" : 10000,
             "USDTHB" : 100,
             "USDIDR" : 1,
             "USDINR" : 100,
             "USDMYR" : 10000,
             "USDTWD" : 1,
             "USDKRW" : 1,
             "USDPHP" : 1}

# Convert to SPOT
spot["USDIDR"] = spot_ndf["USDIDR"] - ndf["USDIDR"]/fwd_scale["USDIDR"]
spot["USDINR"] = spot_ndf["USDINR"] - ndf["USDINR"]/fwd_scale["USDINR"]
spot["USDMYR"] = spot_ndf["USDMYR"] - ndf["USDMYR"]/fwd_scale["USDMYR"]
spot["USDTWD"] = spot_ndf["USDTWD"] - ndf["USDTWD"]/fwd_scale["USDTWD"]
spot["USDKRW"] = spot_ndf["USDKRW"] - ndf["USDKRW"]/fwd_scale["USDKRW"]
spot["USDPHP"] = spot_ndf["USDPHP"] - ndf["USDPHP"]/fwd_scale["USDPHP"]

# Load carry_adjusted_prices
px = read_prices("../../Dymon/Code Data/carry_adj_fx_returns.csv")
px.columns = [x[0] for x in px.columns]

# 5-day week, standardize to USD base
total_price_ori = px.pipe(fill_date).pipe(omit_trailing_na).pipe(omit_leading_na, how = "any").ffill(limit = 2)
total_price_usd = total_price_ori.pipe(standardize_series, "prices")[ALL_FX]
total_returns_ori = px.pipe(fill_date).pipe(omit_trailing_na).pipe(omit_leading_na, how = "any").pct_change(limit = 2)
total_returns_usd = total_returns_ori.pipe(standardize_series, "returns")[ALL_FX]

spot.loc[:"2005", ["USDINR", "USDMYR", "USDKRW", "USDPHP"]] = np.nan
spot.loc[:"2007-06-30", ["USDIDR"]] = np.nan
spot_price_ori = spot.pipe(fill_date).pipe(omit_trailing_na).pipe(omit_leading_na, how = "any").ffill(limit = 2)
spot_price_usd = spot_price_ori.pipe(standardize_series, "prices")[ALL_FX]
spot_returns_ori = spot.pipe(fill_date).pipe(omit_trailing_na).pipe(omit_leading_na, how = "any").pct_change(limit = 2)
spot_returns_usd = spot_returns_ori.pipe(standardize_series, "returns")[ALL_FX]

carry_returns_ori = total_returns_ori - spot_returns_ori
carry_returns_usd = total_returns_usd - spot_returns_usd

# Implied interest rate
one_month_fp_ori = new_raw_data.iloc[:,list(range(4,190,10))].pipe(fill_date).pipe(omit_trailing_na, "all")
two_month_fp_ori = new_raw_data.iloc[:,list(range(5,190,10))].pipe(fill_date).pipe(omit_trailing_na, "all")
three_month_fp_ori = new_raw_data.iloc[:,list(range(6,190,10))].pipe(fill_date).pipe(omit_trailing_na, "all")
twelve_month_fp_ori = new_raw_data.iloc[:,list(range(7,190,10))].pipe(fill_date).pipe(omit_trailing_na, "all")

one_month_fp_ori.loc[:"2011-03-01", "CNH1M CMPN Curncy"] = np.nan
one_month_fp_ori.loc[:"2007-05-23", "IHN1M CMPN Curncy"] = np.nan
one_month_fp_ori.loc[:"2000-02-01", "NTN1M CMPN Curncy"] = np.nan
one_month_fp_ori.loc[:"2006-01-01", ["IRN1M CMPN Curncy", "MRN1M CMPN Curncy", "KWN1M CMPN Curncy", "PPN1M CMPN Curncy"]] = np.nan

two_month_fp_ori.loc[:"2011-03-01", "CNH2M CMPN Curncy"] = np.nan
two_month_fp_ori.loc[:"2007-05-23", "IHN2M CMPN Curncy"] = np.nan
two_month_fp_ori.loc[:"2000-02-01", "NTN2M CMPN Curncy"] = np.nan
two_month_fp_ori.loc[:"2006-01-01", ["IRN2M CMPN Curncy", "MRN2M CMPN Curncy", "KWN2M CMPN Curncy", "PPN2M CMPN Curncy"]] = np.nan

three_month_fp_ori.loc[:"2011-03-01", "CNH3M CMPN Curncy"] = np.nan
three_month_fp_ori.loc[:"2007-05-23", "IHN3M CMPN Curncy"] = np.nan
three_month_fp_ori.loc[:"2000-02-01", "NTN3M CMPN Curncy"] = np.nan
three_month_fp_ori.loc[:"2006-01-01", ["IRN3M CMPN Curncy", "MRN3M CMPN Curncy", "KWN3M CMPN Curncy", "PPN3M CMPN Curncy"]] = np.nan


twelve_month_fp_ori.loc[:"2011-03-01", "CNH12M CMPN Curncy"] = np.nan
twelve_month_fp_ori.loc[:"2007-05-23", "IHN12M CMPN Curncy"] = np.nan
twelve_month_fp_ori.loc[:"2006-01-01", "IRN12M CMPN Curncy"] = np.nan
twelve_month_fp_ori.loc[:"2000-02-01", "NTN12M CMPN Curncy"] = np.nan
twelve_month_fp_ori.loc[:"2006-01-01", ["MRN12M CMPN Curncy", "KWN12M CMPN Curncy", "PPN12M CMPN Curncy"]] = np.nan

two_year_rates_ori = new_raw_data.iloc[:,list(range(8,190,10))].pipe(fill_date).pipe(omit_trailing_na, "any")
ten_year_rates_ori = new_raw_data.iloc[:,list(range(9,190,10))].pipe(fill_date).pipe(omit_trailing_na, "any")
############################################################
###################### SIGNAL HELPERS ######################
############################################################

def zscore(series, window, min_periods=None, typ="mean", method="simple"):
  if not min_periods:
    min_periods = window
  m = rolling_mean(series, window, min_periods, method)
  if typ == "median":
    m = rolling_median(series, window, min_periods)
  s = rolling_vol(series, window, min_periods, method)
  z = (series-m)/s
  return z

def norm(series, window, method="simple"):
  return (series - rolling_min(series, window, method)) / (rolling_max(series, window, method) - rolling_min(series, window, method))

def rolling_mean(series, window, min_periods=None, method="simple"):
  if not min_periods:
    min_periods = window
  if method == "simple":
    return series.rolling(window=window, min_periods=min_periods).mean()
  elif method == "exponential":
    return series.ewm(halflife=window, min_periods=min_periods).mean()
  elif method == "expanding":
    return series.expanding(min_periods=min_periods).mean()

def rolling_median(series, min_periods, window):
  if not min_periods:
    min_periods = window
  return series.rolling(window=window, min_periods=min_periods).median()

def rolling_vol(series, window, min_periods=None, method="simple"):
  if not min_periods:
    min_periods = window
  if method == "simple":
    return series.rolling(window=window, min_periods=min_periods).std(ddof=0)
  elif method == "exponential":
    return series.ewm(halflife=window, min_periods=min_periods).std(ddof=0)
  elif method == "expanding":
    return series.expanding(min_periods=min_periods).std(ddof=0)

def rolling_min(series, window, method="simple"):
  if method == "simple":
    return series.rolling(window=window).min()
  elif method == "exponential":
    return series.ewm(halflife=window, min_periods=window).min()
  elif method == "expanding":
    return series.expanding(min_periods=window).min()

def rolling_max(series, window, method="simple"):
  if method == "simple":
    return series.rolling(window=window).max()
  elif method == "exponential":
    return series.ewm(halflife=window, min_periods=window).max()
  elif method == "expanding":
    return series.expanding(min_periods=window).max()

def cap_helper(series, cap):
  capped_series = copy(series)
  capped_series[capped_series < -cap] = -cap
  capped_series[capped_series > cap] = cap
  return capped_series

def CAP(adf, cap):
  res_df = adf.copy()
  for col in list(res_df):
    res_df[col] = cap_helper(res_df[col], cap)
  return res_df

def demean(series, window, method="simple"):
  return series - rolling_mean(series, window, method=method, min_periods=window)

def demean_pct(series, window, min_periods=None, method="simple"):
  ma = rolling_mean(series, window, method=method, min_periods=min_periods)
  return (series - ma)/ma

def demean_xs(df):
  return df.sub(df.mean(axis=1), axis=0)

# def off_xs(row, n=2):
#   order = np.argsort(row)
#   for off in list(range(n,len(row)-n)):
#     row[order[off]] = 0
#   return row

# def top_bottom_xs(data, n=2):
#   return data.apply(off_xs, axis=1, n=n)

def off_xs(row, n=None):
  new_row = []
  indices = []
  for i,x in enumerate(row):
    if pd.notna(x):
      new_row.append(x)
    else:
      indices.append(i)

  if not n:
    n = max(len(new_row)//4, 1)

  order = np.argsort(new_row)
  for off in list(range(n,len(new_row)-n)):
    new_row[order[off]] = 0
  for i in indices:
    new_row.insert(i, np.nan)
  return pd.Series(new_row)

def top_bottom_xs(data, n=None):
  cols = data.columns
  new_data = data.apply(off_xs, axis=1, n=n)
  new_data.columns = cols
  return new_data

def hml(row, n=None, rank=False):
  new_row = []
  indices = []
  for i,x in enumerate(row):
    if pd.notna(x):
      new_row.append(x)
    else:
      indices.append(i)

  if not n:
    n = max(len(new_row)//4, 1)

  order = np.argsort(new_row)
  for off in list(range(n,len(new_row)-n)):
    new_row[order[off]] = 0
  for j in range(n):
    if rank:
      new_row[order[j]] = (-1) * (n-j)
      new_row[order[(j+1)*-1]] =  (n-j)
    else:
      new_row[order[j]] = -1
      new_row[order[(j+1)*-1]] = 1

  for i in indices:
    new_row.insert(i, np.nan)
  return pd.Series(new_row)

def high_minus_low(data, n=None, rank=False):
  cols = data.columns
  new_data = data.apply(hml, axis=1, n=n, rank=rank)
  new_data.columns = cols
  return new_data

############################################################
###################### REGIME DATASET ######################
############################################################
# GFC
gfc_regime = pd.DataFrame(pd.date_range(start="2001-01-01",end="2017-12-31")).set_index(0)
gfc_regime['regime'] = np.where(gfc_regime.index < datetime(2007,8,1), 'pre-GFC',
                          np.where(gfc_regime.index > datetime(2009,4,30), 'post-GFC','GFC'))
gfc_regime = gfc_regime['regime']
gfc_regime.index = pd.to_datetime(gfc_regime.index)
gfc_regime = gfc_regime.pipe(fill_date).fillna(method='bfill').fillna(method='ffill')

# Bull/bear & steep/flat
rates_regime = pd.read_csv("../../Dymon/Code Data/us_govs.csv", index_col=0).iloc[2:,[0,3]]
rates_regime[["USGG3M Index", "USGG10Y Index"]] = rates_regime[["USGG3M Index", "USGG10Y Index"]].apply(pd.to_numeric)
rates_regime = rates_regime.dropna(how='any')
rates_regime["10Y-3M"] = rates_regime["USGG10Y Index"]-rates_regime["USGG3M Index"]
rates_regime["3M_change"] = rates_regime['USGG3M Index'] - rates_regime['USGG3M Index'].shift(1)
rates_regime["10Y_change"] = rates_regime['USGG10Y Index'] - rates_regime['USGG10Y Index'].shift(1)
rates_regime['12M_MA_spread'] = rates_regime['10Y-3M'].pipe(rolling_mean, 252)

rates_regime['regime'] = np.where((rates_regime['10Y-3M'] > rates_regime['12M_MA_spread']) & (rates_regime["3M_change"] < 0) & (abs(rates_regime["3M_change"]) > rates_regime["10Y_change"]), "bull-steepening",
                np.where((rates_regime['10Y-3M'] > rates_regime['12M_MA_spread']) & (rates_regime["10Y_change"] > 0) & (rates_regime["10Y_change"] > abs(rates_regime["3M_change"])), "bear-steepening",
                np.where((rates_regime['10Y-3M'] < rates_regime['12M_MA_spread']) & (rates_regime["3M_change"] > 0) & (rates_regime["3M_change"] > abs(rates_regime["10Y_change"])), "bull-flattening",
                np.where((rates_regime['10Y-3M'] < rates_regime['12M_MA_spread']) & (rates_regime["10Y_change"] < 0) & (abs(rates_regime["10Y_change"]) > rates_regime["3M_change"]), "bear-flattening", None))))

rates_regime.index = pd.to_datetime(rates_regime.index)
rates_regime = rates_regime['regime'].pipe(fill_date).fillna(method='bfill').fillna(method='ffill')
# rates_regime = pd.DataFrame(pd.date_range(start="2001-01-01",end="2017-12-31")).set_index(0).join(rates_regime)
# rates_regime = rates_regime.fillna(method='bfill').fillna(method='ffill')
# rates_regime = rates_regime['regime']

# Volatility cycle
vix_data = pd.read_csv("../../Dymon/Code Data/VIXCLS.csv", index_col=0)
vix_data = vix_data[vix_data.VIXCLS != "."]
vix_data = vix_data.astype({"VIXCLS": float})

vix_data['5Y_MA'] = vix_data.rolling(window=1260).mean()
vix_data['regime'] = np.where(vix_data['VIXCLS'] > vix_data['5Y_MA'], "high-volatility", "low-volatility")
# vol_regime = pd.DataFrame(pd.date_range(start="2001-01-01",end="2017-12-31")).set_index(0).join(vix_data)
# vol_regime = vol_regime.fillna(method='ffill').fillna(method='bfill')
# vol_regime = vol_regime['regime']
vix_data.index = pd.to_datetime(vix_data.index)
vol_regime = vix_data['regime'].pipe(fill_date).fillna(method='bfill').fillna(method='ffill')

# Economic cycle
oecd = pd.read_csv("../../Dymon/Code Data/OECD_CLI.csv",index_col=0)
oecd = oecd[['TIME', 'Value']]
oecd['TIME'] = pd.to_datetime(oecd['TIME'] + '-1')
oecd = oecd.set_index('TIME')
oecd.groupby(['TIME']).mean()
oecd['12M_MA'] = oecd['Value'].pipe(rolling_mean, 12)
oecd['regime'] = np.where((oecd['Value'] > 100) & (oecd['Value'] > oecd['12M_MA']), "expansion",
                     np.where((oecd['Value'] > 100) & (oecd['Value'] < oecd['12M_MA']), "downturn",
                        np.where((oecd['Value'] < 100) & (oecd['Value'] > oecd['12M_MA']), "recovery",
                            np.where((oecd['Value'] < 100) & (oecd['Value'] < oecd['12M_MA']), "slowing down", None))))

cycle_regime = pd.DataFrame(pd.date_range(start="2001-01-01",end="2017-12-31")).set_index(0).join(oecd)
cycle_regime = cycle_regime['regime']
cycle_regime = cycle_regime.pipe(fill_date).fillna(method='ffill').fillna(method='bfill')
# cycle_regime = pd.DataFrame(pd.date_range(start="2001-01-01",end="2017-12-31")).set_index(0).join(oecd)
# cycle_regime = cycle_regime.fillna(method='bfill')
# cycle_regime = cycle_regime.fillna(method='ffill')
# cycle_regime = cycle_regime['regime']

# Macroeconomic framework
macro_regime = pd.DataFrame(pd.date_range(start="2001-01-01",end="2017-12-31")).set_index(0)
macro_regime['regime'] = np.where(macro_regime.index > datetime(2011,12,31), 'positive growth',
                  np.where((macro_regime.index > datetime(2007,12,31)) & (macro_regime.index < datetime(2010,1,1)), 'recession',
                  np.where((macro_regime.index > datetime(2003,12,31)) & (macro_regime.index < datetime(2007,1,1)), 'stagflation','inflationary')))
macro_regime = macro_regime['regime']
macro_regime.index = pd.to_datetime(macro_regime.index)
macro_regime = macro_regime.pipe(fill_date).fillna(method='bfill').fillna(method='ffill')



##########################################################
###################### SCALER LOGIC ######################
##########################################################

def calc_scaler(signal = None, asset_returns = None, asset_groups = None, settings = None, scaling_factor = None,
                scaling_target = None, scaling_width = None, scaling_method = None, scaling_cap = None):

  if scaling_target <= 0:
    raise Exception("`scaling_target` must be more than 0")
  if scaling_width <= 1:
    raise Exception("`scaling_width` must be more than 1")
  if scaling_method and scaling_method not in ["simple", "exponential"]:
    raise Exception("`scaling_method` can only be simple or exponential")

  if type(signal) == type(None):
    signal_returns = asset_returns.ffill()
  else:
    signal_returns = signal.shift(settings.implementation_lag + 1) * asset_returns.ffill()

  signal_return_grouped = pd.DataFrame()
  if asset_groups:
    print("    Grouping by " + str(list(asset_groups.keys())).replace("'",""))
    for key, value in asset_groups.items():
      signal_return_grouped[key] = signal_returns[value].sum(axis=1)
    vol_scalar = scaling_target/rolling_vol(signal_return_grouped,
                                            window = scaling_width,
                                            method = scaling_method)/sqrt(scaling_factor)
  else:
    vol_scalar = scaling_target/rolling_vol(signal_returns,
                                            window = scaling_width,
                                            method = scaling_method)/sqrt(scaling_factor)

  if asset_groups:
    vol_scalar_grouped = vol_scalar
    vol_scalar = pd.DataFrame()
    for sector in asset_groups.keys():
      for asset in asset_groups[sector]:
        vol_scalar[asset] = vol_scalar_grouped[sector]

  return vol_scalar


#############################################################
###################### SETTINGS OBJECT ######################
#############################################################

class Settings:
  def __init__(self, start_date = None, end_date = None, notional = 100, implementation_lag = 1, rebalance_period = "daily", scaling_factor = ANN_FACTOR,
               asset_scaling_method = None, asset_scaling_target = None, asset_scaling_width = None, asset_scaling_cap = None,
               sector_scaling_method = None, sector_scaling_target = None, sector_scaling_width = None, sector_scaling_cap = None,
               portfolio_scaling_method = None, portfolio_scaling_target = None, portfolio_scaling_width = None, portfolio_scaling_cap = None, use_cov=False):

    self.start_date = start_date
    self.end_date = end_date
    self.notional = notional
    self.implementation_lag = implementation_lag
    self.rebalance_period = rebalance_period.lower()
    self.use_cov = use_cov

    if self.implementation_lag < 0:
      raise Exception('`implementation_lag` must be more than 0 (cannot trade in the past)')

    if self.rebalance_period not in ["daily", "monday", "tuesday", "wednesday", "thursday", "friday"]:
      raise Exception('`rebalance_period` must be one of ["daily", "monday", "tuesday", "wednesday", "thursday", "friday"]')

    self.asset_scaling_method = asset_scaling_method
    self.asset_scaling_target = asset_scaling_target
    self.asset_scaling_width = asset_scaling_width
    self.asset_scaling_cap = asset_scaling_cap

    self.sector_scaling_method = sector_scaling_method
    self.sector_scaling_target = sector_scaling_target
    self.sector_scaling_width = sector_scaling_width
    self.sector_scaling_cap = sector_scaling_cap

    self.portfolio_scaling_method = portfolio_scaling_method
    self.portfolio_scaling_target = portfolio_scaling_target
    self.portfolio_scaling_width = portfolio_scaling_width
    self.portfolio_scaling_cap = portfolio_scaling_cap
    self.scaling_factor = scaling_factor


#############################################################
###################### BACKTEST ENGINE ######################
#############################################################

class Backtest:
  def __init__(self, signal = None, asset_returns = None, tcost = None, asset_groups = None, settings: Settings = None, settings_inject=None):
    self.signal = signal
    self.asset_returns = asset_returns
    self.tcost = tcost
    self.asset_groups = asset_groups
    self.settings = settings
    self.settings_inject = settings_inject

    settings_dict = self.settings.__dict__
    for key, value in settings_dict.items():
      setattr(self, key, value)

    if settings_inject:
      for key, value in settings_inject.items():
        if key not in settings_dict:
          print("[Invalid Parameter]: `{}` is not a valid Settings parameter, will be ignored".format(key))
        setattr(self, key, value)

    if (type(self.signal) != pd.DataFrame):
      raise Exception("Attribute `signal` must be a time series DataFrame")
    if (type(self.asset_returns) != pd.DataFrame):
      raise Exception("Attribute `asset_returns` must be a time series DataFrame")

    if (type(self.signal.index) != pd.core.indexes.datetimes.DatetimeIndex):
      raise Exception("Attribute `signal` must be a time series DataFrame")
    if (type(self.asset_returns.index) != pd.core.indexes.datetimes.DatetimeIndex):
      raise Exception("Attribute `asset_returns` must be a time series DataFrame")

    if (type(self.signal) == pd.DataFrame):
      if (sorted(self.signal.columns) != sorted(self.asset_returns.columns)):
        raise Exception("Mismatch of assets between `signal` and `asset_returns`")
      if (not set(self.asset_returns.columns).issubset(self.tcost.columns)):
        raise Exception("Missing `tcost` data for some assets")

    if getattr(self, "asset_groups", None) != None:
      if not set(self.signal).issubset([a for b in self.asset_groups.values() for a in b]):
        raise Exception("Some assets in `signal` do not belong in any `asset_groups`")

    if len(self.signal.columns) != len(self.asset_returns.columns):
      raise Exception("Mismatch of assets between signal and returns")
    elif sorted(self.signal.columns) != sorted(self.asset_returns.columns):
      raise Exception("Mismatch of assets between signal and returns")
    else:
      print("Re-arranging asset columns")
      self.signal = self.signal[self.asset_returns.columns]
      for col in self.asset_returns.columns:
        rtn_first_index = self.asset_returns[col].pipe(omit_leading_na).index[0]
        self.signal.loc[:rtn_first_index, col] = np.nan

  def run(self):
    if len(self.signal.columns) > 0 and self.asset_scaling_target:
      print("Asset scaling ...")
      self.asset_scaler = calc_scaler(asset_returns = self.asset_returns,
                                           scaling_target = self.asset_scaling_target,
                                           scaling_width = self.asset_scaling_width,
                                           scaling_method = self.asset_scaling_method,
                                           scaling_cap = self.asset_scaling_cap,
                                           scaling_factor = self.scaling_factor,
                                           settings = self.settings)
    else:
      self.asset_scaler = 1

    if self.sector_scaling_target and self.asset_groups:
      print("Sector scaling ...")
      self.sector_scaler = calc_scaler(signal = self.signal * self.asset_scaler,
                                            asset_returns = self.asset_returns,
                                            asset_groups = self.asset_groups,
                                            scaling_target = self.sector_scaling_target,
                                            scaling_width = self.sector_scaling_width,
                                            scaling_method = self.sector_scaling_method,
                                            scaling_cap = self.sector_scaling_cap,
                                            scaling_factor = self.scaling_factor,
                                            settings = self.settings)
    else:
      self.sector_scaler = 1

    if self.portfolio_scaling_target:
      if self.use_cov:
        print("Portfolio scaling (using cov-matrix)...")
        new_signal = self.signal * self.asset_scaler * self.sector_scaler
        input_data = self.asset_returns.loc[new_signal.index[0]:new_signal.index[-1]]
        window = self.portfolio_scaling_width
        df_new = pd.DataFrame(np.zeros((input_data.shape[0] - window + 1, 1)) )
        def rolling_target(window_data):
          main = input_data.iloc[window_data]
          main = main.dropna(axis=0, how="all").dropna(axis=1, how="any")

          tmp_cols = list(new_signal.iloc[window_data].tail(1).dropna(axis=1).columns)
          main = main[[x for x in list(main) if x in tmp_cols]]

          mean = main.mean(axis=0)
          res = main - mean
          cm = np.dot(res.T, res)

          last_weight = new_signal.iloc[window_data].tail(1)[main.columns]

          out = np.linalg.multi_dot([last_weight, cm, last_weight.T])
          df_new.iloc[int(window_data.iloc[0])] = np.sqrt(out[0])
          return True

        df_idx = pd.DataFrame(np.arange(input_data.shape[0]))
        _ = df_idx.rolling(window).apply(rolling_target)

        scaler = (self.portfolio_scaling_target/df_new).shift(self.implementation_lag + 1)
        scaler.index = input_data.tail(df_new.shape[0]).index
        self.portfolio_scaler = pd.concat([scaler for x in range(input_data.shape[1])], axis = 1)
        self.portfolio_scaler.columns = input_data.columns
        self.portfolio_scaler[np.isinf(self.portfolio_scaler)] = np.nan
        self.portfolio_scaler = self.portfolio_scaler.ffill()

      else:
        print("Portfolio scaling ...")
        self.portfolio_scaler = calc_scaler(signal = self.signal * self.asset_scaler * self.sector_scaler,
                                            asset_returns = self.asset_returns,
                                            asset_groups = {"ALL" : list(self.signal.columns)},
                                            scaling_target = self.portfolio_scaling_target,
                                            scaling_width = self.portfolio_scaling_width,
                                            scaling_method = self.portfolio_scaling_method,
                                            scaling_cap = self.portfolio_scaling_cap,
                                            scaling_factor = self.scaling_factor,
                                            settings = self.settings)
    else:
      self.portfolio_scaler = 1

    self.wts = self.signal * self.asset_scaler * self.sector_scaler * self.portfolio_scaler

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"]

    if self.rebalance_period in weekdays:
      self.wts = self.wts[self.wts.index.weekday == (weekdays.index(self.rebalance_period) - (self.implementation_lag))%5].pipe(fill_date, fill_forward=True)
    elif self.rebalance_period == "monthly":
      new_wts = self.wts.groupby(self.wts.index.to_period('M')).last()
      new_wts.index = self.wts.loc[self.wts.groupby(self.wts.index.to_period('M')).apply(lambda x: x.index.max())].index
      self.wts = new_wts.pipe(fill_date, True)

    self.start_date = max(pd.to_datetime(self.start_date), self.wts.index[0], self.asset_returns.index[0])
    self.end_date   = min(pd.to_datetime(self.end_date), self.wts.index[-1], self.asset_returns.index[-1])

    self.signal = self.signal.loc[self.start_date:self.end_date]
    self.wts = self.wts.loc[self.start_date:self.end_date]
    for col in list(self.wts):
      tmp = self.wts[col].pipe(omit_leading_na)
      if self.settings.asset_scaling_width:
        tmp.iloc[:self.settings.asset_scaling_width] = 0
      self.wts[col] = tmp.pipe(omit_leading_zeros)

    self.wts[np.isinf(self.wts)] = np.nan
    self.wts = self.wts.pipe(fill_date, True)
    self.asset_returns = self.asset_returns[self.start_date:self.end_date]

    self.asset_rtn = self.asset_returns * self.wts.shift(self.implementation_lag + 1)
    self.asset_rtn[np.isfinite(self.asset_rtn) == False] = 0
    self.asset_rtn = self.asset_rtn.loc[self.start_date:self.end_date]

    self.model_rtn = self.asset_rtn.sum(axis = 1).loc[self.start_date:self.end_date].pipe(omit_leading_trailing_zeros)
    self.pos_chg = self.wts.diff(self.implementation_lag + 1).loc[self.start_date:self.end_date]
    self.wts_tcost = abs(self.pos_chg * self.tcost).loc[self.start_date:self.end_date]

    self.asset_rtn_tc = (self.asset_rtn - self.wts_tcost).loc[self.start_date:self.end_date]
    self.model_rtn_tc = self.asset_rtn_tc.sum(axis = 1).loc[self.start_date:self.end_date].pipe(omit_leading_trailing_zeros)

    self.model_rtn = self.model_rtn.rename("Model").pipe(fill_date).pipe(omit_leading_na)
    self.model_rtn_tc = self.model_rtn_tc.rename("Model_TC").pipe(fill_date).pipe(omit_leading_na)
    self.asset_rtn = self.asset_rtn.pipe(fill_date).pipe(omit_leading_na, "any").pipe(omit_leading_zeros)
    # self.asset_rtn_tc = self.asset_rtn_tc.pipe(fill_date).pipe(omit_leading_na, "any")



###############################################################
###################### BACKTEST ANALYSIS ######################
###############################################################

def calc_win_loss(returns_series, zero_treat="neutral"):
  wins = len([x for x in returns_series if x > 0])/len(returns_series)
  neut = len([x for x in returns_series if x == 0])/len(returns_series)
  loss = len([x for x in returns_series if x < 0])/len(returns_series)

  if zero_treat == "neutral":
    if loss == 0:
      win_loss = (wins+1)/(loss+1)
    else:
      win_loss = wins/loss
  elif zero_treat == "loss":
    win_loss = wins/(loss+neut)
  elif zero_treat == "win":
    win_loss = (wins+neut)/loss
  return win_loss

def calc_hit_rate(returns_series, zero_treat="neutral"):
  pos = len([x for x in returns_series if x > 0])
  neg = len([x for x in returns_series if x < 0])
  neu = len([x for x in returns_series if x == 0])
  if zero_treat == "neutral":
    return pos/(pos + neg)
  elif zero_treat == "loss":
    return pos/len(returns_series)
  elif zero_treat == "win":
    return (pos+neg)/len(returns_series)


def get_summary(bt, tc=False):

  if tc:
    model_rtn = bt.model_rtn_tc
  else:
    model_rtn = bt.model_rtn

  ann_ret = np.mean(model_rtn) * ANN_FACTOR
  ann_vol = np.std(model_rtn) * np.sqrt(ANN_FACTOR)
  ann_sr  = ann_ret/ann_vol

  dollar_ann_ret = annualized_return(model_rtn, compound=True)/100
  dollar_ann_sr = dollar_sharpe(model_rtn)

  hit_rate = calc_hit_rate(model_rtn, zero_treat="neutral")
  wins = len([x for x in model_rtn if x > 0])/len(model_rtn)
  neut = len([x for x in model_rtn if x == 0])/len(model_rtn)
  loss = len([x for x in model_rtn if x < 0])/len(model_rtn)
  win_loss = calc_win_loss(model_rtn, zero_treat="neutral")

  cumret = (model_rtn).cumsum() + 1
  cumret[0] = 1
  max_dd = (cumret-cumret.cummax()).min()

  c_cumret = (1 + model_rtn).cumprod()
  c_cummax = c_cumret.cummax()
  c_dd = c_cumret/c_cummax - 1

  print("Returns   : " + str(ann_ret))
  print("Vol       : " + str(ann_vol))
  print("SR        : " + str(ann_sr))
  print("Max DD    : " + str(round(max_dd * 100, 2)))
  print("")
  print("C Returns : " + str(dollar_ann_ret))
  print("C SR      : " + str(dollar_ann_sr))
  print("C Max DD  : " + str(round(c_dd.min() * 100, 2)))
  print("")
  print("Hit rate  : " + str(round(hit_rate, 3)))
  print("W | N | L : " + str(round(wins, 2)) + " | " + str(round(neut, 2)) + " | " + str(round(loss, 2)))
  print("W/L Ratio : " + str(round(win_loss, 2)))

def full_summary(bt, time = "2001"):
  get_summary(bt, time)
  print("----------------------------------")
  get_summary(bt, time, tc=True)


def get_returns_stats(returns, position):
    '''
    For input DataFrame with daily return series for a number of assets, returns a DataFrame with
    sharpe, sortino, and max drawdown (percentage, start and end date) for each asset
    '''
    stats = pd.DataFrame()
    for ticker in returns.columns:
        ticker_stats = pd.Series()
        ticker_stats['annualized_return'] = annualized_return(returns[ticker])
        ticker_stats['sharpe'] = sharpe(returns[ticker])
        ticker_stats['dollar_sharpe'] = dollar_sharpe(returns[ticker])
        ticker_stats['trade_sharpe'] = sharpe(returns[ticker].loc[position[ticker] != 0])
        ticker_stats['long_sharpe'] = sharpe(returns[ticker].loc[position[ticker] > 0])
        ticker_stats['short_sharpe'] = sharpe(returns[ticker].loc[position[ticker] < 0])
        ticker_stats['sortino'] = sortino(returns[ticker])
        ticker_stats['max_drawdown'], ticker_stats['max_drawdown_start'], ticker_stats['max_drawdown_end'] = max_drawdown(returns[ticker])
        stats[ticker] = ticker_stats
    return stats


def sharpe(ret, rf = 0, obs_per_year = 252):
  return (ret - rf).mean() * np.sqrt(obs_per_year) / ret.std()

def dollar_sharpe(ret, rf = 0, obs_per_year = 252):
  dollar_ret = ret * (ret + 1).cumprod().shift(fill_value = 1)
  return (dollar_ret - rf).mean() * np.sqrt(obs_per_year) / dollar_ret.std()

def sortino(ret, rf = 0, obs_per_year = 252):
  neg_ret = ret.loc[ret < 0]
  if neg_ret.shape[0] > 0:
    return (ret - rf).mean() * np.sqrt(obs_per_year) / neg_ret.std() * 0.5
  else:
    return np.nan

def annualized_return(ret, obs_per_year = 252, compound = False):
  if compound is True:
    dollar_ret = (((1 + ret ).cumprod()).iloc[-1])**( 1/ (len(ret)/obs_per_year)) *100 -100
  else:
    dollar_ret = ret.mean()*(obs_per_year)*100
  return dollar_ret

def max_drawdown(ret, compound = False):
  # ret is given in decimals, as % of AUM.
  if compound is True:
    cumret = (1 + ret).cumprod()
    cummax = cumret.cummax()
    dd = cumret/cummax - 1
  else:
    cumret = (ret).cumsum() + 1
    cumret[0] = 1
    cummax = cumret.cummax()
    dd = cumret-cummax
  max_dd = dd.min()
  cummax_idx = cummax.index.to_series().loc[cummax == cumret].reindex(cumret.index).ffill()
  max_dd_end = cummax_idx.loc[dd == max_dd].index[-1]
  max_dd_start = cummax_idx.loc[max_dd_end]
  return max_dd * 100, max_dd_start.strftime("%Y-%m-%d"), max_dd_end.strftime("%Y-%m-%d")


################################################################
###################### PLOTTING FUNCTIONS ######################
################################################################

from matplotlib.ticker import FuncFormatter
import empyrical as ep

def two_dec_places(x, pos):
  return '%.2f' % x

def gen_plot_rtn(returns, sr_sort=False, rtn_sort=False, main=None, compound=True):
  if type(returns) != pd.DataFrame:
    returns = pd.DataFrame(returns.rename("Model")).pipe(fill_date).pipe(omit_leading_na, "any")
  else:
    returns = returns.pipe(fill_date).pipe(omit_leading_na, "any")
  ax = plt.gca()
  ax.set_xlabel('')
  ax.set_ylabel('Cumulative returns')
  ax.set_yscale('linear')

  y_axis_formatter = FuncFormatter(two_dec_places)
  ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

  for x, i in enumerate(returns.columns):
    if all(returns[i] == 0) or all(returns[i].isna()):
      returns = returns.drop([i], axis=1)
      continue
    if compound:
      asset_label = i + " | dSR: " + str(round(dollar_sharpe(returns[i]), 2)) + \
      " | rtn: " + str(round(annualized_return(returns[i]), 2)) + \
      " | vol: " + str(round(np.std(returns[i]) * np.sqrt(ANN_FACTOR) * 100, 2))
      ep.cum_returns(returns[i], 1.0).plot(lw=2, color=PALETTE[x], alpha=1, label=asset_label, ax=ax)
    else:
      asset_label = i + " | SR: " + str(round(sharpe(returns[i]), 2)) + \
      " | rtn: " + str(round(annualized_return(returns[i]), 2)) + \
      " | vol: " + str(round(np.std(returns[i]) * np.sqrt(ANN_FACTOR) * 100, 2))
      cumret = returns[i].cumsum() + 1
      cumret[0] = 1
      cumret.plot(lw=2, color=PALETTE[x], alpha=1, label=asset_label, ax=ax)
  handles, labels = ax.get_legend_handles_labels()

  if sr_sort:
    if compound:
      sharpes_order = np.argsort([dollar_sharpe(returns[i]) for i in list(returns)])
      leg = ax.legend(handles = [handles[x] for x in sharpes_order[::-1]],
                      labels = [labels[x] for x in sharpes_order[::-1]],
                      loc='upper left', frameon=True, framealpha=0.7)
    else:
      sharpes_order = np.argsort([sharpe(returns[i]) for i in list(returns)])
      leg = ax.legend(handles = [handles[x] for x in sharpes_order[::-1]],
                      labels = [labels[x] for x in sharpes_order[::-1]],
                      loc='upper left', frameon=True, framealpha=0.7)
  elif rtn_sort:
    returns_order = np.argsort([annualized_return(returns[i]) for i in list(returns)])
    leg = ax.legend(handles = [handles[x] for x in returns_order[::-1]],
                    labels = [labels[x] for x in returns_order[::-1]],
                    loc='upper left', frameon=True, framealpha=0.7)

  else:
    leg = ax.legend(loc='upper left', frameon=True, framealpha=0.7)

  for line in leg.get_lines():
    line.set_linewidth(4.0)

  ax.grid(linestyle = 'dashed', linewidth = 1, zorder = 1)
  ax.axhline(1.0, linestyle='--', color='red', lw=1, zorder = 2)
  if main:
    ax.set_title(main)
  return ax

def gen_drawdown(bt_obj, tc = False, main="Drawdown", compound=True):
  returns = getattr(bt_obj, "model_rtn_tc" if tc else "model_rtn")
  returns = returns.rename("Model").pipe(fill_date).pipe(omit_leading_na)
  ax = plt.gca()
  ax.set_xlabel('')
  ax.set_ylabel('Drawdown')
  ax.set_yscale('linear')

  if compound:
    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    label = "MDD: " + str(round(max_drawdown(returns, compound=True)[0], 2))
  else:
    df_cum_rets = (returns).cumsum() + 1
    df_cum_rets[0] = 1
    running_max = df_cum_rets.cummax()
    underwater = df_cum_rets - running_max
    label = "MDD: " + str(round(max_drawdown(returns, compound=False)[0], 2))
  (underwater).plot(kind='area', color=PALETTE[1], alpha=0.7, ax=ax, zorder = 3).set_title(main)

  handles = ax.get_legend_handles_labels()[0]
  ax.legend(handles = [handles[0]],
                    labels = [label],
                    loc='lower left', frameon=True, framealpha=0.7)

  ax.grid(linestyle = 'dashed', linewidth = 1, zorder = 1)
  ax.axhline(0.0, linestyle='--', color='red', lw=1, zorder = 2)

  return ax

def gen_rtn_longshort(bt_obj, tc = False, start_date=None, end_date=None, main="Returns by Long/Short", xs=False, compound=True):
  long_positions_index = bt_obj.wts.shift(bt_obj.implementation_lag + 1) > 0
  short_positions_index = bt_obj.wts.shift(bt_obj.implementation_lag + 1) < 0

  returns = getattr(bt_obj, "asset_rtn_tc" if tc else "asset_rtn")
  model = returns.sum(1).loc[start_date:end_date].pipe(omit_leading_trailing_zeros)
  if xs:
    returns = returns.pipe(demean_xs)
  longs  = returns[long_positions_index].fillna(0).sum(1).loc[start_date:end_date].pipe(omit_leading_trailing_zeros)
  shorts = returns[short_positions_index].fillna(0).sum(1).loc[start_date:end_date].pipe(omit_leading_trailing_zeros)

  long_short_df = pd.concat([model, longs, shorts], axis=1)
  long_short_df.columns = ["Model", "Long", "Short"]

  return gen_plot_rtn(long_short_df, sr_sort=False, compound=compound).set_title(main)

def gen_rtn_col_longshort(bt_obj, tc = False, main="Returns by Long/Short", compound=True):
  returns = getattr(bt_obj, "asset_rtn_tc" if tc else "asset_rtn")

  nrows = len(list(returns))//3 + 1
  gs = GridSpec(nrows = nrows, ncols = 3)
  fig = plt.figure(figsize=(20, nrows * 6))
  fig.suptitle(main, fontsize=30, y=.925)

  for x, col in enumerate(list(returns)):
    long_positions_index = bt_obj.wts[col] > 0
    short_positions_index = bt_obj.wts[col] < 0
    returns = getattr(bt_obj, "asset_rtn_tc" if tc else "asset_rtn")[col]

    shorts = returns[short_positions_index].fillna(0)
    longs  = returns[long_positions_index].fillna(0)

    long_short_df = pd.concat([returns, longs, shorts], axis=1)
    long_short_df.columns = ["Model", "Long", "Short"]

    fig.add_subplot(gs[x]).set_title(col).set_label(gen_plot_rtn(long_short_df, compound=compound))

def gen_rtn_component(bt_obj, spot_rtn, carry_rtn, tc = False, main="Returns by Spot/Carry", compound=True):
  returns = getattr(bt_obj, "model_rtn_tc" if tc else "model_rtn")
  spot_component = (bt_obj.wts.shift(bt_obj.implementation_lag + 1) * spot_rtn).sum(axis=1).loc["2006":]
  carry_component = (bt_obj.wts.shift(bt_obj.implementation_lag + 1) * carry_rtn).sum(axis=1).loc["2006":]
  total_component = returns.loc["2006":]

  comb = pd.concat([total_component, spot_component, carry_component], axis=1)
  comb.columns = ["Model", "Spot", "Carry"]

  return gen_plot_rtn(comb, compound=compound).set_title(main)

def gen_rtn_col_component(bt_obj, spot_rtn, carry_rtn, tc = False, main="Returns by Spot/Carry", compound=True):
  returns = getattr(bt_obj, "asset_rtn_tc" if tc else "asset_rtn")
  spot_component = (bt_obj.wts.shift(bt_obj.implementation_lag + 1) * spot_rtn).loc["2006":]
  carry_component = (bt_obj.wts.shift(bt_obj.implementation_lag + 1) * carry_rtn).loc["2006":]
  total_component = returns.loc["2006":]

  nrows = len(list(returns))//3 + 1
  gs = GridSpec(nrows = nrows, ncols = 3)
  fig = plt.figure(figsize=(20, nrows * 6))
  fig.suptitle(main, fontsize=30, y=.925)
  for x, col in enumerate(list(returns)):
    comb = pd.concat([total_component[col], spot_component[col], carry_component[col]], axis=1)
    comb.columns = ["Model", "Spot", "Carry"]
    fig.add_subplot(gs[x]).set_title(col).set_label(gen_plot_rtn(comb, compound=compound))

def gen_leadlag(bt_obj, nlag = 10, nlead = None, compound=True):
  if not nlead:
    nlead = - nlag
  n_range = range(nlead,nlag+1)
  if compound:
    model_rtns = [dollar_sharpe((bt_obj.wts.shift(x+1) * bt_obj.asset_returns).sum(axis=1).pipe(omit_leading_zeros)) for x in n_range]
  else:
    model_rtns = [sharpe((bt_obj.wts.shift(x+1) * bt_obj.asset_returns).sum(axis=1).pipe(omit_leading_zeros)) for x in n_range]
  color=['grey' if x != bt_obj.implementation_lag else 'red' for x in n_range]

  plt.bar(n_range, model_rtns, color=color)
  plt.xticks(n_range, n_range, rotation='horizontal')
  for i in range(len(n_range)):
    plt.text(n_range[i], model_rtns[i], round(model_rtns[i],2), ha="center")
  plt.title("Lead/Lag")

def gen_rolling_sharpe(returns, window=252, main="1Y Rolling Sharpe"):
  ax = plt.gca()
  y_axis_formatter = FuncFormatter(two_dec_places)
  ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

  rolling_sharpe_ts = returns.rolling(window).apply(sharpe)
  rolling_sharpe_ts.plot(alpha=1, lw=2, color=PALETTE[0], zorder = 3)

  ax.set_title(main + ", Average = " + str(round(rolling_sharpe_ts.mean(), 2)))
  ax.axhline(
      rolling_sharpe_ts.mean(),
      color='red',
      linestyle='--',
      lw=2)
  ax.axhline(0.0, color='black', linestyle='-', lw=2, zorder=2)

  ax.set_ylabel('Sharpe ratio')
  ax.set_xlabel('')
  ax.legend(['Sharpe', 'Average ({})'.format(str(round(rolling_sharpe_ts.mean(), 2)))], loc='upper left', frameon=True, framealpha=0.7)
  ax.grid(linestyle = 'dashed', linewidth = 1, zorder = 1)
  return ax

def gen_rolling_winloss(returns, window=252, main="1Y Rolling Win Loss"):
  ax = plt.gca()
  y_axis_formatter = FuncFormatter(two_dec_places)
  ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

  rolling_winloss_ts = returns.rolling(window).apply(calc_win_loss)
  rolling_winloss_ts.plot(alpha=1, lw=2, color=PALETTE[0], zorder = 3)

  ax.set_title(main + ", Average = " + str(round(rolling_winloss_ts.mean(), 3)))
  ax.axhline(
      rolling_winloss_ts.mean(),
      color='red',
      linestyle='--',
      lw=2)
  ax.axhline(1.0, color='black', linestyle='-', lw=2, zorder=2)

  ax.set_ylabel('Win-Loss Ratio')
  ax.set_xlabel('')
  ax.legend(['Win-Loss', 'Average ({})'.format(str(round(rolling_winloss_ts.mean(), 3)))], loc='upper left', frameon=True, framealpha=0.7)
  ax.grid(linestyle = 'dashed', linewidth = 1, zorder = 1)
  return ax

def gen_rolling_hitrate(returns, window=252, main="1Y Rolling Hit Rate"):
  ax = plt.gca()
  y_axis_formatter = FuncFormatter(two_dec_places)
  ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

  rolling_hitrate = returns.rolling(window).apply(calc_hit_rate)
  rolling_hitrate.plot(alpha=1, lw=2, color=PALETTE[0], zorder = 3)

  ax.set_title(main + ", Average = " + str(round(rolling_hitrate.mean(), 3)))
  ax.axhline(
      rolling_hitrate.mean(),
      color='red',
      linestyle='--',
      lw=2)
  ax.axhline(0.5, color='black', linestyle='-', lw=2, zorder=2)

  ax.set_ylabel('Hit Rate')
  ax.set_xlabel('')
  ax.legend(['Hit Rate', 'Average ({})'.format(str(round(rolling_hitrate.mean(), 3)))], loc='upper left', frameon=True, framealpha=0.7)
  ax.grid(linestyle = 'dashed', linewidth = 1, zorder = 1)
  return ax

def gen_rtn_asset(bt_obj, tc=False, compound=True):
  returns = getattr(bt_obj, "asset_rtn_tc" if tc else "asset_rtn")
  return gen_plot_rtn(returns, sr_sort=True, main = "Asset Returns", compound=compound)

def gen_rtn_sector(bt_obj, tc=False, compound=True):
  returns = getattr(bt_obj, "asset_rtn_tc" if tc else "asset_rtn")
  asset_groups = bt_obj.asset_groups
  all_groups = []
  for a in asset_groups.keys():
    group = returns[asset_groups[a]]
    group_rtn = group.sum(axis=1).pipe(omit_leading_na).pipe(omit_leading_zeros)
    all_groups.append(group_rtn.rename(a))
  return gen_plot_rtn(pd.concat(all_groups,axis=1), sr_sort=True, main = "Sector Returns", compound=compound)

def gen_signal(bt_obj):
  ax = plt.gca()
  res = sns.heatmap(
          bt_obj.wts.T.fillna(0),
          alpha=1.0,
          center=0.0,
          cbar=False,
          cmap=RdGn,
          ax=ax)

  ax.set_yticks(np.arange(.5, len(list(bt_obj.wts)), 1))
  ax.set_yticklabels(list(bt_obj.wts), rotation=0)
  ax.set_yticks(np.arange(0, len(list(bt_obj.wts)), 1), minor=True)

  years = [x.year for x in bt_obj.wts.index]
  unique_years = sorted(set(years))
  year_loc = sorted([years.index(x) for x in unique_years])

  ax.xaxis.set_ticks(year_loc)
  ax.set_xticklabels(unique_years)

  ax.grid(which='minor', linestyle = 'dashed', linewidth = 1, zorder = 1, axis = "y")
  for _, spine in res.spines.items():
      spine.set_visible(True)

  ax.set_xlabel('Year')
  ax.set_title("Signal Heatmap")
  return ax

def plot_bt_old(bt_obj):
  gs = GridSpec(nrows = 4, ncols = 3)
  fig = plt.figure(figsize=(30, 30))
  fig.suptitle(t="Full Backtest", fontsize=30, y=.925)

  fig.add_subplot(gs[0]).set_label(gen_plot_rtn(bt_obj.model_rtn, main = "Returns (Full Period)"))
  fig.add_subplot(gs[1]).set_label(gen_plot_rtn(bt_obj.model_rtn["2012":], main = "Returns (2012/)"))
  fig.add_subplot(gs[2]).set_label(gen_rtn_asset(bt_obj))

  fig.add_subplot(gs[3]).set_label(pf.plot_drawdown_underwater(bt_obj.model_rtn))
  fig.add_subplot(gs[4]).set_label(gen_rtn_component(bt_obj, spot_returns_usd.loc[:"2016-12-31", bt_obj.wts.columns], carry_returns_usd.loc[:"2016-12-31", bt_obj.wts.columns]))
  fig.add_subplot(gs[5]).set_label(gen_rtn_longshort(bt_obj))

  fig.add_subplot(gs[6]).set_label(gen_leadlag(bt_obj, nlag=10))
  fig.add_subplot(gs[7]).set_label(gen_rolling_winloss(bt_obj.model_rtn))
  fig.add_subplot(gs[8]).set_label(gen_rolling_hitrate(bt_obj.model_rtn))

  fig.add_subplot(gs[9]).set_label(pf.plot_annual_returns(bt_obj.model_rtn))
  fig.add_subplot(gs[10:]).set_label(gen_signal(bt_obj))

def plot_bt(bt_obj, t="Full Backtest", compound=True):
  gs = GridSpec(nrows = 3, ncols = 4)
  fig = plt.figure(figsize=(34, 17))
  fig.suptitle(t=t, fontsize=30, y=.925)

  fig.add_subplot(gs[0]).set_label(gen_plot_rtn(bt_obj.model_rtn, main = "Returns (Full Period)", compound=compound))
  fig.add_subplot(gs[1]).set_label(gen_plot_rtn(bt_obj.model_rtn["2012":], main = "Returns (2012/)", compound=compound))
  fig.add_subplot(gs[2]).set_label(gen_rtn_component(bt_obj, spot_returns_usd.loc[:bt_obj.model_rtn.index[-1], bt_obj.wts.columns], carry_returns_usd.loc[:bt_obj.model_rtn.index[-1], bt_obj.wts.columns], compound=compound))
  fig.add_subplot(gs[3]).set_label(gen_rtn_longshort(bt_obj, compound=compound))

  fig.add_subplot(gs[4]).set_label(gen_drawdown(bt_obj, compound=compound))
  fig.add_subplot(gs[5]).set_label(gen_leadlag(bt_obj, nlag=10, compound=compound))
  fig.add_subplot(gs[6]).set_label(pf.plot_annual_returns(bt_obj.model_rtn))

  try:
    fig.add_subplot(gs[7]).set_label(gen_rtn_asset(bt_obj, compound=compound))
  except:
    pass

  fig.add_subplot(gs[8]).set_label(gen_rolling_winloss(bt_obj.model_rtn))
  fig.add_subplot(gs[9]).set_label(gen_rolling_hitrate(bt_obj.model_rtn))
  fig.add_subplot(gs[10:]).set_label(gen_signal(bt_obj))

def plot_bt2(bt_obj, t="Full Backtest", compound=True):
  gs = GridSpec(nrows = 3, ncols = 4)
  fig = plt.figure(figsize=(34, 17))
  fig.suptitle(t=t, fontsize=30, y=.925)

  fig.add_subplot(gs[0]).set_label(gen_plot_rtn(bt_obj.model_rtn, main = "Returns (Full Period)", compound=compound))
  fig.add_subplot(gs[1]).set_label(gen_plot_rtn(bt_obj.model_rtn["2012":], main = "Returns (2012/)", compound=compound))
  fig.add_subplot(gs[2]).set_label(gen_rtn_component(bt_obj, spot_returns_usd.loc[:bt_obj.model_rtn.index[-1], bt_obj.wts.columns], carry_returns_usd.loc[:bt_obj.model_rtn.index[-1], bt_obj.wts.columns], compound=compound))
  fig.add_subplot(gs[3]).set_label(gen_rtn_longshort(bt_obj, compound=compound))

  fig.add_subplot(gs[4]).set_label(gen_drawdown(bt_obj, compound=compound))
  fig.add_subplot(gs[5]).set_label(gen_leadlag(bt_obj, nlag=10, compound=compound))
  fig.add_subplot(gs[6]).set_label(pf.plot_annual_returns(bt_obj.model_rtn))
  fig.add_subplot(gs[7]).set_label(gen_rtn_sector(bt_obj, compound=compound))

  fig.add_subplot(gs[8]).set_label(gen_rolling_winloss(bt_obj.model_rtn))
  fig.add_subplot(gs[9]).set_label(gen_rolling_hitrate(bt_obj.model_rtn))
  fig.add_subplot(gs[10:]).set_label(gen_signal(bt_obj))

def gen_regime_plot(model_rtn, regime_data):

  cum_returns = ep.cum_returns(model_rtn.pipe(omit_leading_na).pipe(omit_leading_zeros), 1.0)
  earliest_index = max(min(cum_returns.index), min(regime_data.index))
  latest_index = min(max(cum_returns.index), max(regime_data.index))

  returns_regime = pd.concat([model_rtn, cum_returns.loc[earliest_index:latest_index], regime_data.loc[earliest_index:latest_index]], axis=1)
  returns_regime.columns = ["returns", "cum_returns", "regime"]

  x = returns_regime.index
  y = returns_regime['cum_returns']

  ax = plt.gca()
  inxval = mdates.date2num(x) #convert dates to numbers first

  regime_names = returns_regime["regime"].unique()
  regime_mapper = {k:v for v,k in enumerate(regime_names)}
  returns_regime["regime_number"] = returns_regime["regime"].apply(lambda x: regime_mapper[x])

  num_classes = len(regime_names)
  palette = ['green', 'blue', 'red', 'yellow', 'purple', 'pink']
  cmap = ListedColormap(palette[:num_classes])
  norm = BoundaryNorm(range(num_classes+1), cmap.N)
  points = np.array([inxval, y]).T.reshape(-1,1,2)
  segments = np.concatenate([points[:-1],points[1:]], axis=1)

  lc = LineCollection(segments, cmap=cmap, norm=norm)
  lc.set_array(returns_regime['regime_number'])
  lc.set_linewidth(2)
  ax.add_collection(lc)

  loc = mdates.AutoDateLocator() #covert numbers back to date
  ax.xaxis.set_major_locator(loc)
  ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
  ax.autoscale_view()

  #add legend and labels
  ax.set_xlabel('Dates')
  ax.set_ylabel('Cumulative returns')

  regime_plot_handles = [Line2D([0], [0], label='{} regime'.format(regime_names[x]), color=palette[x]) for x in range(num_classes)]
  plt.legend(handles=regime_plot_handles,loc="upper left")

  ax.grid(linestyle = 'dashed', linewidth = 1, zorder = 1)
  ax.axhline(1.0, linestyle='--', color='red', lw=1, zorder = 2)

  return ax

def full_regime_plot(bt_obj, regime_data, t="Full Regime Plot"):
  gs = GridSpec(nrows = 2, ncols = 4)
  fig = plt.figure(figsize=(24, 12))
  fig.suptitle(t=t, fontsize=30, y=.925)

  regimes = regime_data.unique()
  fig.add_subplot(gs[0:4]).set_label(gen_regime_plot(bt_obj.model_rtn, regime_data))

  for i,r in zip(range(4,8)[:len(regimes)], regimes):
    fig.add_subplot(gs[i]).set_label(gen_plot_rtn((bt_obj.model_rtn * (regime_data == r)).loc[:bt_obj.model_rtn.index[-1]], main = "{} regime ({}%)".format(r, str(round(list(regime_data).count(r)/len(regime_data), 2)))))

########################################################
###################### COMBINE BT ######################
########################################################

class CombinedBacktest:
  def __init__(self, bt_list, bt_wts,
               asset_returns = None, tcost = None, asset_groups = None,
               settings: Settings = None, settings_inject=None,
               method="heuristic"):
    self.bt_list = bt_list
    self.bt_wts = bt_wts

    self.asset_returns = asset_returns
    self.tcost = tcost
    self.asset_groups = asset_groups
    self.settings = settings
    self.method = method
    self.settings_inject = settings_inject

    settings_dict = self.settings.__dict__
    for key, value in settings_dict.items():
      setattr(self, key, value)

    if settings_inject:
      for key, value in settings_inject.items():
        if key not in settings_dict:
          print("[Invalid Parameter]: `{}` is not a valid Settings parameter, will be ignored".format(key))
        setattr(self, key, value)

    if (type(self.asset_returns) != pd.DataFrame):
      raise Exception("Attribute `asset_returns` must be a time series DataFrame")
    if (type(self.asset_returns.index) != pd.core.indexes.datetimes.DatetimeIndex):
      raise Exception("Attribute `asset_returns` must be a time series DataFrame")
    if getattr(self, "asset_groups", None) != None:
      if not set(self.signal).issubset([a for b in self.asset_groups.values() for a in b]):
        raise Exception("Some assets in `signal` do not belong in any `asset_groups`")

    if self.method not in ["naive", "heuristic", "linreg"]:
      raise Exception("Ensemble method must be one of [`naive`, `heuristic`, `linreg`]")

    if self.method in ["naive", "linreg"] and self.bt_wts:
      print("Using `{}` method, `bt_wts` ignored".format(self.method))


    self.bt_signals = [x.wts for x in bt_list]
    all_assets = [list(x.columns) for x in self.bt_signals]
    self.unique_assets = list(set([a for b in all_assets for a in b]))
    for asset in self.unique_assets:
      if asset not in self.asset_returns.columns:
        raise Exception("`{}` not part of trading universe".format(asset))

    if self.method == "naive":
      self.bt_final_signal = pd.concat(self.bt_signals).groupby(level=0).mean()
    elif self.method == "heuristic":
      if len(self.bt_list) != len(self.bt_wts):
        raise Exception("Length mismatch between `bt_list` and `bt_wts`")
      if float(sum([Decimal(str(x)) for x in self.bt_wts])) != 1:
        raise Exception("`bt_wts` does not add up to 1")
      else:
        self.bt_final_signal = pd.DataFrame()
        weighted_signals = [self.bt_wts[x] * self.bt_signals[x] for x in range(len(bt_list))]
        self.bt_final_signal =  pd.concat(weighted_signals).groupby(level=0).sum()

  def run(self):
    self.bt_final = Backtest(signal = self.bt_final_signal, asset_returns = self.asset_returns, tcost = self.tcost, settings = self.settings, settings_inject=self.settings_inject)
    self.bt_final.run()

    self.wts = self.bt_final.wts
    self.asset_rtn = self.bt_final.asset_rtn

    self.pos_chg = self.bt_final.pos_chg
    self.wts_tcost = self.bt_final.wts_tcost

    self.model_rtn = self.bt_final.model_rtn
    self.model_rtn_tc = self.bt_final.model_rtn_tc

    self.asset_rtn = self.bt_final.asset_rtn
    self.asset_rtn_tc = self.bt_final.asset_rtn_tc

def cbt_corrplot(bt_objs, bt_names=None):
  tmp = pd.concat([x.model_rtn for x in bt_objs], axis=1)
  if bt_names:
    tmp.columns = bt_names
  else:
    tmp.columns = [str(x) for x in range(len(bt_objs))]
  ax = plt.gca()
  res = sns.heatmap(tmp.corr().iloc[::-1], annot=True, linewidths=1, linecolor="black",
              alpha=1.0,
              center=0.0,
              cbar=False,
              cmap=RdGn,
              ax=ax)
  for _, spine in res.spines.items():
        spine.set_visible(True)
  ax.set_title("Signal Correlation")

def cbt_rolling_cor(bt_objs, bt_names, window=252, main="Signal Rolling Correlation"):
  ax = plt.gca()
  ax.set_xlabel('')
  ax.set_ylabel('Correlation')
  ax.set_yscale('linear')
  y_axis_formatter = FuncFormatter(two_dec_places)
  ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

  peg = bt_objs[0].model_rtn
  data = pd.concat([peg.rolling(window).corr(x.model_rtn) for x in bt_objs[1:]], axis=1).pipe(fill_date, True)
  data.columns = bt_names[1:]
  data = data.pipe(omit_leading_na, "any")
  data[bt_names[0]] = 1
  data = data[bt_names]
  corr = pd.concat([x.model_rtn for x in bt_objs], axis=1).corr().iloc[0]
  data.columns = [a+" | "+str(round(b, 2)) for a,b in zip(data.columns, corr)]
  for x, i in enumerate(data.columns):
    data[i].plot(lw=2, color=PALETTE[x], alpha=0.7, label=i, ax=ax)
  ax.grid(linestyle = 'dashed', linewidth = 1, zorder = 1)
  ax.set_title(main)
  leg = ax.legend(loc='best', frameon=True, framealpha=0.7)
  return ax