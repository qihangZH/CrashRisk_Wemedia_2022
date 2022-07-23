import numpy as np
from .path.Path import *
from sklearn.linear_model import LinearRegression
from scipy.stats.mstats import winsorize
import pandas as pd
import datetime

#%%-------------------------------------------------------------------------------------------------#
"""
基础函数
"""
def rd_xlsx(PATH = csmar_path, name=None, **kwargs):
    """
    :param PATH: 路径
    :param name: 名称，但是要加.xlsx结尾
    :param kwargs: 其他基于pd.read_xlsx的参数
    :return:
    """
    df = pd.read_excel(PATH + name, engine='openpyxl', **kwargs)
    return df

def rd_csv(PATH=save_path, name = None,**kwargs):

    df = pd.read_csv(PATH + name, **kwargs)
    df = index_reseter(df)
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0']
    return df

def df_colchanger_npdt64(df,Timecol='Time',delTimecol=False):
    """这个函数出现的目的是解决读入csv但是不能识别time为datetime64的问题"""
    df[Timecol] = str_time(df[Timecol], parse='-')
    df['year'] = df[Timecol].apply(lambda x: x.year)
    if delTimecol:
        del df[Timecol]
    return df

def rd_BBS_xlsx(PATH = bbs_path, name="SE_test0.xlsx",dtype= csmar_dtype_changer,skiprows=[1,2]):
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype,skiprows = (lambda x: x in skiprows))
    df.rename(columns=csmar_name_changer, inplace=True)
    df = df[['Time', 'Stkcd','PostSource',
             'TotalPosts', 'AvgReadings',
             'BullishPosts', 'AvgBullishReadings',
             'NeutralPosts', 'AvgNeutrReadings',
             'BearishPosts', 'AvgBearishReadings']]
    df['year'] = df['Time'].apply(lambda x: x.year)
    return df

def save_csv(df,PATH=save_path,name=None):
    df.to_csv(PATH+name)

def index_reseter(df):
    dfreturn = df.reset_index().copy()
    del dfreturn['index']
    return dfreturn

def reg_residual(y,x,fit_intercept=True):
    regression = LinearRegression(fit_intercept=fit_intercept)  # 有截距
    regression.fit(x, y)
    y_pred = regression.predict(x)
    redisual = y - y_pred
    return redisual

def drop_firm(df,Stkcd='Stkcd',Return='Rweekbonus',Time='Time',nan_thresh=30):
    """建立阈值，设置检验nan后剔除数据的情况"""
    firm_set = list(set(df[Stkcd]))
    Time_set = list(set(df[Time].apply(lambda x: x.year)))

    ser_list = []
    for i in firm_set:
        ser = df[[Return,Time]][df[Stkcd]==i]
        for j in Time_set:
            serj = ser[Return][(ser[Time].apply(lambda x: x.year))==j]
            if sum(np.isnan(serj)) >=nan_thresh:
                ser_list.append(i)
                break

    return ser_list

def df_drop_firm(df,droplist,Stkcd='Stkcd'):
    ser = np.full(df.shape[0],False)
    for i in droplist:
        seri = (df[Stkcd]==i)
        ser = ser | seri
    return df[~ser]

def nanaverage(a,weights):
    sera = ~(np.isnan(a) | np.isnan(weights))
    return np.average(a=a[sera],weights=weights[sera])

def str_time(columns,parse='-'):
    """
    :param columns: pd.Series,obj
    :return: pd.Series,datetime
    """
    columns = pd.to_datetime(columns, format='%Y'+parse+'%m'+parse+'%d')
    return columns

def nanmultiply_sum(ser1,ser2):
    return np.nansum(np.multiply(ser1,ser2))

def lnxplus1(x):
    return np.log(x+1)#ln(x+1)对数化

def choosingIND(ser):
    """选取频率最高的作为其行业代码"""
    name_set = np.array(list(set(ser)))
    name_set = name_set[~pd.isnull(name_set)]
    freq_set = [np.nansum(ser==i) for i in name_set]
    return name_set[np.argmax(freq_set)]

def fillser(df,ser1,ser2):
    #用ser2填充ser1

    innum = pd.isnull(df[ser1])
    df[ser1][innum] = df[ser2][innum]

    return df[ser1]

def maxs_places(Series,thresh=0.1,reverse=False):#必须输入Series
    # 以数组X为例，X[index1][index2]为一个新的临时数据，即与X无联系，相应的对其赋值对X无影响。
    top_k = int(np.round(len(Series)*thresh)+1)

    Val = Series.values
    top_k_idx = Val[~np.isnan(Val)].argsort()[::-1][0:top_k]
    ser_return = np.full(len(Val),False)
    #设置中间变量
    Middle = ser_return[~np.isnan(Val)]
    Middle[top_k_idx] = True
    if reverse:
        Middle = ~Middle
    ser_return[~np.isnan(Val)] = Middle
    ser_return = pd.Series(ser_return, index=Series.index)
    return ser_return

def year_changer(x,dict=next_year_changer):
    return dict[x]

#%%-------------------------------------------------------------------------------------------------#
def NCSKEW_builder(df,Wit_w='Wit_w'):
    """
    :param df: dataframe
    :param Wit_w: Wit_Weekly,Unsumed
    :return: NCSKEW_year
    """
    n = df.shape[0]
    SumWit3 = np.sum(np.power(df[Wit_w],3))
    SumWit2_1p5 = np.power(np.sum(np.power(df[Wit_w], 2)),1.5)
    NSCKEW = (-1*(SumWit3*n*np.power((n-1),(3/2)))/
               ((n-1)*(n-2)*SumWit2_1p5) )
    return NSCKEW

def DUVOL_builder(df,Wit_w='Wit_w',up='up',down='down'):
    """
    :param df: dataframe
    :param Wit_w: Wit_Weekly,Unsumed
    :param up: upper Rweek
    :param down: lower Rweek
    :return: DUVOL
    """
    nu = np.sum(df[up])
    ndown = np.sum(df[down])
    SumDownWit2 = np.sum(np.power(df[Wit_w],2)[df[down]])
    SumUpWit2 = np.sum(np.power(df[Wit_w], 2)[df[up]])
    UnloggedDUVOL = ((nu-1)*SumDownWit2)/((ndown-1)*SumUpWit2)
    DUVOL = np.log(UnloggedDUVOL)
    return DUVOL

#%%-------------------------------------------------------------------------------------------------#
"""Std_Winsor"""

def standardize(factor_ser):  # 标准化函数
    if len(factor_ser) <= 1:
        return factor_ser
    else:
        ser = np.array(factor_ser)
        return (ser - np.nanmean(ser)) / np.nanstd(ser)

def winsor_001(ser):
    ser = ser.copy()
    ser = np.where(ser.isnull(), np.nan, winsorize(np.ma.masked_invalid(ser), limits=(0.01, 0.01)))
    return ser
    # np.where(condition, x, y),满足condition是x，否则y
    # 此处判断是否空值，是的话为空，否的话进行屏蔽空值和无效值的1%和99%缩尾处理
def stargazer(x):
    """根据大小画***，**，*,分别p为0.01,0.05,0.1"""
    if np.abs(x)<=0.01:
        return '***'
    elif np.abs(x)<=0.05:
        return '**'
    elif np.abs(x) <= 0.1:
        return '*'
    else:
        return ''

def del_anyNarows(df):
    return df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#%%-------------------------------------------------------------------------------------------------#
def lineprinter(str):
    print("==============================================================================",'\n')
    print('         PRINTING:',str,',Please waiting...','\n')
    print("==============================================================================", '\n')
    return 0

def Try(func,**kwargs):
    try:
        func(**kwargs)
        print('done')
    except:
        print('Wrong in Try Func')

    return 0
#%%-------------------------------------------------------------------------------------------------#
print('"builder_basic.py" activated,done')