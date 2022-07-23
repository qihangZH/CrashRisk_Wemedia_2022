import numpy as np
import pandas as pd

from .builder_basic.builder_basic import *
from sklearn.impute import SimpleImputer
#%%-------------------------------------------------------------------------------------------------#
"""
year_data -> year_data
"""
"""没补入插值的ANALYST，暂不采用，采用外源版（即插值版）"""
def ANALYST_y(PATH = csmar_path, name="ANALYST_y.xlsx",dtype= csmar_dtype_changer):
    df = rd_xlsx(PATH = PATH, name=name,dtype=dtype)
    df.rename(columns=csmar_name_changer,inplace=True)

    df = index_reseter(df)
    return df

def ANALYST_y_Insert(PATH = csmar_path, name="ANALYST_y.xlsx",dtype= csmar_dtype_changer):
    #基于可补差值的2015~2021数据，且+1 ln
    df = rd_xlsx(PATH = PATH, name=name,dtype=dtype)
    df.rename(columns=csmar_name_changer,inplace=True)
    df['year'] = df['Time'].apply(lambda x:x.year)
    df = df[['Stkcd','year','ANALYST']].set_index(['Stkcd','year']).unstack(level='Stkcd')
    df = df.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).dropna(axis=1)
    df['ANALYST'] = df['ANALYST'].apply(lnxplus1)

    df = df.stack(level='Stkcd')
    df = df.reset_index()#[(df['year']<=2020)|(df['year']>=2018)]
    df = df[(df['year']<=2020)&(df['year']>=2018)]

    df = index_reseter(df)
    return df

def INS_y(PATH = csmar_path, name="INS_y.xlsx",dtype= csmar_dtype_changer):
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer,inplace=True)

    df['INS'] = df['INS']*0.01
    df = index_reseter(df)
    return df

def SIZE_y(PATH = csmar_path, name="SIZE_ACCM_y.xlsx",dtype= csmar_dtype_changer,ACCM_dict={'A':4,'B':3,'C':2,'D':1}):
    """
    不需要这里的ACCM，去除,本函数可以进行对数化处理
    """
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer,inplace=True)
    del df['ACCM']
    df['SIZE'] = df['SIZE'].apply(np.log)
    df = index_reseter(df)
    return df

def Year_y(PATH = csmar_path, name="ANALYST_y.xlsx",dtype= csmar_dtype_changer):
    #废弃，使用原生的dummy
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer, inplace=True)

    df = df[['Time','Stkcd']]
    Year_list = list(set(df['Time']))
    for i in Year_list:
        yeari = str(i.year) + '_y'
        df[yeari] = pd.Series(np.zeros(df.shape[0]))
        df[yeari][df['Time'] == i] = 1

    df = index_reseter(df)
    return df[['Time','Stkcd','2019_y','2020_y']]
#%%-------------------------------------------------------------------------------------------------#
"""
quartely_data -> year_data
"""
def LEV_y(PATH = csmar_path, name="LEV_q_AB.xlsx",Typerep='A',dtype= csmar_dtype_changer):
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer, inplace=True)
    df = df[df['Typrep']==Typerep]
    #资产负债率只留12月的数
    # compare_m = df['Time'].apply(lambda x:x.month)
    # df = df.iloc[compare_m==12,:]
    compare_m = df['Time'].apply(lambda x: x.month)
    df = df[compare_m == 12][['Stkcd','Time','LEV']]

    df = index_reseter(df)
    return df

def ROA_TTM_y(PATH = csmar_path, name="ROATTM_q_AB.xlsx",Typerep='A',dtype= csmar_dtype_changer):
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer, inplace=True)
    df = df[df['Typrep'] == Typerep]
    compare_m = df['Time'].apply(lambda x: x.month)

    df = df[compare_m == 12][['Stkcd','Time','ROA_TTM']]

    df = index_reseter(df)
    return df

def IND_y(PATH = csmar_path, name="IND_q.xlsx",dtype= csmar_dtype_changer):
    # 本函数输出dummy（对比IND_noind_y)
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer, inplace=True)
    compare_m = df['Time'].apply(lambda x: x.month)
    compare_d = df['Time'].apply(lambda x: x.day)
    df = df[(compare_m == 12)&(compare_d == 31)]

    Indcd = df.groupby('Stkcd').apply(lambda x: choosingIND(x['Indcd']))
    Indcd.name = 'Indcdin'
    Indcd = Indcd.reset_index()
    df = pd.merge(df,Indcd,on='Stkcd')

    df['Indcd'] = fillser(df,'Indcd','Indcdin')
    del df['Indcdin']

    INDsets = list(set(Indcd['Indcdin']))

    for i in INDsets:
        IND_i = 'Indcd_'+str(i)
        df[IND_i] = pd.Series(np.zeros(df.shape[0]))
        df[IND_i][df['Indcd']==i] = 1

    df = index_reseter(df)
    return df

def IND_nodummy_y(PATH = csmar_path, name="IND_q.xlsx",dtype= csmar_dtype_changer):
    #本函数不输出dummy（对比IND_y)
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer, inplace=True)
    compare_m = df['Time'].apply(lambda x: x.month)
    compare_d = df['Time'].apply(lambda x: x.day)
    df = df[(compare_m == 12)&(compare_d == 31)]

    Indcd = df.groupby('Stkcd').apply(lambda x: choosingIND(x['Indcd']))
    Indcd.name = 'Indcdin'
    Indcd = Indcd.reset_index()
    df = pd.merge(df,Indcd,on='Stkcd')

    df['Indcd'] = fillser(df,'Indcd','Indcdin')
    del df['Indcdin']

    df = index_reseter(df)
    return df

def BP_y(PATH = csmar_path, name="BP_q.xlsx",dtype= csmar_dtype_changer,recip=False):
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer, inplace=True)
    compare_m = df['Time'].apply(lambda x: x.month)
    compare_d = df['Time'].apply(lambda x: x.day)
    df = df[(compare_m == 12) & (compare_d == 31)]

    df = index_reseter(df)

    if recip:
        df['PB'] = 1 / df['BP']
        del df['BP']
    return df
#%%-------------------------------------------------------------------------------------------------#
"""
monthly_data -> year_data
"""
def Oturnover_y(PATH = csmar_path, name="Dturn_m.xlsx",dtype= csmar_dtype_changer):
    """
    DTURN，即第t年公司i的月平均换手率与第t-1年公司i月平均换手率的差额，基于csmar_Dturn
    """
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer, inplace=True)
    df['Time'] = pd.to_datetime(df['Trdmnt'] + '-01', format='%Y-%m-%d')

    # 计算
    df['year'] = df['Time'].apply(lambda x: x.year)
    df['month'] = df['Time'].apply(lambda x: x.month)

    month_turn = df[['Dturn','year','Stkcd']]
    year_turn = month_turn.groupby(['year', 'Stkcd']).mean()
    year_turn = year_turn.rename(columns={'Dturn':'tyDturn'})

    lyear_turn = year_turn.unstack(level='year').shift(axis=1, periods=1).stack()
    lyear_turn = lyear_turn.rename(columns={'tyDturn':'lyDturn'})
    df = pd.merge(lyear_turn,year_turn,on=['Stkcd','year'])

    df['Dturn'] = np.subtract(df['tyDturn'],df['lyDturn'])

    Dturn_df = df.reset_index()
    Dturn_df['year'] = Dturn_df['year'].apply(lambda x:
                                              np.datetime64(datetime.datetime(x, 12, 31)))

    Dturn_df = Dturn_df.rename(columns={'year': 'Time'})
    Dturn_df = index_reseter(Dturn_df)
    return Dturn_df[['Stkcd','Time','Dturn']]

#%%-------------------------------------------------------------------------------------------------#
"""
day_data -> year_data
"""
#暂不采用:
"""本函数由于源数据数据不足，2018数据缺失不做考虑"""
def Dturn_y_droped(PATH = csmar_path, name_start="PB_Turnover_d",dtype= csmar_dtype_changer):
    """
    DTURN，即第t年公司i的月平均换手率与第t-1年公司i月平均换手率的差额
    """
    df_list = []
    for i in range(4):
        dfi = rd_xlsx(PATH=PATH, name=name_start+'_'+ str(i) +'.xlsx', dtype=dtype)
        df_list.append(dfi)
    df = pd.concat(df_list, ignore_index=True)
    df.rename(columns=csmar_name_changer, inplace=True)
    #计算
    df['year'] = df['Time'].apply(lambda x: x.year)
    df['month'] = df['Time'].apply(lambda x: x.month)

    month_turn = df.groupby(['year','month','Stkcd']).aggregate({'Turnover':np.sum})['Turnover']
    year_turn = month_turn.groupby(['year','Stkcd']).mean()
    lyear_turn = year_turn.unstack(level='year').shift(axis=1,periods=1).stack()
    Dturn = year_turn - lyear_turn

    Dturn.name = 'Dturn'
    Dturn_df = Dturn.reset_index()
    Dturn_df['year'] = Dturn_df['year'].apply(lambda x:
                           np.datetime64(datetime.datetime(x, 12, 31)))

    Dturn_df = Dturn_df.rename(columns={'year':'Time'})
    Dturn_df = index_reseter(Dturn_df)
    return Dturn_df

#暂不采用:
"""本函数由于源数据数据不足，2018数据缺失不做考虑"""
def PB_y_droped(PATH = csmar_path, name_start="PB_Turnover_d",dtype= csmar_dtype_changer,recip=False):
    df_list = []
    for i in range(4):
        dfi = rd_xlsx(PATH=PATH, name=name_start + '_' + str(i) + '.xlsx', dtype=dtype)
        df_list.append(dfi)
    df = pd.concat(df_list, ignore_index=True)
    df.rename(columns=csmar_name_changer, inplace=True)
    #计算
    df['day'] = df['Time'].apply(lambda x: x.day)
    df['month'] = df['Time'].apply(lambda x: x.month)
    df = df[(df['day']==31) & (df['month']==12)][['Stkcd','Time','PB']]

    #recip?
    if recip:
        df['PB'] = 1/df['PB']

    df = index_reseter(df)
    return df
#%%-------------------------------------------------------------------------------------------------#
"""week_data -> year_data"""
#%%-------------------------------------------------------------------------------------------------#
"""RET/SIGMA"""
def Rit_w(PATH = csmar_path, name="Rweek_w.xlsx",dtype= csmar_dtype_changer):
    """实际上它是处理过时间的矩阵，仅此而已"""
    df = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    df.rename(columns=csmar_name_changer, inplace=True)

    before = df['Trdwnt'].str.extract(r'^(.*-).*').loc[:,0]
    after = pd.to_numeric((df['Trdwnt'].str.extract(r'-(.*)$')).loc[:,0])-1
    after_str = after.apply(str)
    df['Trdwnt'] = before+after_str

    df['Time'] = pd.to_datetime(df['Trdwnt']+'-1', format='%Y-%W-%w')

    df['after'] = after
    df = df.loc[df['after']!=0,:]

    del df['Trdwnt']
    del df['after']

    df = index_reseter(df)
    return df
#%%-------------------------------------------------------------------------------------------------#
"""以下算的不对，暂不采用"""
def SIGMA_y(PATH = csmar_path, name="Rweek_w.xlsx",dtype= csmar_dtype_changer):
    """公司年平均周收益标准差"""#写错了，外援重算
    Rit = Rit_w(PATH=PATH, name=name, dtype=dtype)
    df = Rit[['Stkcd','Time','Rweek']]
    df['year'] = df['Time'].apply(lambda x: x.year)
    df = df[['Stkcd','year','Rweek']].groupby(['year','Stkcd']).apply(np.nanstd)

    df.name = 'RET'
    df = df.reset_index()
    return df

def RET_y(PATH = csmar_path, name="Rweek_w.xlsx",dtype= csmar_dtype_changer):
    """公司年平均周收益率"""#写错了，外源重新计算
    Rit = Rit_w(PATH=PATH, name=name, dtype=dtype)
    df = Rit[['Stkcd','Time','Rweek']]
    df['year'] = df['Time'].apply(lambda x: x.year)
    df = df[['Stkcd','year','Rweek']].groupby(['year','Stkcd']).apply(np.nanmean)

    df.name = 'RET'
    df = df.reset_index()
    return df
#%%-------------------------------------------------------------------------------------------------#
"""CrashRisk"""
def Rmt_w(PATH = csmar_path, name="Rweek_w.xlsx",dtype= csmar_dtype_changer):
    """中间变量矩阵：得出Rmt"""
    Rit = Rit_w(PATH = PATH, name=name,dtype= dtype)
    df = Rit[['Stkcd','Time','SIZEweek','Rweek']]

    df_dealed = df.groupby('Time').apply(lambda x:nanaverage(x['Rweek'],weights=x['SIZEweek']))
    df_dealed.name = 'Rmweek'

    df = df_dealed.reset_index()
    return df

def Reg_Wit_df(PATH = csmar_path, name="Rweek_w.xlsx",dtype= csmar_dtype_changer,
               ranging=(np.datetime64(datetime.datetime(2018,1,1)),np.datetime64(datetime.datetime(2020,12,31)))
               ):
    """
    这个函数的意义是把回归用的df算出来，要不然一次次算太麻烦了，之后保存一下再继续做
    cite:Rit_w&Rmt_w
    且被引用函数用不上
    """
    Rmt = Rmt_w(PATH = PATH, name=name,dtype= dtype)
    Rmt['Rmweek+2'] = Rmt['Rmweek'].shift(periods=-2)
    Rmt['Rmweek+1'] = Rmt['Rmweek'].shift(periods=-1)
    Rmt['Rmweek-2'] = Rmt['Rmweek'].shift(periods=2)
    Rmt['Rmweek-1'] = Rmt['Rmweek'].shift(periods=1)

    Rit = Rit_w(PATH=PATH, name=name, dtype=dtype)
    df = Rit[['Stkcd', 'Time', 'Rweekbonus','Rweek']]#修改了一下
    df = pd.merge(df,Rmt,on='Time')
    # # 选取2018年及以后的数据
    df = df[(df['Time']>=ranging[0])&(df['Time']<=ranging[1])]

    drop_firm_list = drop_firm(df=df,Stkcd='Stkcd',Return='Rweekbonus',nan_thresh=90)
    print(drop_firm_list)
    df = df_drop_firm(df=df,droplist=drop_firm_list,Stkcd='Stkcd')

    df = index_reseter(df)
    return df

def Wi_csv_y(PATH = save_path, name="Reg_Wit_df_w.csv",dtype= csmar_dtype_changer):
    """
    Ri,t为第i个股票第t周考虑现金红利再投资的周收益率，Rm,t为市场中所有股票在第t周流通市值加权的平均收
    益率。接着，利用回归模型的残差，我们计算第i个股票t周的特有收益
    winsorize被进行（取消），且需要注意的是nan也全部被去除
    """

    df = rd_csv(PATH=PATH,name=name,dtype=dtype)
    df = index_reseter(df)
    df = df.dropna(axis=0)#去除行数据nan
    """统一缩尾"""
    # df.loc[:,'Time':] = df.loc[:,'Time':].apply(
    #     lambda x: winsorize(x,limits=(0.01,0.01)),axis=0)#winsor
    # #看似影响时间，但是并不影响！因为年度计算，且限制在2018~2021，每年都有大量数据，这就导致
    # #2018年数据不会进入2019，2021不会进入2020，done！
    y = df['Rweekbonus']
    x = df.loc[:,'Rmweek':]
    df['epsilonit_w'] = reg_residual(y,x,fit_intercept=True)#有截距回归

    df = index_reseter(df)
    df['Wit_w'] = np.log(np.add(df['epsilonit_w'],1))

    return df[['Stkcd','Time','Rweek','Rmweek','epsilonit_w','Wit_w']]

def NCSKEW_y(PATH = save_path, name="Wit_w.csv",dtype= csmar_dtype_changer):
    """
    到这里就比较麻烦了，需要一步一步的解决。
    就是先得出每年个股的交易，基于此进行分析。
    """
    df = rd_csv(PATH=PATH, name=name, dtype=dtype)
    df = index_reseter(df)

    df['Time'] = str_time(df['Time'],parse='-')
    df['year'] = df['Time'].apply(lambda x:x.year)
    #NCSKEW://builder_basic.NCSKEW_builder
    NCSKEW = df.groupby(['Stkcd','year']).apply(NCSKEW_builder)

    NCSKEW.name = 'NCSKEW'
    NCSKEW = NCSKEW.reset_index()
    NCSKEW = index_reseter(NCSKEW)
    return NCSKEW

def DUVOL_y(PATH = save_path, name="Wit_w.csv",dtype= csmar_dtype_changer):
    """
    同 NCSKEW_y
    """
    df = rd_csv(PATH=PATH, name=name, dtype=dtype)
    df = index_reseter(df)

    df['Time'] = str_time(df['Time'], parse='-')
    df['year'] = df['Time'].apply(lambda x: x.year)
    df['up'] = df['Rweek'] > df['Rmweek']
    df['down'] = df['Rweek'] < df['Rmweek']

    DUVOL = df.groupby(['Stkcd', 'year']).apply(DUVOL_builder)

    DUVOL.name = 'DUVOL'
    DUVOL = DUVOL.reset_index()
    DUVOL = index_reseter(DUVOL)
    return DUVOL

def CrashRisk_Timemerger(PATH=save_path, name=None, dtype=csmar_dtype_changer,colname=None):
    #拆解18-21crashrisk并回归表格
    dfs = rd_csv(PATH=PATH, name=name, dtype=dtype)
    #copy1
    df = dfs[(dfs['year'] <= 2021) & (dfs['year'] >= 2019)].copy()
    df['year'] = df['year'].apply(year_changer)

    dfcontrol = dfs[(dfs['year'] <= 2020) & (dfs['year'] >= 2018)].copy()
    dfcontrol = dfcontrol.rename(columns={colname:'Control'+colname})

    dfmerge = pd.merge(df, dfcontrol, on=['Stkcd', 'year'])

    dfmerge = index_reseter(dfmerge)
    return dfmerge

def RET_SIGMA_y(PATH=save_path, name='Wi18_21.csv', dtype=csmar_dtype_changer):
    Wit = rd_csv(PATH=PATH, name=name, dtype=dtype)
    Wit = df_colchanger_npdt64(Wit,Timecol='Time',delTimecol=False)
    SIGMA = Wit.groupby(['year','Stkcd']).aggregate({'Wit_w':np.nanstd})
    RET = Wit.groupby(['year','Stkcd']).aggregate({'Wit_w':np.nanmean})
    Wit = pd.merge(RET,SIGMA,left_index=True,right_index=True)
    Wit = Wit.reset_index()
    Wit = Wit.rename(columns={'Wit_w_x':'RET','Wit_w_y':'SIGMA'})

    return Wit
#%%-------------------------------------------------------------------------------------------------#
"""WIND 外源数据"""
"""ACCM_outsource"""
def ACCM_y(PATH=csmar_path, name='ACCM_y.xlsx', dtype=csmar_dtype_changer):
    a = rd_xlsx(PATH=PATH, name=name, dtype=dtype)
    a = a.set_index('Stkcd').stack().apply(np.absolute)
    a.name = 'ACCM'
    a = a.reset_index()
    a = index_reseter(a)
    a = a.rename(columns={'level_1':'year'})
    a['Stkcd'] = a['Stkcd'].str.extract(r'^(\d+).*')
    return a

#%%-------------------------------------------------------------------------------------------------#

"""all done"""
def csv_merger(PATH=save_path,name_list=csmarcsv_list,dtype= csmar_dtype_changer,on=('Stkcd','year'),how='left'):
    """输出整合结果"""
    matrix = None
    for i in name_list:
        dfi = rd_csv(PATH=PATH, name=i, dtype=dtype)
        dfi = index_reseter(dfi)
        if (not 'year' in dfi.columns) and 'Time' in dfi.columns:
            dfi['Time'] = str_time(dfi['Time'], parse='-')
            dfi['year'] = dfi['Time'].apply(lambda x: x.year)
            del dfi['Time']

        if matrix is None:
            matrix = dfi
        else:
            matrix = pd.merge(matrix,dfi,on=on,how=how)


    return matrix

print('"data_dealer_csmar.py" activated,done')
