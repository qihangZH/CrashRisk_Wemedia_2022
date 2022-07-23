import numpy as np
import pandas as pd

from .builder_basic.builder_basic import *
import statsmodels.api as sm
from linearmodels.panel.model import PanelOLS
#%%-------------------------------------------------------------------------------------------------#
class Reg:
    def __init__(self,BBSpath=main_data_path,BBSname='AT_EXINDAT_thr0.1.csv',
                 csmar_path=main_data_path,csmar_name='csv_merger.csv',
                 dtype=csmar_dtype_changer):
        #数据包导入
        self._csmar = rd_csv(PATH=csmar_path, name=csmar_name, dtype=dtype)#回归总数据集
        self._csmar = self._csmar.set_index(['Stkcd','year'])

        self._BBS = rd_csv(PATH=BBSpath, name=BBSname, dtype=dtype)
        self._BBS = self._BBS[(self._BBS['year']<=2020)&(self._BBS['year']>=2018)]
        self._BBS = self._BBS.set_index(['Stkcd', 'year'])

        self._data = pd.merge(self._csmar,self._BBS,
                              left_index=True,right_index=True,how='left')
        self._data = self._data.reset_index(level='year')

        """在这里处理AT,EXINDAT等，进行对数化处理"""
        self._data['AT'] = self._data['AT'].apply(lnxplus1)#处理AT，EXINDAT保持原始值
        #对数化-控制
        if 'EX_INDAT_read' in self._data.columns:
            self._data['EX_INDAT_read'] = (self._data['EX_INDAT_read']/self._data['SIZE'])

        """进行数据缩尾"""
        for i in winsor_list:
            self._data[i] = winsor_001(self._data[i])

        """存放Categorical化但是没有虚拟变量的数据"""
        self._data_Categorical = self._data.copy()

        self._data_Categorical['Indcd'] = pd.Categorical(self._data_Categorical['Indcd'])
        self._data_Categorical = self._data_Categorical[~self._data_Categorical.isin([np.nan, np.inf, -np.inf]).any(1)]
        self._data_Categorical = self._data_Categorical.reset_index().set_index(['Stkcd', 'year'])

        """去除Categorical化某些年份没有的数据,保证统一化"""
        choose = self._data_Categorical.copy()
        choose['choose'] = True
        choose = choose['choose']
        choose = choose.unstack(level='year').dropna(axis=0)[2018]
        choose.name = 'choose'
        self._data_Ctg_allyear = pd.merge(self._data_Categorical.reset_index().set_index(['Stkcd'])
                                          ,choose,on='Stkcd').reset_index().set_index(['Stkcd', 'year'])
        del self._data_Ctg_allyear['choose']

        """存放回归结果"""
        self.regFEM_Firm_result = {}
        self.regFEM_Ind_result = {}#空字典

    def cols_changer(self,colname,func):
        """利用函数来处理列"""
        self._data[colname] = self._data[colname].apply(func)
        self._data_Ctg_allyear[colname] = self._data_Ctg_allyear[colname].apply(func)
        self._data_Categorical[colname] = self._data_Categorical[colname].apply(func)

    def corr(self):
        cols = self._data_Categorical.columns
        if 'EX_INDAT' in cols:
            corr_df = self._data_Categorical[['AT','EX_INDAT','DUVOL','NCSKEW']]
        elif 'EX_INDAT_read' in cols:
            corr_df = self._data_Categorical[['AT', 'EX_INDAT_read', 'DUVOL', 'NCSKEW']]

        self.pearsoncorr = corr_df.corr(method='pearson')

        return self.pearsoncorr

    def regression_Categorical(self,crashrisk='NCSKEW',reg_method='FEM_Ind',Allyear=False):
        """
        :param crashrisk: 'NCSKEW' | 'DUVOL'
        :param reg_method: 'FEM_Ind' | 'FEM_Firm' , FEM controling the entity effect
        :return:Regression Summary, Besides return self.regFEM_result/regOLS_result as full reg return
        :Allyear:bool,True to choose firm have all three year data,else No
        """
        if Allyear:
            df = self._data_Ctg_allyear
        else:
            df = self._data_Categorical
        np.random.seed(20220306)
        regX = df.copy()
        del regX['NCSKEW'], regX['DUVOL'], regX['Indcd']

        if crashrisk == 'NCSKEW':
            del regX['ControlDUVOL']
            regX = regX.rename(columns={'ControlNCSKEW':'ControlCrashRisk'})
        elif crashrisk == 'DUVOL':
            del regX['ControlNCSKEW']
            regX = regX.rename(columns={'ControlDUVOL': 'ControlCrashRisk'})

        regX = sm.add_constant(regX)
        regY = df[crashrisk]
        Indcd = df['Indcd']
        # regXdummy = regX.reset_index()
        # regXdummy = pd.concat([regXdummy,pd.get_dummies(regXdummy['year'],drop_first=True)]
        #                       ,axis=1).set_index(['Stkcd', 'year'])
        if reg_method == 'FEM_Firm':
            results = PanelOLS(dependent=regY, exog=regX,
                               entity_effects=True,time_effects=True,
                               check_rank=False).fit(cov_type='kernel',kernel='bartlett')
            self.regFEM_Firm_result[crashrisk] = results
            return results

        elif reg_method == 'FEM_Ind':
            results = PanelOLS(dependent=regY, exog=regX,
                               entity_effects=False,
                               time_effects=True,check_rank=False,other_effects=Indcd).fit(cov_type='kernel',kernel='bartlett')
            self.regFEM_Ind_result[crashrisk] = results
            return results
        else:
            raise ValueError('Undefined Reg type')

    def SummarySer(self,RegResults,IsInd=None):
        """为SumDF提供基础,NameInput为输入的名字判断输出的ser"""
        N_obs = RegResults.nobs
        params = RegResults.params.round(3).apply(str)
        t_values = RegResults.tstats.round(2).apply(str)
        stars = RegResults.pvalues.round(3).apply(stargazer)
        ser = params+stars+'('+t_values+')'

        F = np.round(RegResults.f_statistic.stat,decimals=2)
        R2 = np.round(RegResults.rsquared,decimals=2)
        if IsInd:
           ser['Industry effect'] = 'Controled'
           ser['Firm Effect'] = 'Not Controled'
        else :
            ser['Industry effect'] = 'Not Controled'
            ser['Firm Effect'] = 'Controled'
        ser['Year'] = 'Controled'
        ser['R2'] = str(R2)
        ser['f_statistic'] = str(F)
        ser['N_obs'] = str(N_obs)
        return ser

    def SummaryDF(self):
        """
        必须完成所有的回归方式与crashrisk度量后才能进行本函数处理
        :return:regSummaryDF
        """
        if (not 'NCSKEW' in self.regFEM_Firm_result.keys()) or (not 'DUVOL' in self.regFEM_Firm_result.keys()):
            raise ValueError('OLS regression results of crashrisks are not full, Pls check Duvol or Ncskew')
        if (not 'NCSKEW' in self.regFEM_Ind_result.keys()) or (not 'DUVOL' in self.regFEM_Ind_result.keys()):
            raise ValueError('FEM regression results of crashrisks are not full, Pls check Duvol or Ncskew')

        FEM_Ind_NCSKEW = self.regFEM_Ind_result['NCSKEW']
        FEM_Ind_DUVOL = self.regFEM_Ind_result['DUVOL']
        FEM_Firm_NCSKEW = self.regFEM_Firm_result['NCSKEW']
        FEM_Firm_DUVOL = self.regFEM_Firm_result['DUVOL']

        loopdict = {'FEM_Ind_NCSKEW':FEM_Ind_NCSKEW,'FEM_Ind_DUVOL':FEM_Ind_DUVOL,
                    'FEM_Firm_NCSKEW':FEM_Firm_NCSKEW,'FEM_Firm_DUVOL':FEM_Firm_DUVOL
                    }
        IsIndlist = (True,True,False,False)
        ser_list =[]
        for i in range(len(loopdict)):
            lkeys = list(loopdict.keys())[i]
            ser = loopdict[lkeys]
            ser = self.SummarySer(RegResults=ser,IsInd=IsIndlist[i])
            ser.name = lkeys
            ser_list.append(ser)

        Summary_df = pd.concat(ser_list,axis=1)
        Summary_df = Summary_df['const':'N_obs']

        self.regSummaryDF = Summary_df
        return Summary_df

    def DescribeDF(self):
        """'EX_INDAT'专用，描述性统计"""
        df = self._data_Categorical.copy()
        df['EX_INDAT'] = df['EX_INDAT'].apply(lnxplus1)
        del df['Indcd']
        df = df.apply(axis=0,func=lambda x:
                      pd.Series([len(x),np.nanmean(x).round(2),
                                 np.nanstd(x).round(2),np.nanmax(x).round(2),np.nanmin(x).round(2),np.nanmedian(x).round(2)],
                                index=['N_obs','Mean','Std','Max','Min','Median'])).T
        df['N_obs'] = df['N_obs'].apply(int)

        self.describedf = df
        return df



#%%-------------------------------------------------------------------------------------------------#
print('"reg_builder.py" activated,done')

