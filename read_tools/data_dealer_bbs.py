import numpy as np
import pandas as pd
from .builder_basic.builder_basic import *
#%%-------------------------------------------------------------------------------------------------#
class BBS:
    def __init__(self,
                 BBS=bbs_path,BBSlist=("SE_test0.xlsx","SE_test1.xlsx"),BBS_skiprows=(1,2),BBS_thresh=0.1,BBS_Source='east',
                 IND=save_path,INDname='IND_nodummy_y.csv',delINDTimecol=False,
                 Csmar_PATH = main_data_path,Csmar_name='csv_merger.csv',
                 dtype=csmar_dtype_changer):
        #数据来源、极端值判断
        self.BBS_source = BBS_Source
        self.BBS_thresh = BBS_thresh

        bbs_dflist = []
        for i in BBSlist:
            dfi = rd_BBS_xlsx(PATH = BBS, name=i,dtype= dtype,skiprows=BBS_skiprows)
            bbs_dflist.append(dfi)
        self._bbs = pd.concat(bbs_dflist,axis=0,ignore_index=True)
        self._bbs['year'] = self._bbs['year'].astype(str)

        #数据来源
        self.east = self._bbs['PostSource'] == 1
        self.sina = self._bbs['PostSource'] == 2
        #公司数量-求均值
        self.corpnum = len(list(set(self._bbs['Stkcd'])))


        self._IND_y = rd_csv(PATH=IND,name=INDname,dtype=dtype)
        self._IND_y = df_colchanger_npdt64(self._IND_y,delTimecol=delINDTimecol)
        self._IND_y['year'] = self._IND_y['year'].astype(str)

        self.Indcd = self._IND_y[['Stkcd','year','Indcd']]
        self.Indnum = len(list(set(self.Indcd['Indcd'])))
        #输入Csmar数据：

        self._csmar = rd_csv(PATH=Csmar_PATH,name=Csmar_name,dtype=dtype)
        self._csmar['year'] = self._csmar['year'].astype(str)
        self.csmar_cols = list(self._csmar.columns)



    @property
    def IND_full(self):#返回全值
        return self._IND_y

    def BBS_dtypes(self):
        return self._bbs.dtypes

    def data_source(self):
        # count:TotalReading(year) on per stock
        if self.BBS_source == 'east':
            return self.east
        elif self.BBS_source == 'sina':
            return self.sina

    def BBS_Totalread(self,add_Attr = True):
        #帖子总阅读数:作为极端情况的
        ds = self.data_source()
        matrix=(self._bbs[ds]
                .groupby(['Stkcd','year']).apply(lambda x:nanmultiply_sum(x['TotalPosts'],x['AvgReadings'])))

        matrix.name = 'TotalRead'
        matrix = matrix.reset_index()
        matrix = index_reseter(matrix)

        if add_Attr:
            self.Totalread = matrix
        return matrix

    def BBS_AT(self,add_Attr = True):
        #唐2018——AT//方先明2020 Guba
        ds = self.data_source()
        matrix = (self._bbs[ds]
                  .groupby(['Stkcd', 'year']).apply(lambda x: np.nansum(x['TotalPosts'])))

        matrix.name = 'AT'
        matrix = matrix.reset_index()
        matrix = index_reseter(matrix)

        if add_Attr:
            self.AT = matrix
        return matrix

    def BBS_ATNEG(self,add_Attr = True):
        #唐2018——ATNEG看跌比率/简化版
        ds = self.data_source()
        matrix = (self._bbs[ds]
                  .groupby(['Stkcd', 'year']).apply(lambda x: np.divide(np.nansum(x['BearishPosts']),
                                                                        np.nansum(x['TotalPosts']))))

        matrix.name = 'ATNEG'
        matrix = matrix.reset_index()
        matrix = index_reseter(matrix)

        if add_Attr:
            self.ATNEG = matrix
        return matrix


    def BBS_ATPOS(self,add_Attr = True):
        # 唐2018——ATPOS看涨比率/简化版
        ds = self.data_source()
        matrix = (self._bbs[ds]
                  .groupby(['Stkcd', 'year']).apply(lambda x: np.divide(np.nansum(x['BullishPosts']),
                                                                        np.nansum(x['TotalPosts']))))

        matrix.name = 'ATPOS'
        matrix = matrix.reset_index()
        matrix = index_reseter(matrix)

        if add_Attr:
            self.ATPOS = matrix
        return matrix
#%%-------------------------------------------------------------------------------------------------#
    """
    我在这里说一下为什么要写两次，首先，第一个函数是基于单个年度的选取用于行业极端值计算的公司，一般来说就是reverse=False
    第二个函数为什么要设置就是设置用于回归的公司，这些公司最后用于Reg
    
    这个chooser是基于行业公司选择的，后面的是在最后merge的时候选择没有作为极端值数据参与计算行业值的公司，类似但不一样。
    """
    def BBS_threshchooser(self,BBS_df,reverse=False,columns_name='AT'):
        #False选择极端数据、True选择非极端数据 选取年度极端公司
        BBS_df = BBS_df.set_index(['Stkcd', 'year'])[columns_name].unstack('year')

        BBS_df = BBS_df.apply(lambda x: maxs_places(x, thresh=self.BBS_thresh,reverse=reverse))

        BBS_df = BBS_df.reset_index()
        BBS_df = index_reseter(BBS_df)
        return BBS_df

    def BBS_corpchooser(self,BBS_df,columns_name='AT',method='bitwise_or'):
        #为什么是or？因为只要被选中一次，它之后在非型数列中就都出现na了，数据就要被剔除，那么不如or多选点。
        BBS_df = BBS_df.set_index(['Stkcd', 'year'])[columns_name].unstack('year')
        BBS_df = BBS_df.apply(lambda x: maxs_places(x, thresh=self.BBS_thresh, reverse=False))

        columns_lists = list(BBS_df.columns)
        choose = None
        for i in columns_lists:
            if choose is None:
                choose = BBS_df[i]
            elif method == 'bitwise_and':
                choose = np.bitwise_and(choose,BBS_df[i])
            elif method == 'bitwise_or':
                choose = np.bitwise_or(choose, BBS_df[i])

        BBS_df['Ex'] = choose
        BBS_df['NotEx'] = ~choose

        BBS_df = BBS_df.reset_index()
        BBS_df = index_reseter(BBS_df)
        return BBS_df

    def BBS_EX_filter(self,df=None,df_name = None,reverse=False):
        #加入具有Indcd,year,以及需要处理的colname，如果错误就无法运行，除外，必须输入index为0~n类型的df
        Pre_matrix = df

        matrix = self.BBS_threshchooser(Pre_matrix,reverse=reverse,columns_name=df_name)
        #这一步计算得出公司某年度是否被选取
        matrix = matrix.set_index('Stkcd').stack(level='year')
        matrix.name = df_name

        Indus = self.Indcd.set_index(['Stkcd','year'])
        # 获取公司各年度的IndusCode
        matrix = pd.merge(matrix,Indus,left_index=True,right_index=True)

        matrix = matrix.groupby(by=['year','Indcd']).sum()

        matrix = matrix.reset_index()
        matrix = index_reseter(matrix)

        return matrix
#%%-------------------------------------------------------------------------------------------------#
    """极端关注公司投资者同行业关注度（无情绪因素）"""
    def BBS_EX_INDAT(self,add_Attr = True,colname='EX_INDAT',reverse=False):
        #极端关注公司投资者同行业关注度：计算每年度帖子量后，再计算年度最大的公司设置为1为极端代理变量，最后计算全行业代理变量之和
        ds = self.data_source()
        Pre_matrix = (self._bbs[ds]
                  .groupby(['Stkcd', 'year']).apply(lambda x: np.nansum(x['TotalPosts'])))
        Pre_matrix.name = colname
        Pre_matrix = Pre_matrix.reset_index()
        Pre_matrix = index_reseter(Pre_matrix)

        matrix = self.BBS_EX_filter(df=Pre_matrix,df_name=colname,reverse=reverse)

        if add_Attr:
            self.EX_INDAT = matrix

        return matrix

    def BBS_EX_INDAT_weighted(self,add_Attr = True,colname='EX_INDAT_read',weighting='AvgReadings',reverse=False):
        #极端关注公司投资者同行业关注度：计算每年度帖子量后，再计算年度最大的公司设置为1为极端代理变量，最后计算全行业代理变量之和
        ds = self.data_source()
        Pre_matrix = (self._bbs[ds].groupby(['Stkcd', 'year']).apply(lambda x: np.nansum(
                                                        np.multiply(x['TotalPosts'],x[weighting])
                                                                                        )))
        Pre_matrix.name = colname
        Pre_matrix = Pre_matrix.reset_index()
        Pre_matrix = index_reseter(Pre_matrix)

        matrix = self.BBS_EX_filter(df=Pre_matrix, df_name=colname, reverse=reverse)

        if add_Attr:
            self.EX_INDAT_read = matrix

        return matrix
#%%-------------------------------------------------------------------------------------------------#
    """极端关注公司投资者同行业情绪影响指标"""
    def BBS_EX_INDAT_POSNEG(self,add_Attr = True,Neg_colname='EX_INDATNEG',Pos_colname='EX_INDATPOS',reverse=False):
        #极端关注公司投资者同行业消费者情绪影响程度：计算每年度帖子量后，再计算年度最大的公司设置为1为极端代理变量，最后计算全行业代理变量之和
        ds = self.data_source()

        Pos_matrix = (self._bbs[ds]
                  .groupby(['Stkcd', 'year']).apply(lambda x: np.nansum(x['BullishPosts'])))
        Pos_matrix.name = Pos_colname
        Pos_matrix = Pos_matrix.reset_index()
        Pos_matrix = index_reseter(Pos_matrix)
        Pos_matrix = self.BBS_EX_filter(df=Pos_matrix,df_name=Pos_colname,reverse=reverse)

        Neg_matrix = (self._bbs[ds]
                      .groupby(['Stkcd', 'year']).apply(lambda x: np.nansum(x['BearishPosts'])))
        Neg_matrix.name = Neg_colname
        Neg_matrix = Neg_matrix.reset_index()
        Neg_matrix = index_reseter(Neg_matrix)
        Neg_matrix = self.BBS_EX_filter(df=Neg_matrix, df_name=Neg_colname, reverse=reverse)

        matrix = pd.merge(Pos_matrix,Neg_matrix,on=['year','Indcd'])

        if add_Attr:
            self.EX_INDAT_POSNEG = matrix

        return matrix

    #带其他权重情况情况：
    def BBS_EX_INDAT_POSNEG_weighted(self,add_Attr = True,Neg_colname='EX_INDATNEG_read',Pos_colname='EX_INDATPOS_read',reverse=False,
                                 Neg_weighting='AvgBearishReadings',Pos_weighting='AvgBullishReadings'):
        #极端关注公司投资者同行业消费者情绪影响程度：计算每年度帖子量后，再计算年度最大的公司设置为1为极端代理变量，最后计算全行业代理变量之和
        ds = self.data_source()

        Pos_matrix = (self._bbs[ds].groupby(['Stkcd', 'year']).apply(lambda x: np.nansum(
                                                        np.multiply(x['BullishPosts'],x[Pos_weighting])
                                                                                        )))
        Pos_matrix.name = Pos_colname
        Pos_matrix = Pos_matrix.reset_index()
        Pos_matrix = index_reseter(Pos_matrix)
        Pos_matrix = self.BBS_EX_filter(df=Pos_matrix,df_name=Pos_colname,reverse=reverse)

        Neg_matrix = (self._bbs[ds].groupby(['Stkcd', 'year']).apply(lambda x: np.nansum(
                                                        np.multiply(x['BearishPosts'],x[Neg_weighting])
                                                                                        )))
        Neg_matrix.name = Neg_colname
        Neg_matrix = Neg_matrix.reset_index()
        Neg_matrix = index_reseter(Neg_matrix)
        Neg_matrix = self.BBS_EX_filter(df=Neg_matrix, df_name=Neg_colname, reverse=reverse)

        matrix = pd.merge(Pos_matrix,Neg_matrix,on=['year','Indcd'])

        if add_Attr:
            self.EX_INDAT_POSNEG_weighted = matrix

        return matrix

#%%-------------------------------------------------------------------------------------------------#

    def corp_filter_ser(self,BBS_df,columns_name='AT'):
        #m某些年份的股票由于作为极端数据加入了计算，因此需要被剔除
        BBS_df = BBS_df.set_index(['Stkcd', 'year'])[columns_name].unstack('year')
        BBS_df = BBS_df.apply(lambda x: ~maxs_places(x, thresh=self.BBS_thresh, reverse=False))#这样可以选多点
        #说实话只用排除那些被选中的公司就完事了
        BBS_df = BBS_df.stack(level='year')
        BBS_df.name = 'choose'
        return BBS_df

    def corp_filter_merge(self,ser1,ser2,method='bitwise_and'):
        #选取被保留的股票
        BBS_df = pd.merge(ser1,ser2,left_index=True,right_index=True)

        columns_lists = list(BBS_df.columns)
        choose = None
        for i in columns_lists:
            if choose is None:
                choose = BBS_df[i]
            elif method == 'bitwise_and':
                choose = np.bitwise_and(choose, BBS_df[i])
            elif method == 'bitwise_or':
                choose = np.bitwise_or(choose, BBS_df[i])
        BBS_df['choose'] = choose
        BBS_df = BBS_df['choose'].reset_index()
        BBS_df = index_reseter(BBS_df)
        return BBS_df
#%%-------------------------------------------------------------------------------------------------#
    """拼接函数"""
    def Csmar_AT_IND_aggr(self,AT_like_df=None,ATIND_like_df=None,
                          AT_like_df_name=None,add_Attr = True):
        AT_like_df_choose = self.corp_filter_ser(AT_like_df, columns_name=AT_like_df_name)
        df = pd.merge(self._csmar,ATIND_like_df,on=['year','Indcd'],how='left')
        df = pd.merge(df,AT_like_df,on=['year','Stkcd'],how='left')
        df = pd.merge(df,AT_like_df_choose,on=['year','Stkcd'],how='left')

        df['choose'] = df['choose'].fillna(True)
        df = df[df['choose'] == True]

        del df['choose']
        del df['Indcd']

        if add_Attr:
            self.AggrDF = df

        return df

    def Csmar_AT_IND_aggr2(self,AT_like_df=None,AT_like_df_name=None,
                           AT_like_df2=None, AT_like_df_name2=None,
                           ATIND_like_df=None,add_Attr = True):

        AT_like_df_choose = self.corp_filter_ser(AT_like_df, columns_name=AT_like_df_name)
        AT_like_df_choose2 = self.corp_filter_ser(AT_like_df2, columns_name=AT_like_df_name2)

        AT_like_df_choose = self.corp_filter_merge(AT_like_df_choose,AT_like_df_choose2,
                                                   method='bitwise_and')

        df = pd.merge(self._csmar, ATIND_like_df, on=['year', 'Indcd'], how='left')
        df = pd.merge(df, AT_like_df, on=['year', 'Stkcd'], how='left')
        df = pd.merge(df, AT_like_df2, on=['year', 'Stkcd'], how='left')
        df = pd.merge(df, AT_like_df_choose, on=['year', 'Stkcd'], how='left')

        df['choose'] = df['choose'].fillna(True)
        df = df[df['choose']==True]
        del df['choose']
        del df['Indcd']

        if add_Attr:
            self.AggrDF2 = df

        return df
#%%-------------------------------------------------------------------------------------------------#\
    """无csmar版本以便后续调整"""
    def Csmar_AT_IND_aggr_nocsmar(self,AT_like_df=None,ATIND_like_df=None,
                          AT_like_df_name=None,add_Attr = True):
        AT_like_df_choose = self.corp_filter_ser(AT_like_df, columns_name=AT_like_df_name)
        df = pd.merge(self.Indcd,ATIND_like_df,on=['year','Indcd'],how='left')
        df = pd.merge(df,AT_like_df,on=['year','Stkcd'],how='left')
        df = pd.merge(df,AT_like_df_choose,on=['year','Stkcd'],how='left')

        df['choose'] = df['choose'].fillna(True)
        df = df[df['choose'] == True]

        del df['choose']
        del df['Indcd']

        if add_Attr:
            self.AggrDF_nocsmar = df

        return df

    def Csmar_AT_IND_aggr2_nocsmar(self,AT_like_df=None,AT_like_df_name=None,
                           AT_like_df2=None, AT_like_df_name2=None,
                           ATIND_like_df=None,add_Attr = True):

        AT_like_df_choose = self.corp_filter_ser(AT_like_df, columns_name=AT_like_df_name)
        AT_like_df_choose2 = self.corp_filter_ser(AT_like_df2, columns_name=AT_like_df_name2)

        AT_like_df_choose = self.corp_filter_merge(AT_like_df_choose,AT_like_df_choose2,
                                                   method='bitwise_and')

        df = pd.merge(self.Indcd, ATIND_like_df, on=['year', 'Indcd'], how='left')
        df = pd.merge(df, AT_like_df, on=['year', 'Stkcd'], how='left')
        df = pd.merge(df, AT_like_df2, on=['year', 'Stkcd'], how='left')
        df = pd.merge(df, AT_like_df_choose, on=['year', 'Stkcd'], how='left')

        df['choose'] = df['choose'].fillna(True)
        df = df[df['choose']==True]
        del df['choose']
        del df['Indcd']

        if add_Attr:
            self.AggrDF2_nocsmar = df

        return df

#%%-------------------------------------------------------------------------------------------------#
print('"data_dealer_bbs.py" activated,done')



