from read_tools.data_dealer_csmar import *
#%%-------------------------------------------------------------------------------------------------#
"""存储中间/控制变量"""
def BASIC_controls():
    save_csv(df=ANALYST_y_Insert(), name='ANALYST_y.csv')#分析师跟踪人数，已补插值，LN（x+1）处理
    save_csv(df=INS_y(), name='INS_y.csv')#INS,机构投资者持股比例
    save_csv(df=SIZE_y(), name='SIZE_y.csv')#SIZE,已经对数化处理
    """重新调整代码的ACCM"""
    save_csv(df = ACCM_y(), name='ACCM_y.csv')
    save_csv(df=LEV_y(), name='LEV_y.csv')
    save_csv(df=ROA_TTM_y(), name='ROA_TTM_y.csv')
    save_csv(df=Oturnover_y(), name='Oturnover_y.csv')#Dturn
    save_csv(df=BP_y(), name='BP_y.csv')#BM
    save_csv(df=BP_y(recip=True), name='PB_y.csv')  # BM
    print('BASIC_controls:ALL DONE')

def Ind_control():
    save_csv(df=IND_nodummy_y(), name='IND_nodummy_y.csv')
# ------------------------------------
"""DUVOL&NCSKEW"""
"""本及下年度数据：由于是面板回归，所以只能回归一次，不能做两次数据"""
def Rweek_derivatives():
    save_csv(df = Reg_Wit_df(name="Rweek_w.xlsx",
                             ranging=(np.datetime64(datetime.datetime(2018,1,1)),np.datetime64(datetime.datetime(2021,12,31)))),
             name='Rweekdf18_21.csv'
             )
    save_csv(df=Wi_csv_y(name="Rweekdf18_21.csv"), name='Wi18_21.csv')
    save_csv(df=NCSKEW_y(name='Wi18_21.csv'), name='NCSKEW_y18_21.csv')
    save_csv(df=DUVOL_y(name='Wi18_21.csv'), name='DUVOL_y18_21.csv')
    save_csv(df=RET_SIGMA_y(name='Wi18_21.csv'), name='RETSIGMA_y18_21.csv')#不用切，合并时自动舍弃2021数据

def NCSKEW_DUVOL_splited():
    save_csv(df=CrashRisk_Timemerger(name='DUVOL_y18_21.csv',colname='DUVOL'), name='DUVOL_merge_y.csv')
    save_csv(df=CrashRisk_Timemerger(name='NCSKEW_y18_21.csv', colname='NCSKEW'), name='NCSKEW_merge_y.csv')

def Merger_csmar():
    save_csv(PATH=main_data_path, df=csv_merger(), name='csv_merger.csv')

if __name__ == '__main__':
    print('"csmar_dealer_main.py" activating as Entrance File...')
    """注意，由于缩尾时间不明，于是统一在合并最终表的时候进行缩尾"""
    # Try(BASIC_controls)
    # Try(Ind_control)
    # Try(Rweek_derivatives)
    # Try(NCSKEW_DUVOL_splited)
    """save_merger"""
    # Try(Merger_csmar)
    """FIN"""
    print('"csmar_dealer_main.py" activated as Entrance File')