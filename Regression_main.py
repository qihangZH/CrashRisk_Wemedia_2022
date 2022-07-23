from read_tools.reg_builder import *
import os
print('"main.py" activating...')

def Regression_Saver(BBSnameStart='AT_EXINDAT_thr0.1',IsWeighted=False,EXINDATln=True,save_path=regression_path):
    """基于east-crashrisk-AT/EXINDAT的回归总结"""
    BBSname = BBSnameStart+'.csv'
    if IsWeighted:
        colnames = 'EX_INDAT_read'
    else:
        colnames = 'EX_INDAT'

    lineprinter(BBSnameStart)
    reg = Reg(BBSname=BBSname)

    if EXINDATln:
        reg.cols_changer(colname=colnames, func=lnxplus1)

    reg.corr()
    reg.regression_Categorical(Allyear=False, reg_method='FEM_Ind', crashrisk='NCSKEW')
    reg.regression_Categorical(Allyear=False, reg_method='FEM_Ind', crashrisk='DUVOL')
    reg.regression_Categorical(Allyear=False, reg_method='FEM_Firm', crashrisk='NCSKEW')
    reg.regression_Categorical(Allyear=False, reg_method='FEM_Firm', crashrisk='DUVOL')
    reg.SummaryDF()

    if EXINDATln:
        save_csv(df = reg.regSummaryDF,PATH = save_path,name='Log_Reg_df_'+BBSname)
        save_csv(df=reg.pearsoncorr, PATH=save_path, name='Log_corr_' + BBSname)
    else:
        save_csv(df=reg.regSummaryDF, PATH=save_path, name='Reg_df_' + BBSname)
        save_csv(df=reg.pearsoncorr, PATH=save_path, name='corr_' + BBSname)


if __name__ == '__main__':
    ATEXlist = pd.Series(os.listdir(main_data_path)).str.extract('(^AT.*)\.csv').dropna()[0]
    ATEXlist.name = 'ATEXlist'
    ATEX_Isweighted = ATEXlist.str.contains('weighted')
    for i in range(len(ATEXlist)):
        Regression_Saver(BBSnameStart=ATEXlist[i],IsWeighted=ATEX_Isweighted[i],EXINDATln=True,save_path=regression_NWest_path)
        Regression_Saver(BBSnameStart=ATEXlist[i], IsWeighted=ATEX_Isweighted[i], EXINDATln=False,save_path=regression_NWest_path)
    """描述性统计"""
    reg = Reg()
    reg.DescribeDF()
    save_csv(df=reg.describedf,PATH=regression_path,name='thr0.1_describe.csv')
    print('"main.py" activated as entrance file,done')

