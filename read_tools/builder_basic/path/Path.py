import numpy as np
import os
#%%-------------------------------------------------------------------------------------------------#
"""根目录为该脚本的曾祖目录"""
Entance_cwd = os.getcwd()#入口脚本的绝对目录
root_relative = os.path.abspath(os.path.dirname(__file__)+
                                       os.path.sep +'..'+
                                       os.path.sep +'..'+
                                       os.path.sep +'..')#本脚本的绝对目录的曾祖父目录,理论的root
# os.path.dirname(__file__)是一个绝对量，在导入的时候就会计算好
# 导入时，就算是全部导入，也有一个顺序，先计算参数，再导入，比如root_relative就是很明显的例子

if not Entance_cwd == root_relative:#不一致就报错
    raise ValueError('Wrong Entance File Place, Pleace Run in top of the Project')
else:
    root_path = root_relative+'/'
"""
数据PATH位置
"""
csmar_path = root_path+"read_tools/csmar_datas/"
save_path = root_path+"read_tools/dealed_datas/"
main_data_path = root_path+"read_tools/main_datas/"
bbs_path = root_path+"read_tools/bbs_datas/"
other_data_path = root_path+"read_tools/out_source_datas/"
regression_path = root_path+"read_tools/regression_output_datas/"
regression_cov_path = root_path+'read_tools/Regression_output_coventity/'
regression_heteroskedastic_path = root_path+'read_tools/Regression_output_heteroskedastic/'
regression_NWest_path = root_path+'read_tools/Regression_output_NeweyWest/'
bbs_start = "SE_InvestorSentimentSta"

#%%-------------------------------------------------------------------------------------------------#
"""csv_merger参数"""
csmarcsv_list = ['NCSKEW_merge_y.csv',
                 'DUVOL_merge_y.csv',
                 'ANALYST_y.csv',
                 'Oturnover_y.csv',
                 'RETSIGMA_y18_21.csv',
                 'PB_y.csv',
                 'ROA_TTM_y.csv',
                 'LEV_y.csv',
                 'SIZE_y.csv',
                 'ACCM_y.csv',
                 'INS_y.csv',
                 'IND_nodummy_y.csv']#CONTROL/NODUMMY

winsor_list = ['Dturn','PB','ROA_TTM','LEV','SIZE','AT','RET','SIGMA','ACCM','NCSKEW','DUVOL','INS',
               'ControlNCSKEW','ControlDUVOL']
#%%-------------------------------------------------------------------------------------------------#
"""变换字典"""
csmar_name_changer = {'Accper':'Time','Enddate':'Time','Symbol':'Stkcd','TradingDate':'Time'
                        ,'PostDate':'Time','Stockcode':'Stkcd'}

csmar_dtype_changer = {'Stkcd':str,'Symbol':str,'Stockcode':str,
                       'Accper':np.datetime64,'Enddate':np.datetime64,'TradingDate':np.datetime64,
                       'PostDate':np.datetime64}
next_year_changer = {2021:2020,2019:2018,2020:2019}
#%%-------------------------------------------------------------------------------------------------#
bbs_xlsx_list = ['SE_InvestorSentimentSta'+str(i)+'.xlsx' for i in range(7)]
bbs_xlsx_list_test = ['SE_InvestorSentimentSta'+str(i)+'.xlsx' for i in range(2)]
#%%-------------------------------------------------------------------------------------------------#

print('"Path.py" activated,done')