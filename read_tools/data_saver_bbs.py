from .data_dealer_bbs import *

def AT_EXINDAT_thr_weighted(BBS_thresh=0.05, BBSlist=bbs_xlsx_list):
    try:
        """1:east-weighted-EXINDAT"""
        bbs2_1 = BBS(BBS=bbs_path, BBSlist=BBSlist, Csmar_name='csv_merger.csv',BBS_thresh=BBS_thresh)
        bbs2_1.BBS_AT()
        bbs2_1.BBS_EX_INDAT_weighted()
        bbs2_1.Csmar_AT_IND_aggr_nocsmar(AT_like_df=bbs2_1.AT,ATIND_like_df=bbs2_1.EX_INDAT_read,
                                 AT_like_df_name='AT',add_Attr=True)
        save_csv(bbs2_1.AggrDF_nocsmar, PATH=main_data_path, name='AT_EXINDAT_weighted_thr'+str(BBS_thresh)+'.csv')
        del bbs2_1
        print('AT_EXINDAT_weighted_thr'+'THR:'+str(BBS_thresh)+' done')
    except:
        print('AT_EXINDAT_weighted_thr' + 'THR:' + str(BBS_thresh) + 'FALSE, something wrong')

def AT_EXINDAT_thr(BBS_thresh=0.05, BBSlist=bbs_xlsx_list):
    """基于east-AT-EX_INDAT-csmar"""
    try:
        bbs1 = BBS(BBS=bbs_path, BBSlist=BBSlist,dtype=csmar_dtype_changer,BBS_thresh=BBS_thresh)
        bbs1.BBS_AT()
        bbs1.BBS_EX_INDAT()
        bbs1.Csmar_AT_IND_aggr_nocsmar(AT_like_df=bbs1.AT,ATIND_like_df=bbs1.EX_INDAT,AT_like_df_name='AT',add_Attr=True)
        save_csv(bbs1.AggrDF_nocsmar,PATH=main_data_path,name='AT_EXINDAT_thr'+str(BBS_thresh)+'.csv')
        del bbs1
        print('AT_EXINDAT_thr'+'THR:'+str(BBS_thresh)+' done')
    except:
        print('AT_EXINDAT_thr'+'THR:'+str(BBS_thresh)+ 'FALSE, something wrong')