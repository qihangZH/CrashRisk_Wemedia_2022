from read_tools.data_saver_bbs import *
#%%-------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    print('"bbs_dealer_main.py" activating as Entrance File...')
    AT_EXINDAT_thr(BBS_thresh=0.2)
    AT_EXINDAT_thr(BBS_thresh=0.1)
    AT_EXINDAT_thr(BBS_thresh=0.05)
    AT_EXINDAT_thr(BBS_thresh=0.025)
    AT_EXINDAT_thr(BBS_thresh=0.01)

    AT_EXINDAT_thr_weighted(BBS_thresh=0.2)
    AT_EXINDAT_thr_weighted(BBS_thresh=0.1)
    AT_EXINDAT_thr_weighted(BBS_thresh=0.05)
    AT_EXINDAT_thr_weighted(BBS_thresh=0.025)
    AT_EXINDAT_thr_weighted(BBS_thresh=0.01)
    print('"bbs_dealer_main.py" activated as Entrance File')