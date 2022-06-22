# -*- coding: utf-8 -*-
from pcode.local_training.base_worker import BaseWorker
from pcode.local_training.normal_worker import NormalWorker
from pcode.local_training.fedrep_worker import FedRepWorker
from pcode.local_training.ditto_worker import DittoWorker
from pcode.local_training.fine_tuning_worker import FineTuningWorker
from pcode.local_training.fedrod_worker import FedRodWorker
from pcode.local_training.memo_worker import MemoWorker
from pcode.local_training.memo_lpft_worker import MemoLPFTWorker
from pcode.local_training.THE_worker import THEWorker
from pcode.local_training.THE_batch_wise_worker import THEBatchWorker
from pcode.local_training.THE_single_worker import THESingleWorker
from pcode.local_training.t3a_worker import T3aWorker
from pcode.local_training.lp_ft_worker import LPFTWorker
from pcode.local_training.apfl_worker import APFLWorker
from pcode.local_training.ttt_worker import TTTWorker
from pcode.local_training.tsa_worker import TsaWorker


def get_worker_class(conf):
    if not conf.is_personalized:
        return BaseWorker(conf)
    else:
        if conf.personalization_scheme["method"] == "Normal":
            return NormalWorker(conf)
        elif conf.personalization_scheme["method"] == "GMA":
            return NormalWorker(conf)
        elif conf.personalization_scheme["method"] == "FedRep":
            return FedRepWorker(conf)
        elif conf.personalization_scheme["method"] == "Ditto":
            return DittoWorker(conf)
        elif conf.personalization_scheme["method"] == "Fine_tune":
            return FineTuningWorker(conf)
        elif conf.personalization_scheme["method"] == "FedRod":
            return FedRodWorker(conf)
        elif conf.personalization_scheme["method"] == "Memo_global":
            return MemoWorker(conf, is_personal=False)
        elif conf.personalization_scheme["method"] == "Memo_personal":
            return MemoWorker(conf, is_personal=True)
        elif conf.personalization_scheme["method"] == "Memo_personal_lpft":
            return MemoLPFTWorker(conf, is_personal=True)
        elif conf.personalization_scheme["method"] == "THE":
            return THEWorker(conf, is_fine_tune=False)
        elif conf.personalization_scheme["method"] == "THE_FT":
            return THEWorker(conf, is_fine_tune=True)
        elif conf.personalization_scheme["method"] == "THE_single":
            return THESingleWorker(conf, is_fine_tune=False)
        elif conf.personalization_scheme["method"] == "THE_FT_single":
            return THESingleWorker(conf, is_fine_tune=True)
        elif conf.personalization_scheme["method"] == "THE_batch_wise_ema":
            return THEBatchWorker(conf, is_fine_tune=False, is_ema=True)
        elif conf.personalization_scheme["method"] == "THE_batch_wise":
            return THEBatchWorker(conf, is_fine_tune=False, is_ema=False)
        elif conf.personalization_scheme["method"] == "THE_FT_batch_wise_ema":
            return THEBatchWorker(conf, is_fine_tune=True, is_ema=True)
        elif conf.personalization_scheme["method"] == "THE_FT_batch_wise_ema":
            return THEBatchWorker(conf, is_fine_tune=True, is_ema=False)
        elif conf.personalization_scheme["method"] == "T3A":
            return T3aWorker(conf)
        elif conf.personalization_scheme["method"] == "lp_ft":
            return LPFTWorker(conf)
        elif conf.personalization_scheme["method"] == "apfl":
            return APFLWorker(conf)
        elif conf.personalization_scheme["method"] == "ttt":
            return TTTWorker(conf)
        elif conf.personalization_scheme["method"] == "tsa":
            return TsaWorker(conf)
        else:
            raise NotImplementedError("invalid personalization_scheme")
