import sys
from alive_progress import alive_bar
import os
from ROOT import TFile # pyright: ignore # type: ignore
sys.path.append("./")
from utils import logger, get_centrality_bins

def get_sparse_dict(sparse_name, dmeson, beforeDMesonPR=False):

    print(f"Getting sparse dict for {sparse_name}")
    print(f"sparse_name: {sparse_name} vs CorrelMaps")
    if sparse_name == "CorrelMaps":
        return {
                'PoolBin': 0,
                'PtTrig': 1,
                'PtAssoc': 2,
                'DeltaEta': 3,
                'DeltaPhi': 4,
                'Mass': 5,
                'ScoreBkg': 6,
                'ScoreFD': 7
                }
    elif sparse_name == "CorrelTrig":
        return {
                'Mass': 0,
                'PtTrig': 1,
                'ScoreBkg': 2,
                'ScoreFD': 3
                }
    elif sparse_name == "FlowSP":
        return {
                'Mass': 0,
                'Pt': 1,
                'Cent': 2,
                'Sp': 3,
                'ScoreBkg': 4,
                'ScoreFD': 5,
                'Occ': 6
                }
    else:
        logger("Retrieving MC sparse ... ", level='INFO')
        if dmeson == 'Dzero':
            if sparse_name == "RecoPrompt" or sparse_name == "RecoFD" or sparse_name == "RecoRefl" or sparse_name == "RecoReflPrompt" or sparse_name == "RecoReflFD":
                return {
                    'ScoreBkg': 0,
                    'ScorePrompt': 1,
                    'ScoreFD': 2,
                    'Mass': 3,
                    'Pt': 4,
                    'Y': 5,
                    'CandType': 6,
                    'PtBMoth': 7,
                    'Origin': 8,
                    'NPvContr': 9,
                    'Cent': 10,
                    'Occ': 11,
                }
            elif sparse_name == "GenPrompt" or sparse_name == "GenFD":
                return {
                    'Pt': 0,
                    'PtBMoth': 1,
                    'Y': 2,
                    'Origin': 3,
                    'NPvContr': 4,
                    'Cent': 5,
                    'Occ': 6
                }
            else:
                logger(f"Unknown sparse type for Ds {sparse_name}", level='ERROR')
        elif dmeson == 'Dplus':
            if sparse_name == "RecoPrompt":
                return {
                    'Mass': 0,
                    'Pt': 1,
                    'ScoreBkg': 2,
                    'ScorePrompt': 3,
                    'ScoreFD': 4,
                    'Cent': 5,
                    'Occ': 6,
                }
            elif sparse_name == "RecoFD":
                return {
                    'Mass': 0 if not beforeDMesonPR else 0,
                    'Pt': 1 if not beforeDMesonPR else 1,
                    'ScoreBkg': 2 if not beforeDMesonPR else 4,
                    'ScorePrompt': 3 if not beforeDMesonPR else 5,
                    'ScoreFD': 4 if not beforeDMesonPR else 6,
                    'Cent': 5 if not beforeDMesonPR else 7,
                    'Occ': 6 if not beforeDMesonPR else 8,
                    'PtBMoth': 7 if not beforeDMesonPR else 2,
                    'FlagBHad': 8 if not beforeDMesonPR else 3,
                }
            elif sparse_name == "GenPrompt":
                return {
                    'Pt': 0,
                    'Y': 1,
                    'Cent': 2,
                    'Occ': 3
                }
            elif sparse_name == "GenFD":
                return {
                'Pt': 0,
                'Y': 1,
                'Cent': 2 if not beforeDMesonPR else 4,
                'Occ': 3 if not beforeDMesonPR else 5,
                'PtBMoth': 4 if not beforeDMesonPR else 2,
                'FlagBHad': 5 if not beforeDMesonPR else 3,
            }
            else:
                logger(f"Unknown sparse type for Ds {sparse_name}", level='ERROR')
        elif dmeson == 'Ds':
            if sparse_name == "RecoPrompt":
                return {
                    'Mass': 0,
                    'Pt': 1,
                    'Cent': 3,  # Check number 2
                    'NPvContr': 4,
                    'ScoreBkg': 5,
                    'ScorePrompt': 6,
                    'ScoreFD': 7,
                    'Occ': 8,
                }
            elif sparse_name == "RecoFD":
                return {
                    'Mass': 0,
                    'Pt': 1,
                    'Cent': 2,
                    'ScoreBkg': 3,
                    'ScorePrompt': 4,
                    'ScoreFD': 5,
                    'PtBMoth': 6,
                    'FlagBHad': 7,
                    'Occ': 8
                }
            elif sparse_name == "GenPrompt":
                return {
                    'Pt': 0,
                    'Y': 1,
                    'NPvContr': 2,
                    'Cent': 3,
                    'Occ': 4
                }
            elif sparse_name == "GenFD":
                return {
                    'Pt': 0,
                    'Y': 1,
                    'Cent': 2,
                    'PtBMoth': 3,
                    'FlagBHad': 4,
                    'Occ': 5
                }
            else:
                logger(f"Unknown sparse type for Ds {sparse_name}", level='ERROR')
        else:
            logger(f"Data type {data_type} not recognized", level='ERROR')

def get_pt_preprocessed_sparses(config, iPt):

    logger("Loading preprocessed sparses", level='INFO')
    sparses_data, sparses_reco, sparses_gen, axes, resolutions = {}, {}, {}, {}, {}
    pre_cfg = config['preprocess']

    # Find preprocess config of sparse with name "FlowSP" (this is the one to be projected)
    for sparse_cfg in pre_cfg['sparses_data']:
        if sparse_cfg['name'] == 'FlowSP':
            sparse_proj_cfg = sparse_cfg
            break
    print(f"\n\naxes: {axes}")
    ptmin = config["ptbins"][iPt]
    ptmax = config["ptbins"][iPt+1]

    if config.get("outdirPrep") and config["outdirPrep"] != "":
        infileprep = TFile(f"{config['outdirPrep']}/preprocess/{int(ptmin*10)}_{int(ptmax*10)}/AnalysisResults.root")
    else:
        infileprep = TFile(f"{config['outdir']}/preprocess/{int(ptmin*10)}_{int(ptmax*10)}/AnalysisResults.root")

    if config["operations"].get("proj_data"):
        inputs_dir = f"Data_FlowSP/hf-task-flow-charm-hadrons"
        sparse_data_name = f"Data_{sparse_proj_cfg['name']}/{sparse_proj_cfg['path']}"
        axes['Flow'] = {ax: iax for iax, ax in enumerate(sparse_proj_cfg['axes']['names'])}
        sparses_data[sparse_proj_cfg['name']] = infileprep.Get(sparse_data_name)
        resolutions[f'Reso_{sparse_proj_cfg["name"]}'] = infileprep.Get(f'{inputs_dir}/histo_reso_delta_cent')

    if config["operations"].get("proj_mc"):
        subdir = infileprep.Get("MC/Reco")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            sparses_reco[key.GetName()[1:]] = obj
            axes[key.GetName()[1:]] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_reco'].keys())}

        subdir = infileprep.Get("MC/Gen")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            sparses_gen[key.GetName()[1:]] = obj
            axes[key.GetName()[1:]] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_gen'].keys())}

    infileprep.Close()

    return sparses_data, sparses_reco, sparses_gen, axes, resolutions
