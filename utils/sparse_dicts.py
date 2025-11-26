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
    sparses_data, sparses_reco, sparses_gen, axes = {}, {}, {}, {}
    pre_cfg = config['preprocess']

    # Find preprocess config of sparse with name "FlowSP" (this is the one to be projected)
    for input_cfg in pre_cfg['inputs']:
        for sparse_cfg in input_cfg['sparses']:
            if sparse_cfg['name'] == 'FlowSP':
                sparse_proj_cfg = sparse_cfg
                break

    print(f"\n\naxes: {axes}")
    ptmin = config["ptbins"][iPt]
    ptmax = config["ptbins"][iPt+1]
    pt_str = f"{int(ptmin*10)}_{int(ptmax*10)}"
    prep_dir = config.get("outdirPrep", config["outdir"])
    if config["operations"].get("proj_data"):
        infile_prep_data = TFile.Open(f"{prep_dir}/preprocess/{pt_str}/FlowSP/AnalysisResults_pt_{pt_str}.root", "read")
        print(f"Reading data sparse from: {prep_dir}/preprocess/{pt_str}/FlowSP/AnalysisResults_pt_{pt_str}.root")
        axes['FlowSP'] = {ax: iax for iax, ax in enumerate(sparse_proj_cfg['axes']['names'])}
        sparses_data["FlowSP"] = infile_prep_data.Get("FlowSP/hSparseFlowSP")
        infile_prep_data.Close()

    if config["operations"].get("proj_mc"):
        infile_prep_mc = TFile.Open(f"{prep_dir}/preprocess/{pt_str}/MC/AnalysisResults_pt_{pt_str}.root", "read")
        # subdir = infile_prep_mc.Get("MC")
        # print(f"MC subdir: {subdir}")
        for key in infile_prep_mc.GetListOfKeys():
            key_name = key.GetName()
            print(f"Key in MC subdir: {key_name}")
            if "Reco" in key_name:
                sparses_reco[key_name] = infile_prep_mc.Get(f"{key_name}/hSparse{key_name}")
            elif "Gen" in key_name:
                sparses_gen[key_name] = infile_prep_mc.Get(f"{key_name}/hSparse{key_name}")
            else:
                logger(f"Unknown sparse type in MC folder: {key_name}", level='ERROR')
            # Retrieve axes
            for input_cfg in pre_cfg['inputs']:
                for sparse_cfg in input_cfg['sparses']:
                    if sparse_cfg['name'] == key_name:
                        sparse_proj_cfg = sparse_cfg
                        break
            axes[key_name] = {ax: iax for iax, ax in enumerate(sparse_proj_cfg['axes']['names'])}
        infile_prep_mc.Close()

    return sparses_data, sparses_reco, sparses_gen, axes
