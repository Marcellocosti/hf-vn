import sys
from alive_progress import alive_bar
import os
from ROOT import TFile # pyright: ignore # type: ignore
sys.path.append("./")
from utils import logger, get_centrality_bins

def get_sparses_dicts(config, beforeDMesonPR = False):
    
    axes_dict = {}
    ### Data sparse dictionary
    axes_dict['Flow'] = {
        'Mass': 0,
        'Pt': 1,
        'Cent': 2,
        'Sp': 3,
        'ScoreBkg': 4,
        'ScoreFD': 5,
        'Occ': 6
    }
    ### MC sparse dictionary
    if config['Dmeson'] == 'Dzero':
        axes_dict['RecoPrompt'] = {
            'ScoreBkg': 0,
            'score_prompt': 1,
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
        axes_dict['RecoFD'] = axes_dict['RecoPrompt']
        axes_dict['RecoRefl'] = axes_dict['RecoPrompt']
        axes_dict['RecoReflPrompt'] = axes_dict['RecoPrompt']
        axes_dict['RecoReflFD'] = axes_dict['RecoPrompt']
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'PtBMoth': 1,
            'Y': 2,
            'Origin': 3,
            'NPvContr': 4,
            'Cent': 5,
            'Occ': 6
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'PtBMoth': 1,
            'Y': 2,
            'Origin': 3,
            'NPvContr': 4,
            'Cent': 5,
            'Occ': 6
        }
    elif config['Dmeson'] == 'Dplus':
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'ScoreBkg': 2,
            'score_prompt': 3,
            'ScoreFD': 4,
            'Cent': 5,
            'Occ': 6,
        }
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'ScoreBkg': 2,
            'score_prompt': 3,
            'ScoreFD': 4,
            'Cent': 5,
            'Occ': 6,
            'PtBMoth': 7,
            'FlagBHad': 8,
        }
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'Y': 1,
            'Cent': 2,
            'Occ': 3
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'Y': 1,
            'Cent': 2,
            'PtBMoth': 3,
            'FlagBHad': 4,
        }

    elif config['Dmeson'] == 'Ds':
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'Cent': 3,  # Check number 2
            'NPvContr': 4,
            'ScoreBkg': 5,
            'score_prompt': 6,
            'ScoreFD': 7,
            'Occ': 8,
        }
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'Cent': 2,
            'ScoreBkg': 3,
            'score_prompt': 4,
            'ScoreFD': 5,
            'PtBMoth': 6,
            'FlagBHad': 7,
            'Occ': 8
        }
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'Y': 1,
            'NPvContr': 2,
            'Cent': 3,
            'Occ': 4
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'Y': 1,
            'Cent': 2,
            'PtBMoth': 3,
            'FlagBHad': 4,
            'Occ': 5
        }

    return axes_dict

def get_pt_preprocessed_sparses(config, iPt):
    
    logger("Loading preprocessed sparses", level='INFO')
    sparsesFlow, sparsesReco, sparsesGen, axes_dict, resolutions = {}, {}, {}, {}, {}
    pre_cfg = config['preprocess']
    axes_dict['Flow'] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_data'].keys())}
    ptmin = config["ptbins"][iPt]
    ptmax = config["ptbins"][iPt+1]

    if config.get("outdirPrep") and config["outdirPrep"] != "":
        infileprep = TFile(f"{config['outdirPrep']}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root")
    else:
        infileprep = TFile(f"{config['outdir']}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root")

    if config["operations"].get("proj_data"):
        for key, _ in pre_cfg["data"].items():
            sparsesFlow[f'Flow_{key}'] = infileprep.Get(f'Data_Flow_{key}/hSparseFlowCharm')
            resolutions[f'Reso_Flow_{key}'] = infileprep.Get(f'Data_Flow_{key}/hResolution')

    if config["operations"].get("proj_mc"):
        subdir = infileprep.Get("MC/Reco")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            sparsesReco[key.GetName()[1:]] = obj
            axes_dict[key.GetName()[1:]] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_reco'].keys())}

        subdir = infileprep.Get("MC/Gen")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            sparsesGen[key.GetName()[1:]] = obj
            axes_dict[key.GetName()[1:]] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_gen'].keys())}

    infileprep.Close()

    return sparsesFlow, sparsesReco, sparsesGen, axes_dict, resolutions

def get_sparses(config, get_data=True, get_mc=True, debug=False, MCBeforePRDplus=False):
    """Load the sparses and axes infos

    Args:
        config (dict): the flow config dictionary
        get_data (bool, optional): load data sparses. Defaults to True.
        get_mc (bool, optional): load mc sparses. Defaults to True.
        debug (bool, optional): print debug info. Defaults to False.

    Outputs:
        sparsesFlow: thnSparse in the flow task
        sparsesReco: thnSparse of reco level from the D meson task
        sparsesGen: thnSparse of gen level from the D meson task
        axes_dict (dict): dictionary of the axes for each sparse
    """

    sparsesFlow, sparsesReco, sparsesGen, resolutions = {}, {}, {}, {}
    print(f"MCBeforePRDplus: {MCBeforePRDplus}")
    axes_dict = get_sparses_dicts(config, MCBeforePRDplus)
    pre_cfg = config['preprocess'] if config.get('preprocess') else config
    if get_data:
        logger(f"\t\t[Data] Loading data sparses from: {pre_cfg['data']}")
        infileflow = []
        resolutions = {}
        for name, dataset in pre_cfg['data'].items():
            # Collect all files starting with AnalysisResults_ and ending with .root in the dataset['files'] string
            if isinstance(dataset["files"], str) and not dataset["files"].endswith(".root"):
                list_of_files = [f for f in os.listdir(dataset["files"]) if f.endswith(".root")]
                infileflow = [TFile(os.path.join(dataset["files"], file)) for file in list_of_files]
            elif isinstance(dataset["files"], list):
                if len(dataset["files"]) == 1:
                    if dataset["files"][0].endswith(".root"):
                        infileflow = [TFile(dataset["files"][0])]
                    else:
                        list_of_files = [f for f in os.listdir(dataset["files"][0]) if f.endswith(".root")]
                        infileflow = [TFile(os.path.join(dataset["files"][0], file)) for file in list_of_files]
                elif len(dataset["files"]) > 1:
                    if all(file.endswith(".root") for file in dataset["files"]):
                        infileflow = [TFile(file) for file in dataset["files"]]
                    else:
                        logger("The dataset contains multiple files, but not all of them are root files. Provide a single root file or a list of root files or a directory containing all root files.", level='ERROR')
            else:
                infileflow = [TFile(dataset["files"])] if isinstance(dataset["files"], str) else [TFile(file) for file in dataset["files"]]
            sparsesFlow[f'Flow_{name}'] = []
            with alive_bar(len(infileflow), title=f"[INFO]\t\t[Data] Loading data sparses for {name}") as bar:
                for infile in infileflow:
                    sparsesFlow[f'Flow_{name}'].append(infile.Get('hf-task-flow-charm-hadrons/hSparseFlowCharm'))
                    bar()
            [infile.Close() for infile in infileflow]
            resofile = TFile.Open(dataset["resolution"], 'r')
            det_A = config.get('detA', 'FT0c')
            det_B = config.get('detB', 'FV0a')
            det_C = config.get('detC', 'TPCtot')
            _, (centMin, centMax) = get_centrality_bins(config["centrality"])
            print(f"cent_{centMin}_{centMax}/{det_A}_{det_B}_{det_C}/histo_reso_delta_cent")
            resolutions[f"Reso_Flow_{name}"] = resofile.Get(f'cent_{centMin}_{centMax}/{det_A}_{det_B}_{det_C}/histo_reso_delta_cent')
            resolutions[f"Reso_Flow_{name}"].SetDirectory(0)
            resofile.Close()

    if get_mc:
        print(f"Loading mc sparse from: {pre_cfg['mc']}")
        infiletask = [TFile(pre_cfg['mc'])] if isinstance(pre_cfg['mc'], str) else [TFile(pre_cfg['mc']) for file in pre_cfg['mc']]

        if config['Dmeson'] == 'Dzero':
            sparseD0Path = 'hf-task-d0/hBdtScoreVsMassVsPtVsPtBVsYVsOriginVsD0Type'
            sparsesReco['RecoPrompt'] = [file.Get(sparseD0Path) for file in infiletask]
            for ifile in range(len(sparsesReco['RecoPrompt'])):
                sparsesReco['RecoPrompt'][ifile].GetAxis(axes_dict['RecoPrompt']['Origin']).SetRange(2, 2)    # make sure it is prompt
                sparsesReco['RecoPrompt'][ifile].GetAxis(axes_dict['RecoPrompt']['CandType']).SetRange(1, 2) # make sure it is signal

            sparsesReco['RecoFD'] = [file.Get(sparseD0Path) for file in infiletask]
            for ifile in range(len(sparsesReco['RecoFD'])):
                sparsesReco['RecoFD'][ifile].GetAxis(axes_dict['RecoPrompt']['Origin']).SetRange(3, 3)       # make sure it is non-prompt
                sparsesReco['RecoFD'][ifile].GetAxis(axes_dict['RecoPrompt']['CandType']).SetRange(1, 2)    # make sure it is signal

            sparsesReco['RecoRefl'] = [file.Get(sparseD0Path) for file in infiletask]
            for ifile in range(len(sparsesReco['RecoRefl'])):
                sparsesReco['RecoRefl'][ifile].GetAxis(axes_dict['RecoPrompt']['CandType']).SetRange(3, 4)  # make sure it is reflection

            sparsesReco['RecoReflPrompt'] = [file.Get(sparseD0Path) for file in infiletask]
            for ifile in range(len(sparsesReco['RecoReflPrompt'])):
                sparsesReco['RecoReflPrompt'][ifile].GetAxis(axes_dict['RecoPrompt']['CandType']).SetRange(3, 4)  # make sure it is reflection
                sparsesReco['RecoReflPrompt'][ifile].GetAxis(axes_dict['RecoPrompt']['Origin']).SetRange(2, 2)       # make sure it is prompt   

            sparsesReco['RecoReflFD'] = [file.Get(sparseD0Path) for file in infiletask]
            for ifile in range(len(sparsesReco['RecoReflFD'])):
                sparsesReco['RecoReflFD'][ifile].GetAxis(axes_dict['RecoPrompt']['CandType']).SetRange(3, 4)    # make sure it is reflection
                sparsesReco['RecoReflFD'][ifile].GetAxis(axes_dict['RecoPrompt']['Origin']).SetRange(3, 3)       # make sure it is FD
            #TODO: safety checks for Dmeson reflecton and secondary peak

            sparsesGen['GenPrompt'] = [file.Get('hf-task-d0/hSparseAcc') for file in infiletask]
            for ifile in range(len(sparsesGen['GenPrompt'])):
                sparsesGen['GenPrompt'][ifile].GetAxis(axes_dict['GenPrompt']['Origin']).SetRange(2, 2)  # make sure it is prompt

            sparsesGen['GenFD'] = [file.Get('hf-task-d0/hSparseAcc') for file in infiletask]
            for ifile in range(len(sparsesGen['GenFD'])):
                sparsesGen['GenFD'][ifile].GetAxis(axes_dict['GenFD']['Origin']).SetRange(3, 3)  # make sure it is non-prompt
        elif config['Dmeson'] == 'Dplus':
            sparsesReco['RecoFD']     = [file.Get('hf-task-dplus/hSparseMassFD') for file in infiletask]
            sparsesReco['RecoPrompt'] = [file.Get('hf-task-dplus/hSparseMassPrompt') for file in infiletask]
            sparsesGen['GenPrompt']   = [file.Get('hf-task-dplus/hSparseMassGenPrompt') for file in infiletask]
            sparsesGen['GenFD']       = [file.Get('hf-task-dplus/hSparseMassGenFD') for file in infiletask]

        elif config['Dmeson'] == 'Ds':
            sparsesReco['RecoPrompt'] = [file.Get('hf-task-ds/MC/Ds/Prompt/hSparseMass') for file in infiletask]
            sparsesReco['RecoFD']     = [file.Get('hf-task-ds/MC/Ds/NonPrompt/hSparseMass') for file in infiletask]
            sparsesGen['GenPrompt']   = [file.Get('hf-task-ds/MC/Ds/Prompt/hSparseGen') for file in infiletask]
            sparsesGen['GenFD']       = [file.Get('hf-task-ds/MC/Ds/NonPrompt/hSparseGen') for file in infiletask]

        [infile.Close() for infile in infiletask]

    logger("Sparses loaded", level='INFO')
    if debug:
        print('\n')
        print('###############################################################')
        for key, value in axes_dict.items():
            logger(f"{key}:", level='DEBUG')
            for sub_key, sub_value in value.items():
                logger(f"    {sub_key}: {sub_value}", level='DEBUG')
        print('###############################################################')
        print('\n')
    return sparsesFlow, sparsesReco, sparsesGen, axes_dict, resolutions
