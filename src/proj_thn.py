'''
Script to project the MC distributions and apply the pt weights from the AnRes.root of Dtask
python3 proj_thn.py config_flow.yml --cutsetConfig config_cutset.yml [-c --correlated]
If the last argument is not provided, the script will project the combined cutsets.
'''
import ROOT
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import yaml
import argparse
import sys
import os
import glob
from pathlib import Path
from functools import partial
from ROOT import TFile, TObject
from alive_progress import alive_bar
from scipy.interpolate import make_interp_spline
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from sparse_dicts import get_pt_preprocessed_sparses
from utils import reweight_histo_1D, reweight_histo_2D, reweight_histo_3D, get_vn_versus_mass, profile_mass_sp, make_dir_root_file, logger

ROOT.TH1.AddDirectory(False)

import yaml
from ROOT import TFile

def proj_multitrial(config, multitrial_folder):

    pt_bin_label = Path(multitrial_folder).name
    logger(f"Processing multitrial projections for pt bin {pt_bin_label} ...", level='INFO')

    # Load default cutsets
    default_cutsets = [f"{config['outdir']}/cutvar_{config['suffix']}_combined/cutsets/{f}" for f in os.listdir(f"{config['outdir']}/cutvar_{config['suffix']}_combined/cutsets") if f.endswith('.yml')]
    # Load Mass and MassSp histos from the default cases
    default_histos = {}
    for default_cutset in default_cutsets:
        suffix = os.path.basename(default_cutset).replace(".yml", "").replace("cutset_", "")
        default_proj = TFile.Open(default_cutset.replace(".yml", ".root").replace("cutset", "proj"), "READ")
        default_histos[suffix] = {}
        default_histos[suffix]['Mass'] = default_proj.Get(f"{pt_bin_label}/hMassData")
        default_histos[suffix]['MassSp'] = default_proj.Get(f"{pt_bin_label}/hMassSpData")
        default_proj.Close()

    def process_cutset(multitrial_dir, default_histos):
        trial_number = Path(multitrial_dir).name.replace("trial_", "")
        try:
            with open(f"{multitrial_dir}/config_trial_{trial_number}.yml", 'r') as ymlCutSetFile:
                config_trial = yaml.safe_load(ymlCutSetFile)
        except Exception as e:
            logger(f"Error opening or reading config file for trial {trial_number}: {e}", level='ERROR')
            return
        
        multitrial_cutsets = glob.glob(f"{multitrial_dir}/cutsets/*.yml")
        for multitrial_cutset in multitrial_cutsets:
            suffix = os.path.basename(multitrial_cutset).replace(".yml", "").replace("cutset_", "")
            output_dir = os.path.dirname(multitrial_cutset).replace('cutset', 'proj')
            os.makedirs(output_dir, exist_ok=True)
            output_path = multitrial_cutset.replace('.yml', '.root').replace('cutset', 'proj')
            output_file = TFile.Open(output_path, "RECREATE")
            output_file.mkdir(pt_bin_label)
            output_file.cd(pt_bin_label)
            default_histos[suffix]['Mass'].Write("hMassData")
            hist_vn_vs_mass = profile_mass_sp(default_histos[suffix]['MassSp'], config_trial['projections']['inv_mass_bins'][0], 0.746)
            hist_vn_vs_mass.Write("hVnVsMassData")
            output_file.Close()

    # Parallel execution
    multitrial_dirs = [f for f in glob.glob(f"{multitrial_folder}/trial_*/")]
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(partial(process_cutset, default_histos=default_histos), multitrial_dirs)

def proj_data(i_bin, sparse, axes, resolution, proj_cfg, writeopt):

    proj_vars = ['Mass', 'Sp', 'ScoreFD', 'ScoreBkg'] if proj_cfg.get('storeML') else ['Mass', 'Sp']
    proj_axes = [axes['FlowSP'][var] for var in proj_vars]

    for var, ax in zip(proj_vars, proj_axes):
        hist_var = sparse.Projection(ax)
        hist_var.Write(f'h{var.capitalize()}Data', writeopt)

    # for var, ax in zip(proj_vars, proj_axes):
    #     for isparse, (_, sparse) in enumerate(sparses_dict.items()):
    #         hist_var_temp = sparse.Projection(ax)
    #         hist_var_temp.SetName(f'h{var.capitalize()}_{isparse}')
    #         if isparse == 0:
    #             hist_var = hist_var_temp.Clone(f'h{var.capitalize()}')
    #             hist_var.Reset()

    #         hist_var.Add(hist_var_temp)

    #     hist_var.Write(f'h{var.capitalize()}Data', writeopt)

    hist_vn_sp = get_vn_versus_mass(sparse, proj_cfg['inv_mass_bins'][i_bin], axes['FlowSP']['Mass'], axes['FlowSP']['Sp'])
    hist_vn_sp.Scale(1/resolution) # Correct for resolution
    hist_vn_sp.Write('hVnVsMassData', writeopt)

    # Save a TH2 of (Mass, Sp) for the multitrial systematic
    mass_lowest_bin, mass_highest_bin, sp_lowest_bin, sp_highest_bin = -1, 1e10, -1, 1e10
    # for isparse, (_, sparse) in enumerate(sparses_dict.items()):
    hist_mass_sp = sparse.Projection(axes['FlowSP']['Sp'], axes['FlowSP']['Mass'])
    sp_lowest_bin = hist_mass_sp.ProjectionY().FindFirstBinAbove(0)
    sp_highest_bin = hist_mass_sp.ProjectionY().FindLastBinAbove(0)
    mass_lowest_bin = hist_mass_sp.ProjectionX().FindFirstBinAbove(0)
    mass_highest_bin = hist_mass_sp.ProjectionX().FindLastBinAbove(0)
    hist_mass_sp.GetXaxis().SetRange(mass_lowest_bin, mass_highest_bin)
    hist_mass_sp.GetYaxis().SetRange(sp_lowest_bin, sp_highest_bin)
    hist_mass_sp.Write(f'hMassSpData', writeopt)

def proj_mc_reco(sparses_reco, sPtWeightsD, sPtWeightsB, Bspeciesweights, writeopt):

    for key, sparse in sparses_reco.items():
        if key != 'RecoPrompt' and key != 'RecoFD':
            sparse.Projection(axes[key]['Mass']).Write(f'h{key}Mass')
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    hMassPrompt = sparses_reco['RecoPrompt'].Projection(axes['RecoPrompt']['Mass'])
    hMassPrompt.SetName(f'hPromptMass_{ptMin}_{ptMax}')
    hMassFD = sparses_reco['RecoFD'].Projection(axes['RecoFD']['Mass'])
    hMassFD.SetName(f'hFDMass_{ptMin}_{ptMax}')

    ### project pt prompt
    hPtPrompt = sparses_reco['RecoPrompt'].Projection(axes['RecoPrompt']['Pt'])
    if sPtWeightsD:
        hPtPrompt = reweight_histo_1D(hPtPrompt, sPtWeightsD, binned=False)

    ### project pt FD
    if sPtWeightsD:
        hPtFD = reweight_histo_1D(sparses_reco['RecoFD'].Projection(axes['RecoFD']['Pt']), sPtWeightsD, binned=False)
    elif sPtWeightsB:
        if Bspeciesweights:
            hPtFD = reweight_histo_3D(
                sparses_reco['RecoFD'].Projection(axes['RecoFD']['Pt'], axes['RecoFD']['PtBMoth'], axes['RecoFD']['FlagBHad']), 
                sPtWeightsB, Bspeciesweights
            )
        else:
            hPtFD = reweight_histo_2D(
                sparses_reco['RecoFD'].Projection(axes['RecoFD']['PtBMoth'], axes['RecoFD']['Pt']),          # 2D projection: Projection(ydim, xdim)
                sPtWeightsB, binned=False
            )
    elif Bspeciesweights:
        hPtFD = reweight_histo_2D(
            sparses_reco['RecoFD'].Projection(axes['RecoFD']['FlagBHad'], axes['RecoFD']['Pt']),             # 2D projection: Projection(ydim, xdim)
            Bspeciesweights, binned=True
        )
    else:
        hPtFD = sparses_reco['RecoFD'].Projection(axes['RecoFD']['Pt'])

    ## write the output 
    hMassPrompt.Write('hPromptMass', writeopt)
    hMassFD.Write('hFDMass', writeopt)
    hPtPrompt.Write('hPromptPt', writeopt)
    hPtFD.Write('hFDPt', writeopt)

def proj_mc_gen(sparses_gen, sPtWeightsD, sPtWeightsB, Bspeciesweights, writeopt):

    for key, sparse in sparses_gen.items():
        if key != 'GenPrompt' and key != 'GenFD':
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    ### prompt
    hGenPtPrompt = sparses_gen['GenPrompt'].Projection(axes['GenPrompt']['Pt'])
    if sPtWeightsD:
        hGenPtPrompt = reweight_histo_1D(hGenPtPrompt, sPtWeightsD, binned=False)

    ### FD
    if sPtWeightsD:
        hGenPtFD = reweight_histo_1D(sparses_gen['GenFD'].Projection(axes['GenFD']['Pt']), sPtWeightsD, binned=False)
    elif sPtWeightsB:
        if Bspeciesweights:
            hGenPtFD = reweight_histo_3D(
                sparses_gen['GenFD'].Projection(axes['GenFD']['Pt'], axes['GenFD']['PtBMoth'], axes['GenFD']['FlagBHad']),
                sPtWeightsB, Bspeciesweights
            )
        else:
            hGenPtFD = reweight_histo_2D(
                sparses_gen['GenFD'].Projection(axes['GenFD']['PtBMoth'], axes['GenFD']['Pt']),         # 2D projection: Projection(ydim, xdim)
                sPtWeightsB, binned=False
            )
    elif Bspeciesweights:
        hGenPtFD = reweight_histo_2D(
            sparses_gen['GenFD'].Projection(axes['GenFD']['FlagBHad'], axes['GenFD']['Pt']),            # 2D projection: Projection(ydim, xdim)
            Bspeciesweights, binned=True
        )
    else:
        hGenPtFD = sparses_gen['GenFD'].Projection(axes['GenFD']['Pt'])

    ## write the output
    hGenPtPrompt.Write('hPromptGenPt', writeopt)
    hGenPtFD.Write('hFDGenPt', writeopt)

def get_pt_weights(cfgProj):
    """Get pt weights and return weights flags with spline

    Args:
        cfgProj (dict): Configuration dictionary for projections

    Outputs:
        sPtWeights (spline): Spline for ptWeights interpolation
        sPtWeightsB (spline): Spline for ptWeightsB weights interpolation
        Bspeciesweights (str): B species weights # TODO
    """

    # REVIEW: the ptWeights inputed is a list, but the ptWeights outputed is a TH1D object
    # and actually ptweights is used as a flag
        # compute info for pt weights
    if not cfgProj.get('PtWeightsFile'):
        logger('No pt weights for D and B mesons provided in the config file!', level='WARNING')
        return None, None, None
        
    ptWeightsFile = TFile.Open(cfgProj["PtWeightsFile"], 'r')

    if cfgProj.get('ApplyPtWeightsD'):
        hPtWeightsD = ptWeightsFile.Get('hPtWeightsFONLLtimesTAMUDcent')
        ptBinCentersD = [ (hPtWeightsD.GetBinLowEdge(i)+hPtWeightsD.GetBinLowEdge(i+1))/2 for i in range(1, hPtWeightsD.GetNbinsX()+1)]
        ptBinContentsD = [hPtWeightsD.GetBinContent(i) for i in range(1, hPtWeightsD.GetNbinsX()+1)]
        sPtWeights = make_interp_spline(ptBinCentersD, ptBinContentsD)
    else:
        logger('pt weights for D mesons will not be provided!', level='WARNING')
        sPtWeights = None

    if cfgProj.get('ApplyPtWeightsB'):
        hPtWeightsB = ptWeightsFile.Get('hPtWeightsFONLLtimesTAMUBcent')
        ptBinCentersB = [ (hPtWeightsB.GetBinLowEdge(i)+hPtWeightsB.GetBinLowEdge(i+1))/2 for i in range(1, hPtWeightsB.GetNbinsX()+1)]
        ptBinContentsB = [hPtWeightsB.GetBinContent(i) for i in range(1, hPtWeightsB.GetNbinsX()+1)]
        sPtWeightsB = make_interp_spline(ptBinCentersB, ptBinContentsB)
    else:
        logger('pt weights for B mesons will not be provided!', level='WARNING')
        sPtWeightsB = None

    if cfgProj.get('ApplyBSpeciesWeights'):
        Bspeciesweights = config['Bspeciesweights']
    else:
        logger('B species weights will not be provided!', level='WARNING')
        Bspeciesweights = None
    
    return sPtWeights, sPtWeightsB, Bspeciesweights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    parser.add_argument('--cutsetConfig', "-cc", metavar='text', type=str, nargs='?',
                        const=None, default='cutsetConfig.yaml',
                        help='Optional cutset configuration file (default: cutsetConfig.yaml)')
    parser.add_argument("--multitrial_folder", "-multfolder", metavar="text",
                        default="", help="Produce projection files for multitrial systematics")
    parser.add_argument("--outputDir", "-o", metavar="text",
                        default="", help="output directory, used only for directly running the script")
    args = parser.parse_args()

    with open(args.config, 'r') as ymlCfgFile:
        config = yaml.load(ymlCfgFile, yaml.FullLoader)
    operations = config["operations"]

    if args.multitrial_folder != "":
        logger(f"Running multitrial projections with config: {args.config}", level='INFO')
        proj_multitrial(config, args.multitrial_folder)
        sys.exit(0)

    with open(args.cutsetConfig, 'r') as ymlCutSetFile:
        cutSetCfg = yaml.load(ymlCutSetFile, yaml.FullLoader)
        iCut = f"{int(cutSetCfg['icutset']):02d}"

    outDir = os.path.join(os.path.dirname(os.path.dirname(args.cutsetConfig)), 'projs') if args.outputDir == "" else args.outputDir
    outfilePath = os.path.join(outDir, f"proj_{iCut}.root")
    os.makedirs(outDir, exist_ok=True)

    if operations["proj_data"] or operations["proj_mc"]:
        if os.path.exists(outfilePath):
            logger(f"Found previous projection file {outfilePath}, will update it", level='INFO')
            outfile = TFile.Open(outfilePath, 'UPDATE')
        else:
            logger(f"No previous projection file found, will create a new one at {outfilePath}", level='INFO')
            outfile = TFile(outfilePath, 'RECREATE')
    else:
        sys.exit(0)

    write_opt_data = TObject.kOverwrite if operations.get("proj_data") else 0 
    write_opt_mc = TObject.kOverwrite if operations.get("proj_mc") else 0 

    # compute info for pt weights
    if operations.get("proj_mc"):
        sPtWeightsD, sPtWeightsB, Bspeciesweights = get_pt_weights(config["projections"]) if config['projections'].get('PtWeightsFile') else (None, None, None)

    if operations.get("proj_data"):
        reso_file = TFile.Open(config["projections"]["Resolution"], 'r')
        print(f"Opened resolution file {config["projections"]['Resolution']}")
        det_A = config["projections"].get('detA', 'FT0c')
        det_B = config["projections"].get('detB', 'FV0a')
        det_C = config["projections"].get('detC', 'TPCtot')
        print(f"Getting resolution histogram for {det_A}_{det_B}_{det_C}")
        reso_hist = reso_file.Get(f'{det_A}_{det_B}_{det_C}/histo_reso_delta_cent')
        reso_hist.SetDirectory(0)
        reso_file.Close()
        outfile.cd()
        reso_hist.Write("hResolution")
        resolution = reso_hist.GetBinContent(1)
        print(f"Wrote resolution histogram for {det_A}_{det_B}_{det_C}")

    with alive_bar(len(cutSetCfg['Pt']['min']), title='Processing pT bins') as bar:
        for iPt, (ptMin, ptMax, bkg_min, bkg_max, fd_min, fd_max) in enumerate(zip(cutSetCfg['Pt']['min'], cutSetCfg['Pt']['max'],
                                                                                   cutSetCfg['ScoreBkg']['min'], cutSetCfg['ScoreBkg']['max'],
                                                                                   cutSetCfg['ScoreFD']['min'], cutSetCfg['ScoreFD']['max'])):

            # Cut on centrality and pt on data applied in the preprocessing
            print(f'Projecting distributions for {ptMin:.1f} < pT < {ptMax:.1f} GeV/c')
            sparse_flow, sparses_reco, sparses_gen, axes = get_pt_preprocessed_sparses(config, iPt)
            make_dir_root_file(f'pt_{int(ptMin*10)}_{int(ptMax*10)}', outfile)
            outfile.cd(f'pt_{int(ptMin*10)}_{int(ptMax*10)}')
            if operations["proj_data"]:
                sparse_flow["FlowSP"].GetAxis(axes['FlowSP']['ScoreFD']).SetRangeUser(fd_min, fd_max)
                sparse_flow["FlowSP"].GetAxis(axes['FlowSP']['ScoreBkg']).SetRangeUser(bkg_min, bkg_max)
                proj_data(iPt, sparse_flow["FlowSP"], axes, resolution, config["projections"], write_opt_data)
                print(f"Projected data!")

            if operations.get("proj_mc"):
                for key, i_sparse in sparses_reco.items():
                    i_sparse.GetAxis(axes[key]['ScoreBkg']).SetRangeUser(bkg_min, bkg_max)
                    i_sparse.GetAxis(axes[key]['ScoreFD']).SetRangeUser(fd_min, fd_max)

                proj_mc_reco(sparses_reco, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc)
                print("Projected mc reco!")
                proj_mc_gen(sparses_gen, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc)
                print("Projected mc gen!")

            print('\n\n')
            bar()

    outfile.Close()
