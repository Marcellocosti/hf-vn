'''
Script to project the MC distributions and apply the pt weights from the AnRes.root of Dtask
python3 proj_thn.py config_flow.yml --cutsetConfig config_cutset.yml [-c --correlated]
If the last argument is not provided, the script will project the combined cutsets.
'''
import ROOT
import uproot
import yaml
import argparse
import sys
import os
from ROOT import TFile, TObject
from alive_progress import alive_bar
from scipy.interpolate import make_interp_spline
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from sparse_dicts import get_pt_preprocessed_sparses
from utils import reweight_histo_1D, reweight_histo_2D, reweight_histo_3D, get_vn_versus_mass, profile_mass_sp, make_dir_root_file

ROOT.TH1.AddDirectory(False)

import yaml
from ROOT import TFile

def proj_multitrial(config, config_cutsets):
    # Load the original YAML config
    with open(config['OriginalConfig'], 'r') as ymlCfgFile:
        original_config = yaml.load(ymlCfgFile, yaml.FullLoader)

    print("Processing multitrial projections...")

    # Define the pt bin directory name
    pt_bin_label = f"pt_{int(config_cutsets['Pt']['min'][0] * 10)}_{int(config_cutsets['Pt']['max'][0] * 10)}"
    icut = config_cutsets['icutset']
    trial = config['iTrial']

    # Input and output file paths
    in_file_path = f"{original_config['outdir']}/cutvar_{original_config['suffix']}_combined/proj/proj_{icut:02d}.root"
    original_file = TFile.Open(in_file_path, "READ")
    hist_mass_sp = original_file.Get(f"{pt_bin_label}/hMassSpData")
    first_reso_key = next(iter(config['preprocess']['data']))            # Changes required when more than one dataset will be processed
    reso = original_file.Get(f"{pt_bin_label}/hResolution_Reso_Flow_{first_reso_key}").GetBinContent(1)

    # Open new ROOT file for writing
    out_file_path = f"{config['outdir']}/cutvar_{trial}_combined/proj/proj_{icut:02d}.root"
    output_file = TFile.Open(out_file_path, "RECREATE")
    output_file.mkdir(pt_bin_label)
    output_file.cd(pt_bin_label)

    # Project mass axis and write it
    hist_mass_sp.ProjectionX().Write("hMassData")
    # Compute vn vs mass profile histogram
    hist_vn_vs_mass = profile_mass_sp(hist_mass_sp, config['projections']['inv_mass_bins'][0], reso)
    hist_vn_vs_mass.SetName("hVnVsMassData")
    hist_vn_vs_mass.Write()

    print(f"Saved histograms to {out_file_path} under '{pt_bin_label}'")

    # Close files
    original_file.Close()
    output_file.Close()

def proj_data(sparses_dict, reso_dict, axes, inv_mass_bins, proj_scores, writeopt):

    proj_vars = ['Mass', 'sp', 'score_FD', 'score_bkg'] if proj_scores else ['Mass', 'sp']
    proj_axes = [axes['Flow'][var] for var in proj_vars]

    for reso_name, reso in reso_dict.items():
        reso.Write(f'hResolution_{reso_name}', writeopt)

    for var, ax in zip(proj_vars, proj_axes):
        for isparse, (_, sparse) in enumerate(sparses_dict.items()):
            hist_var_temp = sparse.Projection(ax)
            hist_var_temp.SetName(f'h{var.capitalize()}_{isparse}')
            if isparse == 0:
                hist_var = hist_var_temp.Clone(f'h{var.capitalize()}')
                hist_var.Reset()

            hist_var.Add(hist_var_temp)

        hist_var.Write(f'h{var.capitalize()}Data', writeopt)

    hist_vn_sp = get_vn_versus_mass(sparses_dict, reso_dict, inv_mass_bins, axes['Flow']['Mass'], axes['Flow']['sp'])
    hist_vn_sp.Write('hVnVsMassData', writeopt)

    # Save a TH2 of (Mass, Sp) for the multitrial systematic
    mass_lowest_bin, mass_highest_bin, sp_lowest_bin, sp_highest_bin = -1, 1e10, -1, 1e10
    for isparse, (_, sparse) in enumerate(sparses_dict.items()):
        hist_mass_sp_temp = sparse.Projection(axes['Flow']['sp'], axes['Flow']['Mass'])
        hist_mass_sp_temp.SetName(f'hMassSp_{isparse}')
        if isparse == 0:
            hist_mass_sp = hist_mass_sp_temp.Clone(f'hMassSp')
            hist_mass_sp.Reset()
            sp_lowest_bin = hist_mass_sp_temp.ProjectionY().FindFirstBinAbove(0)
            sp_highest_bin = hist_mass_sp_temp.ProjectionY().FindLastBinAbove(0)
            mass_lowest_bin = hist_mass_sp_temp.ProjectionX().FindFirstBinAbove(0)
            mass_highest_bin = hist_mass_sp_temp.ProjectionX().FindLastBinAbove(0)
        else:
            if hist_mass_sp_temp.ProjectionY().FindFirstBinAbove(0) < sp_lowest_bin:
                sp_lowest_bin = hist_mass_sp_temp.ProjectionY().FindFirstBinAbove(0)
            if hist_mass_sp_temp.ProjectionY().FindLastBinAbove(0) > sp_highest_bin:
                sp_highest_bin = hist_mass_sp_temp.ProjectionY().FindLastBinAbove(0)
            if hist_mass_sp_temp.ProjectionX().FindFirstBinAbove(0) < mass_lowest_bin:
                mass_lowest_bin = hist_mass_sp_temp.ProjectionX().FindFirstBinAbove(0)
            if hist_mass_sp_temp.ProjectionX().FindLastBinAbove(0) > mass_highest_bin:
                mass_highest_bin = hist_mass_sp_temp.ProjectionX().FindLastBinAbove(0)

        hist_mass_sp.Add(hist_mass_sp_temp)

    hist_mass_sp.GetXaxis().SetRange(mass_lowest_bin, mass_highest_bin)
    hist_mass_sp.GetYaxis().SetRange(sp_lowest_bin, sp_highest_bin)
    hist_mass_sp.Write(f'hMassSpData', writeopt)

def proj_mc_reco(sparsesReco, sPtWeightsD, sPtWeightsB, Bspeciesweights, writeopt):

    for key, sparse in sparsesReco.items():
        if key != 'RecoPrompt' and key != 'RecoFD':
            sparse.Projection(axes[key]['Mass']).Write(f'h{key}Mass')
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    hMassPrompt = sparsesReco['RecoPrompt'].Projection(axes['RecoPrompt']['Mass'])
    hMassPrompt.SetName(f'hPromptMass_{ptMin}_{ptMax}')
    hMassFD = sparsesReco['RecoFD'].Projection(axes['RecoFD']['Mass'])
    hMassFD.SetName(f'hFDMass_{ptMin}_{ptMax}')

    ### project pt prompt
    hPtPrompt = sparsesReco['RecoPrompt'].Projection(axes['RecoPrompt']['Pt'])
    if sPtWeightsD:
        hPtPrompt = reweight_histo_1D(hPtPrompt, sPtWeightsD, binned=False)

    ### project pt FD
    if sPtWeightsD:
        hPtFD = reweight_histo_1D(sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt']), sPtWeightsD, binned=False)
    elif sPtWeightsB:
        if Bspeciesweights:
            hPtFD = reweight_histo_3D(
                sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt'], axes['RecoFD']['pt_bmoth'], axes['RecoFD']['flag_bhad']), 
                sPtWeightsB, Bspeciesweights
            )
        else:
            hPtFD = reweight_histo_2D(
                sparsesReco['RecoFD'].Projection(axes['RecoFD']['pt_bmoth'], axes['RecoFD']['Pt']),          # 2D projection: Projection(ydim, xdim)
                sPtWeightsB, binned=False
            )
    elif Bspeciesweights:
        hPtFD = reweight_histo_2D(
            sparsesReco['RecoFD'].Projection(axes['RecoFD']['flag_bhad'], axes['RecoFD']['Pt']),             # 2D projection: Projection(ydim, xdim)
            Bspeciesweights, binned=True
        )
    else:
        hPtFD = sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt'])

    ## write the output 
    hMassPrompt.Write('hPromptMass', writeopt)
    hMassFD.Write('hFDMass', writeopt)
    hPtPrompt.Write('hPromptPt', writeopt)
    hPtFD.Write('hFDPt', writeopt)

def proj_mc_gen(sparsesGen, sPtWeightsD, sPtWeightsB, Bspeciesweights, writeopt):

    for key, sparse in sparsesGen.items():
        if key != 'GenPrompt' and key != 'GenFD':
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    ### prompt
    hGenPtPrompt = sparsesGen['GenPrompt'].Projection(axes['GenPrompt']['Pt'])
    if sPtWeightsD:
        hGenPtPrompt = reweight_histo_1D(hGenPtPrompt, sPtWeightsD, binned=False)

    ### FD
    if sPtWeightsD:
        hGenPtFD = reweight_histo_1D(sparsesGen['GenFD'].Projection(axes['GenFD']['Pt']), sPtWeightsD, binned=False)
    elif sPtWeightsB:
        if Bspeciesweights:
            hGenPtFD = reweight_histo_3D(
                sparsesGen['GenFD'].Projection(axes['GenFD']['Pt'], axes['GenFD']['pt_bmoth'], axes['GenFD']['flag_bhad']),
                sPtWeightsB, Bspeciesweights
            )
        else:
            hGenPtFD = reweight_histo_2D(
                sparsesGen['GenFD'].Projection(axes['GenFD']['pt_bmoth'], axes['GenFD']['Pt']),         # 2D projection: Projection(ydim, xdim)
                sPtWeightsB, binned=False
            )
    elif Bspeciesweights:
        hGenPtFD = reweight_histo_2D(
            sparsesGen['GenFD'].Projection(axes['GenFD']['flag_bhad'], axes['GenFD']['Pt']),            # 2D projection: Projection(ydim, xdim)
            Bspeciesweights, binned=True
        )
    else:
        hGenPtFD = sparsesGen['GenFD'].Projection(axes['GenFD']['Pt'])

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
        print('\033[91m WARNING: pt weights for D mesons will not be provided! \033[0m')
        print('\033[91m WARNING: pt weights for B mesons will not be provided! \033[0m')
        print('\033[91m WARNING: B species weights will not be provided! \033[0m')
        return None, None, None
        
    ptWeightsFile = TFile.Open(cfgProj["PtWeightsFile"], 'r')

    if cfgProj.get('ApplyPtWeightsD'):
        hPtWeightsD = ptWeightsFile.Get('hPtWeightsFONLLtimesTAMUDcent')
        ptBinCentersD = [ (hPtWeightsD.GetBinLowEdge(i)+hPtWeightsD.GetBinLowEdge(i+1))/2 for i in range(1, hPtWeightsD.GetNbinsX()+1)]
        ptBinContentsD = [hPtWeightsD.GetBinContent(i) for i in range(1, hPtWeightsD.GetNbinsX()+1)]
        print(f'ptBinCentersD: {ptBinCentersD}\n\n')
        print(f'ptBinContentsD: {ptBinContentsD}\n\n')
        sPtWeights = make_interp_spline(ptBinCentersD, ptBinContentsD)
    else:
        print('\033[91m WARNING: pt weights for D mesons will not be provided! \033[0m')
        sPtWeights = None

    if cfgProj.get('ApplyPtWeightsB'):
        hPtWeightsB = ptWeightsFile.Get('hPtWeightsFONLLtimesTAMUBcent')
        ptBinCentersB = [ (hPtWeightsB.GetBinLowEdge(i)+hPtWeightsB.GetBinLowEdge(i+1))/2 for i in range(1, hPtWeightsB.GetNbinsX()+1)]
        ptBinContentsB = [hPtWeightsB.GetBinContent(i) for i in range(1, hPtWeightsB.GetNbinsX()+1)]
        sPtWeightsB = make_interp_spline(ptBinCentersB, ptBinContentsB)
    else:
        print('\033[91m WARNING: pt weights for B mesons will not be provided! \033[0m')
        sPtWeightsB = None

    if cfgProj.get('ApplyBSpeciesWeights'):
        Bspeciesweights = config['Bspeciesweights']
    else:
        print('\033[91m WARNING: B species weights will not be provided! \033[0m')
        Bspeciesweights = None
    
    return sPtWeights, sPtWeightsB, Bspeciesweights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    parser.add_argument('--cutsetConfig', "-cc", metavar='text', type=str, nargs='?',
                        const=None, default='cutsetConfig.yaml',
                        help='Optional cutset configuration file (default: cutsetConfig.yaml)')
    parser.add_argument("--correlated", "-c", action="store_true",
                        help="Produce projection files for correlated cuts")
    parser.add_argument("--multitrial", "-mult", action="store_true",
                        help="Produce projection files for multitrial systematics")
    args = parser.parse_args()

    with open(args.config, 'r') as ymlCfgFile:
        config = yaml.load(ymlCfgFile, yaml.FullLoader)
    operations = config["operations"]

    with open(args.cutsetConfig, 'r') as ymlCutSetFile:
        cutSetCfg = yaml.load(ymlCutSetFile, yaml.FullLoader)
        iCut = f"{int(cutSetCfg['icutset']):02d}"
    cutVars = cutSetCfg['cutvars']

    if args.multitrial:
        print(f"\n\nRunning multitrial projections for cutset {iCut}!")
        proj_multitrial(config, cutSetCfg)
        sys.exit(0)

    method = "correlated" if args.correlated else "combined"
    outDir = config['outdir'] + f'/cutvar_{config["suffix"]}_{method}/proj/'
    previousProjFiles = [f for f in os.listdir(outDir) if f.endswith('.root')]
    if operations["proj_data"] and operations["proj_mc"]:
        print(f"\n\nCreating new file and project data and mc!")
        outfile = TFile(f"{outDir}/proj_{iCut}.root", 'RECREATE')
    elif len(previousProjFiles) == 0:
        print(f"\n\nNo existing previous projections, creating new file and project data ({operations["proj_data"]}) or mc ({operations["proj_mc"]})!")
        outfile = TFile(f"{outDir}/proj_{iCut}.root", 'RECREATE')
    else:
        print(f"\n\nFound previous projections, updating existing file!")
        outfile = TFile.Open(f"{outDir}/proj_{iCut}.root", 'UPDATE')

    write_opt_data = TObject.kOverwrite if operations["proj_data"] else 0 
    write_opt_mc = TObject.kOverwrite if operations["proj_mc"] else 0 

    # compute info for pt weights
    sPtWeightsD, sPtWeightsB, Bspeciesweights = get_pt_weights(config["projections"]) if config['projections'].get('PtWeightsFile') else (None, None, None)

    with alive_bar(len(cutSetCfg['Pt']['min']), title='Processing pT bins') as bar:
        for iPt, (ptMin, ptMax, bkg_min, bkg_max, fd_min, fd_max) in enumerate(zip(cutSetCfg['Pt']['min'], cutSetCfg['Pt']['max'],
                                                                                   cutVars['score_bkg']['min'], cutVars['score_bkg']['max'],
                                                                                   cutVars['score_FD']['min'], cutVars['score_FD']['max'])):

            # Cut on centrality and pt on data applied in the preprocessing
            print(f'Projecting distributions for {ptMin:.1f} < pT < {ptMax:.1f} GeV/c')
            sparsesFlow, sparsesReco, sparsesGen, axes, resolutions = get_pt_preprocessed_sparses(config, iPt)

            make_dir_root_file(f'pt_{int(ptMin*10)}_{int(ptMax*10)}', outfile)
            outfile.cd(f'pt_{int(ptMin*10)}_{int(ptMax*10)}')
            if operations["proj_data"]:
                for key, sparse in sparsesFlow.items():
                    sparse.GetAxis(axes['Flow']['score_FD']).SetRangeUser(fd_min, fd_max)
                    sparse.GetAxis(axes['Flow']['score_bkg']).SetRangeUser(bkg_min, bkg_max)
                proj_data(sparsesFlow, resolutions, axes, config["projections"]['inv_mass_bins'][iPt], config["projections"].get('storeML'), write_opt_data)
                print(f"Projected data!")

            if operations["proj_mc"]:
                for key, iSparse in sparsesReco.items():
                    iSparse.GetAxis(axes[key]['score_bkg']).SetRangeUser(bkg_min, bkg_max)
                    iSparse.GetAxis(axes[key]['score_FD']).SetRangeUser(fd_min, fd_max)

                proj_mc_reco(sparsesReco, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc)
                print("Projected mc reco!")
                proj_mc_gen(sparsesGen, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc)
                print("Projected mc gen!")

            print('\n\n')
            bar()

    outfile.Close()
