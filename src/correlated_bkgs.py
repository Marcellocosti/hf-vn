'''
    Normalizations are given for the fit function to use a signal PDF and
    template from hMassTotalCorrBkgs multiplied by a common normalization 
    constant.
'''

import pandas as pd
import matplotlib.pyplot as plt
import uproot
import numpy as np
import ROOT
from array import array
import os
import sys
import argparse
import yaml
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, get_centrality_bins, make_dir_root_file
from corr_bkgs_brs import final_states_dplus, final_states_ds, final_states_dstar_to_d0_piplus, final_states_dstar_to_dplus_pi0, final_states_lc, final_states_xic

def produce_corr_bkgs_templs(config_flow, cutset_config, correlatedCutsets):

    with open(config_flow, 'r') as f:
        config_flow = yaml.safe_load(f)
    cfg_corrbkgs = config_flow["corr_bkgs"]

    with open(cutset_config, 'r') as f:
        cfg_cutset = yaml.safe_load(f)

    full_dfs = []
    tables = [[] for table in cfg_corrbkgs["table_names"]]
    with uproot.open(cfg_corrbkgs["input_file"]) as f:
        for table_name, table_list in zip(cfg_corrbkgs["table_names"], tables):
            for iKey, key in enumerate(f.keys()):
                if table_name in key:
                    dfData = f[key].arrays(library='pd')
                    table_list.append(dfData)

            full_table_df = pd.concat([df for df in table_list], ignore_index=True)
            full_dfs.append(full_table_df)
    full_df = pd.concat(full_dfs, axis=1)

    ### Centrality selection
    _, (centMin, centMax) = get_centrality_bins(config_flow["centrality"])
    full_df = full_df.query(f"fCentrality >= {centMin} and fCentrality < {centMax}")

    decays_info = {
        "Dplus": {
            "decay_table": final_states_dplus,
            "mc_abundance": cfg_corrbkgs.get('correct_dplus_abundance', 1)
        },
        "Ds": {
            "decay_table": final_states_ds,
            "mc_abundance": cfg_corrbkgs.get('correct_ds_abundance', 1)
        },
        "DstarD0": {
            "decay_table": final_states_dstar_to_d0_piplus,
            "mc_abundance": cfg_corrbkgs.get('correct_dstar_abundance', 1)
        },
        "DstarDplus": {
            "decay_table": final_states_dstar_to_dplus_pi0,
            "mc_abundance": cfg_corrbkgs.get('correct_dstar_abundance', 1)
        },
        "Lc": {
            "decay_table": final_states_lc,
            "mc_abundance": cfg_corrbkgs.get('correct_Lc_abundance', 1)
        },
        "Xic": {
            "decay_table": final_states_xic,
            "mc_abundance": cfg_corrbkgs.get('correct_Xic_abundance', 1)
        }
    }

    ### Extract the total MC branching ratio for all species
    total_br_mc = {}
    for particle, info_dict in decays_info.items():
        total_br_mc_part = 0
        for fin_state, fin_state_info in info_dict["decay_table"].items():
            resonant_states = fin_state_info["ResoStates"]
            for reso_state in resonant_states:
                total_br_mc_part += reso_state['br_mc']
        total_br_mc[particle] = total_br_mc_part

    # Process corr bkgs channels
    final_states_to_include = cfg_corrbkgs["include_final_states"]
    sgn_fin_state = cfg_corrbkgs['sgn_fin_state']
    outfile = ROOT.TFile(cutset_config.replace("cutset", "corrbkg").replace(".yml", ".root"), "RECREATE")
    for ipt_bin, (pt_min, pt_max, score_bkg_max, score_fd_min, score_fd_max) in enumerate(zip(cfg_cutset["Pt"]["min"],
                                                                                              cfg_cutset["Pt"]["max"],
                                                                                              cfg_cutset["score_bkg"]["max"],
                                                                                              cfg_cutset["ScoreFD"]["min"],
                                                                                              cfg_cutset["score_FD"]["max"])):
        pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        histo_weights_dict = {}
        print(f"Processing pt bin: {pt_min} - {pt_max}")
        mass_min = config_flow["simfit"]["MassFitRanges"][ipt_bin][0]
        mass_max = config_flow["simfit"]["MassFitRanges"][ipt_bin][1]
        query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fM >= {mass_min} and fM < {mass_max}"
        # query_str = f"fPt >= {pt_min} and fPt < {pt_max} and {config_flow['bkg_score_column']} < {score_bkg_max} and {config_flow['fd_score_column']} >= {score_fd_min} and {config_flow['fd_score_column']} < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
        cutset_sel_df = full_df.query(query_str)

        for particle, info_dict in decays_info.items():
            for fin_state, fin_state_info in info_dict["decay_table"].items():

                if not fin_state.startswith(f"{sgn_fin_state}_") and not any(fin_state in name for name in final_states_to_include):
                    continue

                hMassChannel = ROOT.TH1F(f"hMass{fin_state}", f"hMass{fin_state}", 600, 1.6, 2.2)
                for reso_state in fin_state_info["ResoStates"]:
                    selected_df = cutset_sel_df.query(f"abs(fFlagMcMatchRec) == {fin_state_info['FlagFinal']} and fFlagMcDecayChanRec == {reso_state['FlagReso']}")
                    
                    if len(selected_df) > 0:
                        make_dir_root_file(f"{pt_key}/{fin_state}/{reso_state['Channel']}", outfile)
                        outfile.cd(f"{pt_key}/{fin_state}/{reso_state['Channel']}")

                        # Fill tree from DataFrame
                        hMass = ROOT.TH1F("hMass", "hMass", 600, 1.6, 2.2)
                        tree = ROOT.TTree("DecayTree", f"DecayTree {particle} {reso_state['Channel']}")
                        mass = array("f", [0.])
                        tree.Branch("fM", mass, "fM/F")

                        mass_values = selected_df["fM"].to_numpy(dtype="float32")
                        for val in mass_values:
                            mass[0] = val
                            tree.Fill()

                        tree.Draw("fM >> hMass", "", "goff")
                        hMass.Smooth(100)
                        hMass.Write()
                        hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
                        hBRs.GetXaxis().SetBinLabel(1, "MC")
                        br_mc = info_dict["mc_abundance"] * (reso_state['br_mc'] / total_br_mc[particle])
                        hBRs.SetBinContent(1, br_mc)
                        hBRs.GetXaxis().SetBinLabel(2, "PDG")
                        br_pdg = reso_state['br_pdg']
                        hBRs.SetBinContent(2, br_pdg)
                        hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
                        raw_yield = tree.GetEntries()
                        hBRs.SetBinContent(3, raw_yield)
                        hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
                        hBRs.SetBinContent(4, raw_yield * (br_pdg/br_mc))
                        hBRs.Write()
                        histo_weights_dict[f"{fin_state}_{reso_state['Channel']}"] = [raw_yield * (br_pdg/br_mc), hMass]

        n_final_states = len(histo_weights_dict)

        hMassTotalSignal = ROOT.TH1F("hMassTotalSignal", "hMassTotalSignal", 600, 1.6, 2.2)
        hMassTotalCorrBkgs = ROOT.TH1F("hMassTotalCorrBkgs", "hMassTotalCorrBkgs", 600, 1.6, 2.2)
        hWeightsAnchorSignal = ROOT.TH1F("hWeightsAnchorSignal", "hWeightsAnchorSignal", n_final_states+1, 0, n_final_states+1)
        hWeightsAnchorToFirst = ROOT.TH1F("hWeightsAnchorToFirst", "hWeightsAnchorToFirst", n_final_states+1, 0, n_final_states+1)
        total_signal_weight = 0
        i_final_state = 1
        for name, (weight, histo) in histo_weights_dict.items():
            if name.startswith(f"{sgn_fin_state}_"):
                hMassTotalSignal.Add(histo, weight)
                total_signal_weight += weight
            else:
                if i_final_state == 1:
                    weight_first_template = weight
                hMassTotalCorrBkgs.Add(histo, weight)
                hWeightsAnchorSignal.GetXaxis().SetBinLabel(i_final_state, name)
                hWeightsAnchorSignal.SetBinContent(i_final_state, weight)
                hWeightsAnchorToFirst.GetXaxis().SetBinLabel(i_final_state, name)
                hWeightsAnchorToFirst.SetBinContent(i_final_state, weight)
                i_final_state += 1

        # Normalize weights histogram to the total signal weight
        hWeightsAnchorSignal.Scale(1 / total_signal_weight)
        hWeightsAnchorToFirst.Scale(1 / weight_first_template)

        outfile.cd(pt_key)
        hMassTotalSignal.Write()
        hMassTotalCorrBkgs.Write()
        hWeightsAnchorSignal.Write()
        hWeightsAnchorToFirst.Write()

    outfile.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("flow_config", metavar="text",
                        default="config_flow.yaml", help="flow configuration file")
    parser.add_argument("cutset_config", metavar="text",
                        default="cfg_cutset.yaml", help="flow configuration file")
    parser.add_argument('--correlated', '-corr', action='store_true',
                        help="perform correlated analysis")
    args = parser.parse_args()

    produce_corr_bkgs_templs(
        args.flow_config,
        args.cutset_config,
        args.correlated
    )