import yaml
import ROOT
import copy
import os
import numpy as np
import argparse
import itertools
import copy
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from utils import check_dir

def get_reference_config_pt_bin(config_flow, iPtBin):
    # Deepcopy to ensure that we are working with a copy and not modifying the original config_flow
    pt_bin_config = copy.deepcopy(config_flow)

    print(f"pt_bin_config['preprocess']['axes_data']: {pt_bin_config['preprocess']['axes_data']}")
    axes_dict = {}
    axes_dict['Flow'] = {ax: iax for iax, (ax, _) in enumerate(pt_bin_config['preprocess']['axes_data'].items())}
    print(f"axes_dict['Flow']: {axes_dict['Flow']}\n")

    # Separate bin-specific keys (those with lists as values) from common settings
    for key, value in pt_bin_config.items():
        if isinstance(value, list):
            if key == 'ptbins':
                pt_bin_config[key] = [value[iPtBin], value[iPtBin+1]]
            else:
                pt_bin_config[key] = value[iPtBin]

    # Modify cut_variation values but leave the original config_flow intact
    if pt_bin_config.get("cut_variation"):
        pt_bin_config["cut_variation"]["corr_bdt_cut"]["bkg_max"] = [pt_bin_config["cut_variation"]["corr_bdt_cut"]["bkg_max"][iPtBin]]
        pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["max"] = [pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["max"][iPtBin]]
        pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["min"] = [pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["min"][iPtBin]]
        pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["step"] = [pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["step"][iPtBin]]
        pt_bin_config["cut_variation"]["uncorr_bdt_cut"]["bkg_max"] = [pt_bin_config["cut_variation"]["uncorr_bdt_cut"]["bkg_max"][iPtBin]]
        pt_bin_config["cut_variation"]["uncorr_bdt_cut"]["sig"] = [pt_bin_config["cut_variation"]["uncorr_bdt_cut"]["sig"][iPtBin]]
    
    return pt_bin_config

def modify_yaml_bdt(config_flow, config_mod, output_dir):
    
    with open(config_flow, 'r') as CfgFlow:
        cfg_flow = yaml.safe_load(CfgFlow)
    with open(config_mod, 'r') as CfgMod:
        cfg_mod = yaml.safe_load(CfgMod)

    check_dir(f"{output_dir}_multitrial/")
    os.makedirs(f"{output_dir}_multitrial/", exist_ok=True)
    os.makedirs(f"{output_dir}_multitrial/config_history/", exist_ok=True)

    with open(os.path.join(f"{output_dir}_multitrial/config_history/", f"config_reference.yml"), 'w') as out_file:
        yaml.dump(cfg_flow, out_file, default_flow_style=False)
    with open(os.path.join(f"{output_dir}_multitrial/config_history/", f"config_modifications.yml"), 'w') as out_file:
        yaml.dump(cfg_mod, out_file, default_flow_style=False)

    for _, pt_bin in enumerate(cfg_mod['ptbins']):
        
        ptmin = pt_bin['range'][0]
        ptmax = pt_bin['range'][1]
        pt_bin_index = cfg_flow['ptbins'].index(ptmin)
        ref_config_ptbin = get_reference_config_pt_bin(cfg_flow, pt_bin_index)
        
        os.makedirs(f"{output_dir}_multitrial/pt_{int(ptmin*10)}_{int(ptmax*10)}/config_sys/", exist_ok=True)
        output_file = os.path.join(f"{output_dir}_multitrial/pt_{int(ptmin*10)}_{int(ptmax*10)}/config_sys/", f"config_reference.yml")
        with open(output_file, 'w') as out_file:
            yaml.dump(ref_config_ptbin, out_file, default_flow_style=False)

        varied_configs = {}
        print(f"ptbin: {pt_bin}")
        for var in pt_bin["multitrial"]:
            print(f"var: {var}")
            if var in ref_config_ptbin:
                print("Already in reference config")
                varied_configs[var] = list(set(ref_config_ptbin[var] + pt_bin["multitrial"][var]))
            else:
                print("Not in reference config")
                varied_configs[var] = pt_bin["multitrial"][var]

        # Perform itertools.product to get all combinations
        keys = list(varied_configs.keys())
        values = list(varied_configs.values())
        print(f"\nvalues: {values}\n")
        combinations = list(itertools.product(*values))
        config_variants = [dict(zip(keys, combination)) for combination in combinations] 

        for config_variant in config_variants:
            print(f"\nconfig_variant: {config_variant}\n")

        # Print results
        for idx, variant in enumerate(config_variants):
            cfg_variant = copy.deepcopy(ref_config_ptbin)
            cfg_variant["iTrial"] = idx
            cfg_variant["OriginalConfig"] = config_flow

            # Set directory to preprocessed files
            cfg_variant["outdirprep"] = ref_config_ptbin['outdir']
            
            # Set which operations to perform
            cfg_variant["operations"]["preprocess_data"] = False
            cfg_variant["operations"]["preprocess_mc"] = False
            cfg_variant["operations"]["make_yaml"] = True
            cfg_variant["operations"]["proj_data"] = False
            cfg_variant["operations"]["proj_mc"] = False
            cfg_variant["operations"]["proj_multitrial"] = True
            
            cfg_variant["outdir"] = f"{output_dir}_multitrial/pt_{int(ptmin*10)}_{int(ptmax*10)}/trials/"
            cfg_variant["suffix"] = f"{idx}"
            cfg_variant["nworkers"] = 1 # parallelization is in the bash script
            cfg_variant["MaxChi2"] = pt_bin['MaxChi2']
            cfg_variant["MinSignificance"] = pt_bin['MinSignificance']
            cfg_variant["MaxSignificance"] = pt_bin['MaxSignificance']
            for varied_var in variant:
                if ref_config_ptbin.get(varied_var):
                    if isinstance(ref_config_ptbin[varied_var], list):
                        cfg_variant[varied_var] = [variant[varied_var]]
                    else:
                        cfg_variant[varied_var] = variant[varied_var]

            cfg_variant["MassFitRanges"] = [[variant["MassMin"], variant["MassMax"]]]
            cfg_variant["projections"]["inv_mass_bins"] = [np.arange(variant['MassMin'], variant['MassMax'] + variant['inv_mass_bins_steps'], variant['inv_mass_bins_steps']).tolist()]
            if variant["MassMin"] > 1.75 or variant["MassMax"] < 1.95:
                # else the region used for the prefit of the bkg 
                # function would be empty, leading to a crash
                cfg_variant["NSigma4SB"] = [2]


            output_file = os.path.join(f"{output_dir}_multitrial/pt_{int(ptmin*10)}_{int(ptmax*10)}/config_sys/", f"config_var_{idx}.yml")
            with open(output_file, 'w') as out_file:
                yaml.dump(cfg_variant, out_file, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_config', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('--modifications_config', "-m", metavar='text', default='')
    parser.add_argument("--outputdir", "-o", metavar="text", default=".", help="output directory")
    args = parser.parse_args()

    modify_yaml_bdt(args.input_config,
                    args.modifications_config,
                    args.outputdir)