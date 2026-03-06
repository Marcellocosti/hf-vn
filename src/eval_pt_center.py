import argparse
import sys
import os
import re
import array
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position
import pandas as pd
import numpy as np
import seaborn as sns
import awkward as ak
import matplotlib.pyplot as plt
import itertools
sys.path.append("./flareflyfitter/")
from raw_yield_fitter import RawYieldFitter
import yaml
import ROOT
from ROOT import TFile, TH1F
import awkward as ak
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from utils import logger, get_centrality_bins
from load_utils import load_aod_file
from matplotlib import gridspec
import uproot
from multiprocessing import Pool, cpu_count
ROOT.gROOT.SetBatch(True)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.config.threading.set_inter_op_parallelism_threads(20)

def eval_pt_center(cfg_file_name, minimizer, workers=1):
    # Read the configuration file
    with open(cfg_file_name, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    # Retrieve cutsets configs
    try:
        cutsets_dir = os.path.join(cfg['outdir'], f"vn_extr_{cfg['suffix']}_combined/cutsets")
        cutset_files = [os.path.join(cutsets_dir, f) for f in os.listdir(cutsets_dir) if f.endswith('.yml')]
        out_dir_type = "combined"
    except Exception as e:
        logger(f"Could not find combined cutsets, trying correlated cutsets ... ", level="WARNING")
        cutsets_dir = os.path.join(cfg['outdir'], f"vn_extr_{cfg['suffix']}_correlated/cutsets")
        cutset_files = [os.path.join(cutsets_dir, f) for f in os.listdir(cutsets_dir) if f.endswith('.yml')]
        out_dir_type = "correlated"
    cutset_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

    infer_vars = ['fPt', 'fScalarProd', 'fCent']
    infer_vars_labels = ['fPt (GeV/c)', 'fScalarProd', 'fCent (%)']

    # Loop over cutset configs
    s_weights = {}
    cfg_fit = cfg["v2extraction"]

    hist_summary_avg_pt = TH1F("h_summary_avg_pt", "h_summary_avg_pt", len(cfg["ptbins"])-1, array.array('d', cfg["ptbins"]))
    hist_summary_pt_shifts = TH1F("h_summary_pt_shifts", "h_summary_pt_shifts", len(cfg["ptbins"])-1, array.array('d', cfg["ptbins"]))
    for i_pt_bin, (pt_min, pt_max) in enumerate(zip(cfg["ptbins"][:-1], cfg["ptbins"][1:])):
        logger(f"Pt bin {i_pt_bin}: {pt_min} - {pt_max}", level="INFO")
        # Load input
        pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        has_sp_cent = True
        prep_dir = f"{cfg['ptCenter'].get('PrepDir', cfg['outdir'])}/preprocess/{pt_str}/TreesPtCenterSp"
        if not os.path.exists(prep_dir):
            prep_dir = f"{cfg['outdir']}/preprocess/{pt_str}/TreesPtCenter"
            has_sp_cent = False
            logger(f"Using tree without SP and centrality!", level="WARNING")
        downsample_frac = cfg['ptCenter']['downsample_fracs'][i_pt_bin] if cfg['ptCenter'].get('downsample_fracs') else 1.0
        df = load_aod_file(f"{prep_dir}/AO2D_{pt_str}.root", has_sp_cent, downsample_frac=downsample_frac)

        out_dir_pt = f"{cfg['outdir']}/vn_extr_{cfg['suffix']}_{out_dir_type}/ptcenter_{minimizer}/{pt_str}"
        if downsample_frac < 1.0:
            out_dir_pt += f"_downsampled_{downsample_frac}"
        os.makedirs(out_dir_pt, exist_ok=True)
        os.makedirs(out_dir_pt + "/fits", exist_ok=True)
        os.makedirs(out_dir_pt + "/vars", exist_ok=True)

        out_file = TFile.Open(f"{out_dir_pt}/pt_center.root", "recreate")
        histos_avgs = {}
        for var in infer_vars:
            histos_avgs[f"h_{var}_sgn"] = ROOT.TH1F(f"h_{var}_sgn", f"h_{var}_sgn", len(cutset_files), array.array('d', [-0.5] + [i+0.5 for i in range(len(cutset_files))]))
            histos_avgs[f"h_{var}_bkg"] = ROOT.TH1F(f"h_{var}_bkg", f"h_{var}_bkg", len(cutset_files), array.array('d', [-0.5] + [i+0.5 for i in range(len(cutset_files))]))

        sgn_funcs = {} # More info for signal functions, a dictionary is better
        sgn_funcs[cfg_fit['SgnFuncLabel']] = {
            'func': cfg_fit['SgnFunc'][i_pt_bin] if isinstance(cfg_fit['SgnFunc'], list) else cfg_fit['SgnFunc'],
            'part': cfg['Dmeson']
        }

        histos_avgs[f"h_ry_{cfg_fit['SgnFuncLabel']}"] = ROOT.TH1F(f"h_ry_{cfg_fit['SgnFuncLabel']}",
                                                                   f"h_ry_{cfg_fit['SgnFuncLabel']}",
                                                                   len(cutset_files),
                                                                   array.array('d', [-0.5] + [i+0.5 for i in range(len(cutset_files))]))
        print(f"Adding signal function: {sgn_funcs[cfg_fit['SgnFuncLabel']]}, {cfg_fit['SgnFuncLabel']} ... ")
        if cfg_fit.get('InclSecPeak'):
            print("Including secondary peak signal function ... ")
            include_sec_peak = cfg_fit['InclSecPeak'][i_pt_bin] if isinstance(cfg_fit['InclSecPeak'], list) else cfg_fit['InclSecPeak']
            print(f"include_sec_peak = {include_sec_peak}")
            if include_sec_peak:
                sgn_func_sec_peak = cfg_fit['SgnFuncSecPeak'][i_pt_bin] if isinstance(cfg_fit['SgnFuncSecPeak'], list) else cfg_fit['SgnFuncSecPeak']
                print(f"Adding secondary peak signal function: {sgn_func_sec_peak} ... ")
                sgn_funcs[cfg_fit['SgnFuncSecPeakLabel']] = {
                    'func': sgn_func_sec_peak,
                    'part': 'Dplus' if cfg['Dmeson'] == 'Ds' else 'Dstar',
                }
                histos_avgs[f"h_ry_{cfg_fit['SgnFuncSecPeakLabel']}"] = ROOT.TH1F(f"h_ry_{cfg_fit['SgnFuncSecPeakLabel']}",
                                                                                  f"h_ry_{cfg_fit['SgnFuncSecPeakLabel']}",
                                                                                  len(cutset_files),
                                                                                  array.array('d', [-0.5] + [i+0.5 for i in range(len(cutset_files))]))

        # Initialize fitter
        fitter = RawYieldFitter(cfg['Dmeson'], pt_min, pt_max, pt_str, minimizer, verbose=True)
        fitter.set_fit_range(cfg_fit['MassFitRanges'][i_pt_bin][0], cfg_fit['MassFitRanges'][i_pt_bin][1])

        for i_cutset, cutset_file in enumerate(cutset_files):
            with open(cutset_file, 'r') as cs_file:
                cutset_cfg = yaml.safe_load(cs_file)
            logger(f"Processing cutset file {cutset_file} ... ", level="INFO")
            cutset_suffix = os.path.basename(cutset_file).replace('.yml', '').split('_')[-1]

            # Setup fitter
            fitter.add_bkg_func(cfg_fit['BkgFunc'][i_pt_bin] if isinstance(cfg_fit['BkgFunc'], list) else cfg_fit['BkgFunc'], "Comb. bkg")
            for i_sgn, (label, sgn_func) in enumerate(sgn_funcs.items()):
                print(f"Adding signal function: {sgn_func}, {label} ... ")
                fitter.add_sgn_func(sgn_func['func'], label, sgn_func['part'])

            score_bkg_min = cutset_cfg["ScoreBkg"]["min"][i_pt_bin]
            score_bkg_max = cutset_cfg["ScoreBkg"]["max"][i_pt_bin]
            score_fd_min = cutset_cfg["ScoreFD"]["min"][i_pt_bin]
            score_fd_max = cutset_cfg["ScoreFD"]["max"][i_pt_bin]

            # Query the dataframe
            mass_min, mass_max = cfg_fit["MassFitRanges"][i_pt_bin]
            sel_string = f"fMlScore0 >= {score_bkg_min} and fMlScore0 < {score_bkg_max} and " \
                         f"fMlScore1 >= {score_fd_min} and fMlScore1 < {score_fd_max} and " \
                         f"fM >= {mass_min} and fM <= {mass_max}"
            print(f"Selection string: {sel_string}")
            sel_df = df.query(sel_string).reset_index(drop=True)
            print(f"Total entries: {len(df)}, selected entries: {len(sel_df)}")
            fitter.set_name(f"{pt_str}_{cutset_suffix}")
            fitter.set_data_to_fit_df(sel_df)
            if cfg_fit.get('Rebin'):
                fitter.set_rebin(cfg_fit['Rebin'][i_pt_bin]) if isinstance(cfg_fit['Rebin'], list) else fitter.set_rebin(cfg_fit['Rebin'])

            # Add correlated background if specified
            if cfg.get('corr_bkgs'):
                fitter.add_corr_bkgs(cfg['corr_bkgs'], f"{cfg['outdir']}/corrbkgs/", sel_string.replace(' and ', ' && '), pt_min, pt_max)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            sel_df['fM'].hist(bins=100, alpha=0.5, range=(mass_min, mass_max))
            ax.set_xlabel('fM')
            ax.set_ylabel('Counts')
            fig.savefig(f"{out_dir_pt}/fits/fM_raw_{cutset_suffix}.pdf", dpi=300, bbox_inches="tight")

            fitter.setup()

            # Prefit the MC prompt enhanced cut to fix the tails, binned fit
            if cfg_fit.get('FixSgnFromMC'):
                fitter.set_fix_sgn_to_mc_prefit(True)
                if i_cutset == 0:
                    fitter.prefit_mc(f"{cfg['outdir']}/corr_bkgs/templs_{pt_str}.root")
                    fitter.plot_mc_prefit(False, True, loc=["lower left", "upper left"],
                                          path=f"{out_dir_pt}/", out_file=out_file)
                    fitter.plot_raw_residuals_mc_prefit(path=f"{out_dir_pt}/fM_mc_prefit_residuals_{cutset_suffix}.pdf")

            cfg_pars_dict = None
            pt_center = 0.5 * (pt_min + pt_max)
            for setting in cfg_fit.get("InitFitPars", []):
                # check if this setting applies to the current pT bin
                if not any(low < pt_center < high for low, high in setting["pt_ranges"]):
                    continue

                cfg_pars_dict = {'init_pars_sgn': [], 'fix_pars_sgn': []}
                init_cfgs, fix_cfgs = [], []
                for par_dict in setting["pars"]:
                    for par_name, par_cfg in par_dict.items():

                        value = par_cfg[0]
                        if fix_pars_from_config[i_pt_bin]:
                            pmin, pmax = par_cfg[1], par_cfg[2]
                            fix_cfgs.append((0, par_name, value, [value, value]))
                        else:
                            pmin = value * 0.9
                            pmax = value * 1.1
                            init_cfgs.append((0, par_name, value, [pmin, pmax]))

                        if not isMultitrial:
                            print(
                                f"Setting initial fit parameter {par_name} "
                                f"to {value} ({pmin}, {pmax}) "
                                f"for pt {pt_min}-{pt_max} GeV/c..."
                            )
                if len(init_cfgs) > 0:
                    cfg_pars_dict['init_pars_sgn'] = []
                    for par in init_cfgs:
                        cfg_pars_dict['init_pars_sgn'].append(par)
                if len(fix_cfgs) > 0:
                    cfg_pars_dict['fix_pars_sgn'] = []
                    for par in fix_cfgs:
                        cfg_pars_dict['fix_pars_sgn'].append(par)
                break

            if cfg_fit.get('ParsFromFile'):
                do_init = True
                if isinstance(cfg_fit['ParsFromFile'], list) and not cfg_fit['ParsFromFile'][i_pt_bin]:
                    do_init = False  # do not init for this pt bin
                if cfg_pars_dict is None:
                    cfg_pars_dict = {'init_pars_sgn': [], 'fix_pars_sgn': []}
                if do_init:
                    file_path = cfg_fit['ParsFromFile'][i_pt_bin] if isinstance(cfg_fit['ParsFromFile'], list) else cfg_fit['ParsFromFile']
                    logger(f"Trying to fix signal parameters from file {file_path} for pt {pt_min} - {pt_max} GeV/c ...", "INFO")
                    try:
                        pt_dir = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
                        print(f"file_path: {file_path}, pt_dir: {pt_dir}")
                        fixParsFile = TFile.Open(file_path, 'r')
                        print(f"fixParsFile: {fixParsFile}")
                        fixParsHisto = fixParsFile.Get(f'{pt_dir}/hist_signal_func')
                        fixParsHisto.SetDirectory(0)
                        print(f"fixParsHisto: {fixParsHisto}")
                        for iBin in range(1, fixParsHisto.GetNbinsX()+1):
                            if iBin <= 3:    # skip integral, mean, sigma
                                continue
                            binLabel = fixParsHisto.GetXaxis().GetBinLabel(iBin)
                            binContent = fixParsHisto.GetBinContent(iBin)
                            print(f"Appending fixed parameter from file: {binLabel} = {binContent}")
                            cfg_pars_dict['fix_pars_sgn'].append((0, binLabel, binContent, [binContent, binContent]))
                        fixParsFile.Close()
                    except Exception as e:
                        logger(f"Exception {e} caught when trying to fix signal parameters from file {file_path}.", level='ERROR')

            if cfg_pars_dict is not None:
                fitter.set_fit_pars_from_config(cfg_pars_dict)
            status, converged = fitter.fit()

            fitter.plot_fit(False, True, loc=["lower left", "upper left"], \
                            path=f"{out_dir_pt}/fits/fM_fit_{cutset_suffix}.pdf",
                            out_file=out_file) # (log, show_extra_info)

            fit_info, _, _, _, _ = fitter.get_fit_info()
            for label in sgn_funcs.keys():
                histos_avgs[f"h_ry_{label}"].SetBinContent(i_cutset + 1, fit_info[label]["ry"])
                histos_avgs[f"h_ry_{label}"].SetBinError(i_cutset + 1, fit_info[label]["ry_unc"])

            if minimizer != "flarefly":
                logger("Skipping sWeights computation: not using flarefly minimizer", level="WARNING")
                continue

            s_weights_sgn = fitter.get_sweights_sgn(cfg_fit['SgnFuncLabel'])
            s_weights_sec_peak = None
            if cfg_fit.get('InclSecPeak'):
                if cfg_fit['InclSecPeak'][i_pt_bin] if isinstance(cfg_fit['InclSecPeak'], list) else cfg_fit['InclSecPeak']:
                    s_weights_sec_peak = fitter.get_sweights_sgn(cfg_fit['SgnFuncSecPeakLabel'])
            # Build sWeights
            sgn_weights = np.asarray(s_weights_sgn)
            if s_weights_sec_peak is not None:
                bkg_weights = (np.ones(len(sgn_weights)) - sgn_weights - np.asarray(s_weights_sec_peak))
            else:
                bkg_weights = np.ones(len(sgn_weights)) - sgn_weights

            with open(f"{out_dir_pt}/fits_status.txt", "a") as f:
                computed_sweights = True if s_weights_sgn is not None else False
                f.write(
                        f"{fitter.get_name()}: "
                        f"fit_res.status -> {status}, "
                        f"fit_res.converged -> {converged}, "
                        f"sweights computed -> {computed_sweights} "
                        f"sgn sweights: {len(sgn_weights)} vs {len(sel_df['fM'])}\n"
                        )

            if len(sgn_weights) != len(sel_df['fM']):
                sel_df = sel_df.query("fM > @mass_min and fM < @mass_max").reset_index(drop=True)

            for var, label in zip(infer_vars, infer_vars_labels):
                print(f"    Drawing {var}")

                # Create figure with two subplots (distros and ratio)
                fig, ax = plt.subplots(figsize=(8, 8))

                # Bins
                if var == "fCent":
                    cent_min, cent_max = get_centrality_bins(cfg['centrality'])[1]
                    nbins = cent_max - cent_min
                    var_range = (sel_df[var].min()-0.5, sel_df[var].max()+0.5)
                else:
                    nbins = 20 if var == "fPt" else 60
                    var_range = (sel_df[var].min(), sel_df[var].max())
                bin_edges = np.linspace(var_range[0], var_range[1], nbins + 1)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_widths = np.diff(bin_edges)

                values = sel_df[var].to_numpy()
                # Signal histogram + errors
                print(f"values.shape: {values.shape}, sgn_weights.shape: {sgn_weights.shape}, bkg_weights.shape: {bkg_weights.shape}")
                sgn_hist, _ = np.histogram(values, bins=bin_edges, weights=sgn_weights)
                sgn_sumw2, _ = np.histogram(values, bins=bin_edges, weights=sgn_weights**2)
                sgn_err = np.sqrt(sgn_sumw2)
                ax.step(bin_edges[:-1], sgn_hist, where="post", label="Signal", color="#1f77b4")
                ax.errorbar(bin_centers, sgn_hist, yerr=sgn_err, xerr=bin_widths / 2, fmt="o", color="#1f77b4", capsize=2)

                # Background histogram + errors
                bkg_hist, _ = np.histogram(values, bins=bin_edges, weights=bkg_weights)
                bkg_sumw2, _ = np.histogram(values, bins=bin_edges, weights=bkg_weights**2)
                bkg_err = np.sqrt(bkg_sumw2)
                ax.step(bin_edges[:-1], bkg_hist, where="post", label="Bkg", color="#ff7f0e")
                ax.errorbar(bin_centers, bkg_hist, yerr=bkg_err, xerr=bin_widths / 2, fmt="o", color="#ff7f0e", capsize=2)

                # Styling
                ax.set_ylabel("Entries")
                ax.set_xlabel(label)
                ax.set_title(var)
                ax.legend()

                fig.tight_layout()
                fig.savefig(os.path.join(out_dir_pt, f"vars/{var}_{cutset_suffix}.pdf"), dpi=300, bbox_inches="tight")
                plt.close(fig)

                # Fill histograms for averages
                weighted_sum_sgn = np.sum(bin_centers * sgn_hist)
                all_weights_sgn = np.sum(sgn_hist)
                avg_sgn = weighted_sum_sgn / all_weights_sgn
                avg_sgn_unc = np.sqrt( (((bin_centers * all_weights_sgn - weighted_sum_sgn) / all_weights_sgn**2)**2 * sgn_err**2 ).sum() )
                histos_avgs[f"h_{var}_sgn"].SetBinContent(i_cutset + 1, avg_sgn)
                histos_avgs[f"h_{var}_sgn"].SetBinError(i_cutset + 1, avg_sgn_unc)

                weighted_sum_bkg = np.sum(bin_centers * bkg_hist)
                all_weights_bkg = np.sum(bkg_hist)
                avg_bkg = weighted_sum_bkg / all_weights_bkg
                avg_bkg_unc = np.sqrt( (((bin_centers * all_weights_bkg - weighted_sum_bkg) / all_weights_bkg**2)**2 * bkg_err**2 ).sum() )
                histos_avgs[f"h_{var}_bkg"].SetBinContent(i_cutset + 1, avg_bkg)
                histos_avgs[f"h_{var}_bkg"].SetBinError(i_cutset + 1, avg_bkg_unc)

            # Reset fitter for new cutset: correlated bkg fracs will change
            fitter.reset()

        # Compute average pt
        histos_avgs["h_avg_pt"] = ROOT.TH1F("h_avg_pt", "h_avg_pt", 1, 0, 1)
        avg_pt = 0
        total_yield = 0
        last_ry = 0     # To skip duplicated cutsets
        for i_bin in range(len(cutset_files)):
            ry = histos_avgs[f"h_ry_{cfg_fit['SgnFuncLabel']}"].GetBinContent(i_bin + 1)
            # Compare with tolerance
            print(f"Checking cutset {i_bin} for pt bin {i_pt_bin}: ry = {histos_avgs[f'h_ry_{cfg_fit['SgnFuncLabel']}'].GetBinContent(i_bin + 1)}, last_ry = {last_ry}")
            if abs(histos_avgs[f"h_ry_{cfg_fit['SgnFuncLabel']}"].GetBinContent(i_bin + 1) - last_ry) < 10:
                print(f"Skipping duplicated cutset {i_bin} for pt bin {i_pt_bin}")
                continue
            total_yield += ry
            last_ry = ry
            avg_pt += (histos_avgs["h_fPt_sgn"].GetBinContent(i_bin + 1) * ry)
            print(f"Added term: {histos_avgs['h_fPt_sgn'].GetBinContent(i_bin + 1)} * {ry}")

        avg_pt /= total_yield
        print(f"Computed average pt for pt bin {i_pt_bin}: {avg_pt} (total yield: {total_yield})")
        histos_avgs["h_avg_pt"].SetBinContent(1, avg_pt)
        hist_summary_avg_pt.SetBinContent(i_pt_bin + 1, avg_pt)
        hist_summary_pt_shifts.SetBinContent(i_pt_bin + 1, avg_pt - hist_summary_pt_shifts.GetBinCenter(i_pt_bin + 1))

        # Write histograms to output ROOT file
        out_file.cd()
        for hist in histos_avgs.values():
            hist.Write()
        out_file.Close()

    summary_file = TFile.Open(f"{cfg['outdir']}/vn_extr_{cfg['suffix']}_{out_dir_type}/ptcenter_{minimizer}/pt_center_summary.root", "recreate")
    hist_summary_avg_pt.Write()
    hist_summary_pt_shifts.Write()
    summary_file.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pt-centering with sPlot and flarefly/roofit')
    parser.add_argument('config_file', help='Path to the input configuration file')
    parser.add_argument("--minimizer", "-m", type=str, default="flarefly", help="minimizer to use")
    parser.add_argument("--workers", "-w", type=int, default=1, help="number of workers")
    args = parser.parse_args()

    eval_pt_center(args.config_file, args.minimizer, args.workers)
