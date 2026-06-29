'''
Scan two BDT score thresholds (ScoreBkg, ScoreFD) and map the invariant-mass
significance per pT bin, using a subset of preprocessed job sparses.
    python3 optimize_working_point.py config.yml
'''

import os
import sys
import glob
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import ROOT
from ROOT import TFile, TH2D

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT.gROOT.SetBatch(True)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{script_dir}/../flareflyfitter")
sys.path.append(f"{script_dir}/../utils")
from raw_yield_fitter import RawYieldFitter
from utils import logger

# build a dictionary of {name: index} from the config's 'names' list
def get_sparse_axis_indices(cfg, input_dir, sparse_name):
    for inp in cfg['preprocess']['inputs']:
        if inp.get('outdir') != input_dir:
            continue
        for sp in inp.get('sparses', []):
            if sp['name'] == sparse_name:
                names = sp['axes']['names']
                return {name: i for i, name in enumerate(names)}
    logger(f"Sparse '{sparse_name}' (outdir '{input_dir}') not found in preprocess config", "ERROR")
    sys.exit(1)

# sum the first n_jobs job sparses in memory
def load_merged_sparse(job_files, obj_path, n_jobs):
    files = sorted(job_files)
    if n_jobs is not None and n_jobs < len(files):
        files = files[:n_jobs]
    logger(f"    Using {len(files)} job file(s)", "INFO")

    merged = None
    for jf in files:
        f = TFile.Open(jf)
        sp = f.Get(obj_path)
        if not sp:
            logger(f"    {obj_path} not found in {jf}, skipping", "WARNING")
            f.Close()
            continue
        if merged is None:
            merged = sp.Clone("merged_sparse")
        else:
            merged.Add(sp)
        f.Close()
    return merged, files


def optimize_working_point(cfg_file, minimizer):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    wp = cfg['working_point']
    fit_cfg = cfg['v2extraction']
    dmeson = cfg['Dmeson']

    input_dir = wp.get('input_dir', 'FlowSP')
    sparse_name = wp.get('sparse_name', 'FlowSP')
    obj_path = f"{sparse_name}/hSparse{sparse_name}"
    axis_idx = get_sparse_axis_indices(cfg, input_dir, sparse_name)

    # two axes to scan: first key is x-axis, second key is y-axis
    scan = wp['scan']
    score_names = list(scan.keys())
    if len(score_names) != 2:
        logger(f"Expected exactly 2 score axes to scan, got {score_names}", "ERROR")
        sys.exit(1)
    sx, sy = score_names
    scan_x, scan_y = scan[sx], scan[sy]
    for s in (sx, sy):
        if s not in axis_idx:
            logger(f"Scan axis '{s}' not among sparse axes {list(axis_idx)}", "ERROR")
            sys.exit(1)

    sgn_func_cfg = fit_cfg['SgnFunc']
    bkg_func_cfg = fit_cfg['BkgFunc']
    sgn_label = fit_cfg.get('SgnFuncLabel', dmeson)
    save_fits = wp.get('save_fits', True)

    out_base = f"{cfg['outdir']}/working_point_{cfg['suffix']}"
    os.makedirs(out_base, exist_ok=True)
    summary = TFile.Open(f"{out_base}/significance_summary.root", "recreate")

    for i_pt, (pt_min, pt_max) in enumerate(zip(cfg['ptbins'][:-1], cfg['ptbins'][1:])):
        pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        logger(f"### {pt_str} ({pt_min} - {pt_max} GeV/c) ###", "INFO")

        # load job files for this pt bin
        job_glob = f"{cfg['outdir']}/preprocess/{pt_str}/{input_dir}/jobs/AnalysisResults_*.root"
        job_files = glob.glob(job_glob)
        if not job_files:
            logger(f"No job files at {job_glob}, skipping", "WARNING")
            continue

        n_jobs_cfg = wp.get('n_jobs')
        n_jobs = n_jobs_cfg[i_pt] if isinstance(n_jobs_cfg, list) else n_jobs_cfg
        merged, used_files = load_merged_sparse(job_files, obj_path, n_jobs)
        if merged is None:
            logger(f"Could not build merged sparse for {pt_str}, skipping", "WARNING")
            continue

        out_dir_pt = f"{out_base}/{pt_str}"
        os.makedirs(f"{out_dir_pt}/fits", exist_ok=True)
        with open(f"{out_dir_pt}/used_jobs.txt", "w") as jf:
            jf.write("\n".join(used_files) + "\n")

        # TH2D and numpy grid to store significance at each (ScoreBkg, ScoreFD) point
        h_signif = TH2D(f"h_signif_{pt_str}", f"Significance {pt_str};{sx} max;{sy} max",
                        len(scan_x), 0, len(scan_x), len(scan_y), 0, len(scan_y))
        for i, v in enumerate(scan_x):
            h_signif.GetXaxis().SetBinLabel(i + 1, f"{v:g}")
        for j, v in enumerate(scan_y):
            h_signif.GetYaxis().SetBinLabel(j + 1, f"{v:g}")
        signif_grid = np.full((len(scan_y), len(scan_x)), np.nan)

        mass_min, mass_max = fit_cfg['MassFitRanges'][i_pt]
        bkg_func = bkg_func_cfg[i_pt] if isinstance(bkg_func_cfg, list) else bkg_func_cfg
        sgn_func = sgn_func_cfg[i_pt] if isinstance(sgn_func_cfg, list) else sgn_func_cfg

        fitter = RawYieldFitter(dmeson, pt_min, pt_max, pt_str, minimizer, verbose=False)
        fitter.set_fit_range(mass_min, mass_max)

        best = {'signif': -np.inf, 'fig': None, 'tag': None}

        # scan all (ScoreBkg max, ScoreFD max) combinations
        for ix, x_max in enumerate(scan_x):
            for iy, y_max in enumerate(scan_y):
                # apply score cuts and project mass axis
                merged.GetAxis(axis_idx[sx]).SetRangeUser(0.0, x_max)
                merged.GetAxis(axis_idx[sy]).SetRangeUser(0.0, y_max)
                h_mass = merged.Projection(axis_idx['Mass'])
                h_mass.SetName(f"hMass_{pt_str}_{ix}_{iy}")
                h_mass.SetDirectory(0)

                tag = f"{sx}{x_max:g}_{sy}{y_max:g}"
                fitter.add_bkg_func(bkg_func, "Comb. bkg")
                fitter.add_sgn_func(sgn_func, sgn_label, dmeson)
                fitter.set_name(f"{pt_str}_{tag}")
                fitter.set_data_to_fit_hist(h_mass)
                fitter.setup()

                try:
                    status, converged = fitter.fit()
                    info, *_ = fitter.get_fit_info()
                    signif = info[sgn_label]['signif']
                except Exception as e:
                    logger(f"    Fit failed for {tag}: {e}", "WARNING")
                    status, converged, signif = -1, False, -1.0

                # track best working point
                if signif > best['signif']:
                    best['signif'] = signif
                    best['tag']    = tag
                    if save_fits and signif > 0:
                        try:
                            if best['fig'] is not None:
                                plt.close(best['fig'])
                            best['fig'] = fitter.plot_fit(False, True)
                        except Exception as e:
                            logger(f"    Could not plot fit for {tag}: {e}", "WARNING")

                h_signif.SetBinContent(ix + 1, iy + 1, signif if signif > 0 else 0.0)
                signif_grid[iy, ix] = signif if signif > 0 else np.nan
                logger(f"    {tag}: signif = {signif:.2f} (status {status}, conv {converged})", "INFO")
                fitter.reset()

        # save best fit plot
        if save_fits and best['fig'] is not None:
            best['fig'].savefig(f"{out_dir_pt}/fits/best_fit_{best['tag']}.pdf",
                                dpi=200, bbox_inches="tight")
            plt.close(best['fig'])
            logger(f"Best signif {best['signif']:.2f} at {best['tag']} "
                   f"-> {out_dir_pt}/fits/best_fit_{best['tag']}.pdf", "INFO")

        summary.cd()
        h_signif.Write()

        # produce significance heatmap
        fig, ax = plt.subplots(figsize=(1.6 * len(scan_x) + 2, 1.2 * len(scan_y) + 2))
        im = ax.imshow(np.ma.masked_invalid(signif_grid), origin="lower",
                       aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(scan_x)), [f"{v:g}" for v in scan_x])
        ax.set_yticks(range(len(scan_y)), [f"{v:g}" for v in scan_y])
        ax.set_xlabel(f"{sx} max")
        ax.set_ylabel(f"{sy} max")
        ax.set_title(f"Significance {pt_str}")
        for iy in range(len(scan_y)):
            for ix in range(len(scan_x)):
                val = signif_grid[iy, ix]
                if np.isfinite(val):
                    ax.text(ix, iy, f"{val:.1f}", ha="center", va="center",
                            color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Significance")
        fig.tight_layout()
        fig.savefig(f"{out_dir_pt}/significance_map_{pt_str}.pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger(f"Wrote significance map -> {out_dir_pt}/significance_map_{pt_str}.pdf", "INFO")

    summary.Close()
    logger(f"Done. Summary: {out_base}/significance_summary.root", "INFO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan ML-score working points and map significance")
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    parser.add_argument("-m", "--minimizer", default="flarefly", help="flarefly or roofit")
    args = parser.parse_args()
    optimize_working_point(args.config_file, args.minimizer)