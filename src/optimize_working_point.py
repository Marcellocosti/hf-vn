'''
Scan two BDT score thresholds (ScoreBkg, ScoreFD) and map the invariant-mass
significance per pT bin, using a subset of preprocessed job sparses.
python3 optimize_working_point.py config.yml [-w N]
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
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT.gROOT.SetBatch(True)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{script_dir}/../flareflyfitter")
sys.path.append(f"{script_dir}/../utils")
from raw_yield_fitter import RawYieldFitter
from utils import logger

# Handles root files with removed Centrality axis
TITLE_TO_NAME = {
    'Inv. mass (GeV/#it{c}^{2})': 'Mass',
    '#it{p}_{T} (GeV/#it{c})':    'Pt',
    'Centrality':                 'Cent',
    'SP':                         'Sp',
    'Bkg score':                  'ScoreBkg',
    'FD score':                   'ScoreFD',
}

def build_axis_idx(merged):
    idx = {}
    for i in range(merged.GetNdimensions()):
        title = merged.GetAxis(i).GetTitle()
        idx[TITLE_TO_NAME.get(title, title)] = i
    return idx

# Returns the merged sparse from the job files
def load_merged_sparse(job_files, obj_path, n_jobs):
    files = sorted(job_files)
    if n_jobs is not None and n_jobs < len(files):
        files = files[:n_jobs]
    merged = None
    for jf in files:
        f = TFile.Open(jf)
        sp = f.Get(obj_path)
        if not sp:
            f.Close()
            continue
        merged = sp.Clone("merged_sparse") if merged is None else (merged.Add(sp), merged)[1]
        f.Close()
    return merged, files

# Builds list for threshold values to scan
def build_scan(axis_cfg, i_pt):
    lo = axis_cfg.get('min', 0.0)
    lo = lo[i_pt] if isinstance(lo, list) else lo
    hi = axis_cfg['max']
    hi = hi[i_pt] if isinstance(hi, list) else hi
    n = axis_cfg.get('nsteps', 5)
    n = n[i_pt] if isinstance(n, list) else n
    if lo <= 0:
        return [(k + 1) * hi / n for k in range(n)]
    return [float(v) for v in np.linspace(lo, hi, n)]

# Formats scan values
def fmt_scores(vals):
    if len(vals) > 1:
        span = min((b - a) for a, b in zip(sorted(vals), sorted(vals)[1:]))
    else:
        span = vals[0] if vals else 1.0
    dec = max(0, min(4, -int(np.floor(np.log10(span)))))
    return [f"{v:.{dec}f}" for v in vals]

# Gets peak mean and width from the fit result parameters
def get_mean_sigma(fitter):
    try:
        params = fitter.fit_result.params
    except Exception:
        return np.nan, np.nan
    mean, sigma = np.nan, np.nan
    for p, v in params.items():
        name = p if isinstance(p, str) else getattr(p, 'name', str(p))
        val = v['value'] if isinstance(v, dict) else float(v)
        if 'mu_signal' in name:
            mean = float(val)
        elif 'sigma_signal' in name:
            sigma = float(val)
    return mean, sigma

# Scans the grid, fits the mass at each point, saves the plots (2D significance map, 3 QA hists)
def process_pt_bin(args):
    cfg, i_pt, pt_min, pt_max, minimizer = args

    wp = cfg['working_point']
    fit_cfg = cfg['v2extraction']
    dmeson = cfg['Dmeson']
    input_dir = wp.get('input_dir', 'FlowSP')
    sparse_name = wp.get('sparse_name', 'FlowSP')
    obj_path = f"{sparse_name}/hSparse{sparse_name}"
    sgn_label = fit_cfg.get('SgnFuncLabel', dmeson)

    scan = wp['scan']
    sx, sy = list(scan.keys())
    scan_x = build_scan(scan[sx], i_pt)
    scan_y = build_scan(scan[sy], i_pt)

    pt_str = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
    job_glob = f"{cfg['outdir']}/preprocess/{pt_str}/{input_dir}/jobs/AnalysisResults_*.root"
    job_files = glob.glob(job_glob)
    if not job_files:
        logger(f"No job files at {job_glob}, skipping", "WARNING")
        return None

    n_jobs_cfg = wp.get('n_jobs')
    n_jobs = n_jobs_cfg[i_pt] if isinstance(n_jobs_cfg, list) else n_jobs_cfg
    merged, used_files = load_merged_sparse(job_files, obj_path, n_jobs)
    if merged is None:
        return None

    axis_idx = build_axis_idx(merged)
    for need in (sx, sy, 'Mass'):
        if need not in axis_idx:
            logger(f"Axis '{need}' not in sparse titles {list(axis_idx)} for {pt_str}", "ERROR")
            return None

    out_dir_pt = f"{cfg['outdir']}/working_point_{cfg['suffix']}/{pt_str}"
    os.makedirs(out_dir_pt, exist_ok=True)
    with open(f"{out_dir_pt}/used_jobs.txt", "w") as jf:
        jf.write("\n".join(used_files) + "\n")

    nx, ny = len(scan_x), len(scan_y)
    xlabels = fmt_scores(scan_x)
    ylabels = fmt_scores(scan_y)
    h_signif = TH2D(f"h_signif_{pt_str}", f"Significance {pt_str};{sx} max;{sy} max", nx, 0, nx, ny, 0, ny)
    h_mean = TH2D(f"h_mean_{pt_str}", f"Mass mean {pt_str};{sx} max;{sy} max", nx, 0, nx, ny, 0, ny)
    h_sigma = TH2D(f"h_sigma_{pt_str}", f"Mass sigma {pt_str};{sx} max;{sy} max", nx, 0, nx, ny, 0, ny)
    for h in (h_signif, h_mean, h_sigma):
        for i, lab in enumerate(xlabels):
            h.GetXaxis().SetBinLabel(i + 1, lab)
        for j, lab in enumerate(ylabels):
            h.GetYaxis().SetBinLabel(j + 1, lab)
    signif_grid = np.full((ny, nx), np.nan)

    mass_min, mass_max = fit_cfg['MassFitRanges'][i_pt]
    bkg_func_cfg, sgn_func_cfg = fit_cfg['BkgFunc'], fit_cfg['SgnFunc']
    bkg_func = bkg_func_cfg[i_pt] if isinstance(bkg_func_cfg, list) else bkg_func_cfg
    sgn_func = sgn_func_cfg[i_pt] if isinstance(sgn_func_cfg, list) else sgn_func_cfg

    fitter = RawYieldFitter(dmeson, pt_min, pt_max, pt_str, minimizer, verbose=False)
    fitter.set_fit_range(mass_min, mass_max)

    # Scans all (ScoreBkg max, ScoreFD max) combinations
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
                fitter.fit()
                info, *_ = fitter.get_fit_info()
                signif = info[sgn_label]['signif']
                mean, sigma = get_mean_sigma(fitter)
            except Exception as e:
                logger(f"    Fit failed for {tag}: {e}", "WARNING")
                signif, mean, sigma = -1.0, np.nan, np.nan

            # Fills QA hists and the 2D plot grid
            h_signif.SetBinContent(ix + 1, iy + 1, signif if signif > 0 else 0.0)
            signif_grid[iy, ix] = signif if signif > 0 else np.nan
            if np.isfinite(mean):
                h_mean.SetBinContent(ix + 1, iy + 1, mean)
            if np.isfinite(sigma):
                h_sigma.SetBinContent(ix + 1, iy + 1, sigma)
            logger(f"    [{pt_str}] {tag}: signif={signif:.2f} mean={mean:.4f} sigma={sigma:.4f}", "INFO")
            fitter.reset()

    # Generates significance 2D plot PDF
    fig, ax = plt.subplots(figsize=(1.6 * nx + 2, 1.2 * ny + 2))
    im = ax.imshow(np.ma.masked_invalid(signif_grid), origin="lower",
                   aspect="auto", cmap="viridis")
    ax.set_xticks(range(nx), xlabels)
    ax.set_yticks(range(ny), ylabels)
    ax.set_xlabel(f"{sx} max")
    ax.set_ylabel(f"{sy} max")
    ax.set_title(f"Significance {pt_str}")
    for iy in range(ny):
        for ix in range(nx):
            val = signif_grid[iy, ix]
            if np.isfinite(val):
                ax.text(ix, iy, f"{val:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Significance")
    fig.tight_layout()
    fig.savefig(f"{out_dir_pt}/significance_map_{pt_str}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger(f"Wrote significance map -> {out_dir_pt}/significance_map_{pt_str}.pdf", "INFO")

    out_path = f"{out_dir_pt}/qa_{pt_str}.root"
    fo = TFile.Open(out_path, "recreate")
    h_signif.Write()
    h_mean.Write()
    h_sigma.Write()
    fo.Close()
    logger(f"Wrote QA hists -> {out_path}", "INFO")
    return out_path

# Runs one task per pT bin, in parallel if -w > 1
def optimize_working_point(cfg_file, minimizer, workers):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    out_base = f"{cfg['outdir']}/working_point_{cfg['suffix']}"
    os.makedirs(out_base, exist_ok=True)

    pt_mins, pt_maxs = cfg['ptbins'][:-1], cfg['ptbins'][1:]
    tasks = [(cfg, i, lo, hi, minimizer) for i, (lo, hi) in enumerate(zip(pt_mins, pt_maxs))]

    results = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_pt_bin, t) for t in tasks]
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    results.append(r)
    else:
        for t in tasks:
            r = process_pt_bin(t)
            if r:
                results.append(r)

    logger(f"Done. Wrote {len(results)} per-bin QA file(s) under {out_base}", "INFO")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan ML-score working points and map significance")
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    parser.add_argument("-m", "--minimizer", default="flarefly", help="flarefly or roofit")
    parser.add_argument("-w", "--workers", type=int, default=1, help="parallel pt-bin workers")
    args = parser.parse_args()
    optimize_working_point(args.config_file, args.minimizer, args.workers)