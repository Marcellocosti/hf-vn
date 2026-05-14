#!/usr/bin/env python3
"""
Supports two extraction modes:
  - DeltaPhiBinning (PairYieldsVsPhi.root):
    ONE histogram per (ptCand, ptHad).  Single integrated mass bin.
  - MassBinning (CorrelationsResults.root):
    One mass sub-bin per (ptCand, mass) — different pt bins may have
    different mass edge arrays.

Input YAML config:
  - Dmeson, outdir, suffix        : identify D meson species & extraction output
  - ptBinsCand, ptBinsHad, invMassBins : binning definitions
  - task_LM (optional)            : low-multiplicity template config
  - fitConfig (optional)          : fit parameters (FitFunction, FixBaseline, …)

Output:
  {outdir}/CorrelExtract_{suffix}/CorrelationFitResults/
    Output_CorrelationFitting_Root/CorrPhi{DMeson}_FinalPlots.root
    outputPathOutput_CorrelationFitting_png/*.png

Usage:
  python3 FitCorrel.py config_CorrAnalysis_v2_010_negDeta.yaml
"""

import argparse
import math
import os
import sys
from array import array
from ctypes import c_int, c_double

import ROOT
import yaml

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# ===================================================================
# Compile DhCorrelationFitter C++ class
# ===================================================================
ROOT.gSystem.AddIncludePath("-I/home/wuct/Software/miniforge3/envs/alice/include")
_fitter_cxx = os.path.join(os.path.dirname(__file__), "DhCorrelationFitter.cxx")
ROOT.gSystem.CompileMacro(_fitter_cxx, "kO")
from ROOT import DhCorrelationFitter

# ===================================================================
# ROOT style helpers
# ===================================================================
def set_canvas_style():
    style = ROOT.gStyle
    style.SetOptStat(0)
    style.SetPadLeftMargin(0.2)
    style.SetPadRightMargin(0.005)
    style.SetPadBottomMargin(0.2)
    style.SetFrameLineWidth(2)
    style.SetLineWidth(2)
    style.SetCanvasDefH(1126)
    style.SetCanvasDefW(1840)


def set_th1_style(histo, title, x_title, y_title,
                  marker_style=ROOT.kFullCircle,
                  marker_color=ROOT.kRed + 1,
                  marker_size=1.4,
                  line_color=ROOT.kRed + 1,
                  line_width=3):
    histo.SetTitle(title)
    histo.GetXaxis().SetTitle(x_title)
    histo.GetYaxis().SetTitle(y_title)
    histo.SetMarkerStyle(marker_style)
    histo.SetMarkerColor(marker_color)
    histo.SetMarkerSize(marker_size)
    histo.SetLineColor(line_color)
    histo.SetLineWidth(line_width)
    histo.GetXaxis().SetTitleOffset(0.8)
    histo.GetYaxis().SetTitleOffset(1.3)
    histo.GetXaxis().SetTitleSize(0.045)
    histo.GetYaxis().SetTitleSize(0.045)
    histo.GetXaxis().SetLabelSize(0.045)
    histo.GetYaxis().SetLabelSize(0.045)


# ===================================================================
# Directory name helpers
# ===================================================================
def pt_cand_dir_name(pt_min, pt_max):
    low = int(round(pt_min * 10))
    high = int(round(pt_max * 10))
    return f"PtCandBin_{low}_{high}"


def pt_had_dir_name(pt_min, pt_max):
    low = int(round(pt_min * 10))
    high = int(round(pt_max * 10))
    return f"PtHadBin_{low}_{high}"


# ===================================================================
# Helper: ensure per-bin parameter values
# ===================================================================
def ensure_per_bin(value, n_bins, name, default):
    if value is None:
        return [default] * n_bins
    if isinstance(value, list):
        if len(value) < n_bins:
            raise ValueError(f"{name} must provide at least {n_bins} entries, got {len(value)}")
        return value[:n_bins]
    return [value] * n_bins


# ===================================================================
# Main entry point
# ===================================================================
def fit_correl(cfg_path):
    set_canvas_style()

    # ---- Load config --------------------------------------------------
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    suffix = config["suffix"]
    outdir = config["outdir"]

    # ---- D meson species ----------------------------------------------
    Dmeson = config.get("Dmeson", "Dzero")
    species_map = {
        "D0": (0, "D0", "D^{0}"),
        "Dzero": (0, "D0", "D^{0}"),
        "Dplus": (1, "Dplus", "D^{+}"),
        "Ds": (2, "Ds", "D_{s}^{+}"),
    }
    if Dmeson not in species_map:
        print(f"[ERROR] Unknown D meson: {Dmeson}")
        sys.exit(1)
    dmeson_species, dmeson_name, dmeson_label = species_map[Dmeson]

    # ---- Paths --------------------------------------------------------
    extract_dir = os.path.join(outdir, f"CorrelExtract_{suffix}")

    pair_yields_path = os.path.join(extract_dir, "AssociatedPairsYields",
                                    "PairYieldsVsPhi.root")
    correlations_path = os.path.join(extract_dir, "CorrelationsResults",
                                     "CorrelationsResults.root")

    use_pairs_file = os.path.exists(pair_yields_path)
    input_file_path = pair_yields_path if use_pairs_file else correlations_path

    if not os.path.exists(input_file_path):
        print(f"[ERROR] Input file not found")
        print(f"  Checked: {pair_yields_path}")
        print(f"  Checked: {correlations_path}")
        sys.exit(1)

    output_path = os.path.join(extract_dir, "CorrelationFitResults")
    out_png_dir = os.path.join(output_path, "outputPathOutput_CorrelationFitting_png")
    out_root_dir = os.path.join(output_path, "Output_CorrelationFitting_Root")
    os.makedirs(out_png_dir, exist_ok=True)
    os.makedirs(out_root_dir, exist_ok=True)

    print(f"[INFO] Input file:  {input_file_path}")
    print(f"[INFO] Output PNG:  {out_png_dir}")
    print(f"[INFO] Output ROOT: {out_root_dir}")

    # ---- Binning ------------------------------------------------------
    pt_bins_cand = [float(x) for x in config["ptBinsCand"]]
    pt_bins_had = [float(x) for x in config["ptBinsHad"]]
    inv_mass_bins_raw = config["invMassBins"]

    n_pt_cand = len(pt_bins_cand) - 1
    n_pt_had = len(pt_bins_had) - 1

    # Build mass edges per pt-candidate bin (different pt bins may have
    # different mass edge arrays)
    mass_edges_per_pt = []
    for i_pt in range(n_pt_cand):
        if i_pt < len(inv_mass_bins_raw):
            edges = [float(x) for x in inv_mass_bins_raw[i_pt]]
        else:
            edges = [float(x) for x in inv_mass_bins_raw[-1]]
        mass_edges_per_pt.append(edges)

    if use_pairs_file:
        # ---- DeltaPhiBinning ----------------
        # PairYieldsVsPhi.root has ONE histogram per (ptCand, ptHad).
        # Single integrated mass bin (matching C++ nBinsInvMass=1).
        # All ptCand fits fill one output histogram per (ptHad).
        mass_min = mass_edges_per_pt[0][0]
        mass_max = mass_edges_per_pt[0][-1]
        n_mass_total = 1
        mass_combos = [(0, mass_min, mass_max)]  # dummy entry for the loop
        print(f"[INFO] Mass range: [{mass_min}, {mass_max}]  "
              f"(single integrated bin — DeltaPhiBinning)")
    else:
        # ---- MassBinning ----------------------------
        # CorrelationsResults.root has one directory per mass sub-bin.
        mass_combos = []
        for i_pt in range(n_pt_cand):
            sub_edges = mass_edges_per_pt[i_pt]
            for i_mass_local in range(len(sub_edges) - 1):
                mass_combos.append((i_pt, i_mass_local,
                                    sub_edges[i_mass_local],
                                    sub_edges[i_mass_local + 1]))
        n_mass_total = len(mass_combos)
        print(f"[INFO] MassBinning: {n_mass_total} mass sub-bins "
              f"across {n_pt_cand} pt cand bins")

    # ---- LM template --------------------------------------------------
    method = config.get("method", "DeltaPhiBinning")
    task_lm = config.get("task_LM", {})
    lm_template_path = None
    if task_lm and task_lm.get("do", False):
        lm_outdir = task_lm.get("outdir", "")
        if lm_outdir:
            if method == "MassBinning":
                lm_template_path = os.path.join(
                    lm_outdir, f"CorrelExtract_{suffix}",
                    "CorrelationsResults", "CorrelationsResults.root",
                )
            else:
                lm_template_path = os.path.join(
                    lm_outdir, f"CorrelExtract_{suffix}",
                    "AssociatedPairsYields", "PairYieldsVsPhi.root",
                )
            if not os.path.exists(lm_template_path):
                print(f"[WARNING] LM template not found: {lm_template_path}")
                lm_template_path = None
            else:
                print(f"[INFO] LM template: {lm_template_path}")

    # ---- Fit config ---------------------------------------------------
    fit_config = config.get("fitConfig", {})
    fit_functions_raw = fit_config.get("FitFunction", config.get("FitFunction"))
    if fit_functions_raw is None:
        fit_functions = [8] * n_pt_cand
    else:
        fit_functions = ensure_per_bin(fit_functions_raw, n_pt_cand,
                                       "FitFunction", 8)
    fit_functions = [int(f) for f in fit_functions]

    # If no LM template, force all to type 8
    if lm_template_path is None:
        for i in range(n_pt_cand):
            fit_functions[i] = DhCorrelationFitter.kV2DeltaModulationLowMult
        print("[INFO] No LM template — all fit functions forced to type 8")

    fix_baseline = int(fit_config.get("FixBaseline", config.get("FixBaseline", 0)))
    fix_mean = int(fit_config.get("FixMean", config.get("FixMean", 0)))
    n_baseline_points = int(fit_config.get("nBaselinePoints",
                                         config.get("nBaselinePoints", 0)))
    points_for_baseline = fit_config.get("binsForBaseline",
                                       config.get("binsForBaseline", []))
    if n_baseline_points and len(points_for_baseline) != n_baseline_points:
        print("[ERROR] 'binsForBaseline' length must match 'nBaselinePoints'")
        sys.exit(1)
    points_for_baseline = [int(x) for x in points_for_baseline]

    with_ped_lm = fit_config.get("WithPedLM", False)
    fix_lm_factor = fit_config.get("FixLMFactor", False)

    par_vals = [float(x) for x in fit_config.get("parVals", config.get("parVals", []))]
    par_low = [float(x) for x in fit_config.get("parLowBounds", config.get("parLowBounds", []))]
    par_up = [float(x) for x in fit_config.get("parUpperBounds", config.get("parUpperBounds", []))]
    if par_vals and not (len(par_vals) == len(par_low) == len(par_up)):
        print("[ERROR] parVals, parLowBounds, parUpperBounds must have same length")
        sys.exit(1)

    is_reflected = config.get("IsReflected", config.get("IsRiflected", False))
    shift_base_up = config.get("ShiftBaseUp", False)
    shift_base_down = config.get("ShiftBaseDown", False)
    draw_systematics = config.get("DrawSystematics", False)
    same_systematics = config.get("SameSystematics", False)

    # ---- Print config summary -----------------------------------------
    print("===========================")
    print("Input variables from config")
    for i, func_val in enumerate(fit_functions):
        print(f"  iPt = {i + 1}  FitFunction = {func_val}")
    print(f"  FixBaseline = {fix_baseline}")
    print(f"  FixMean     = {fix_mean}")
    print("===========================\n")

    # ---- Open input file -----------------------------------------------
    in_file = ROOT.TFile.Open(input_file_path)
    if not in_file or in_file.IsZombie():
        print(f"[ERROR] Could not open {input_file_path}")
        sys.exit(1)

    in_file_lm = None
    if lm_template_path:
        in_file_lm = ROOT.TFile.Open(lm_template_path)
        if not in_file_lm or in_file_lm.IsZombie():
            print(f"[WARNING] Could not open LM template: {lm_template_path}")
            in_file_lm = None

    # ---- Fit range -------------------------
    f_min = -0.5 * math.pi
    f_max = 1.5 * math.pi

    # ---- Prepare output histograms (indexed by [i_pt_had][i_mass_local])
    pt_cand_edges_arr = array("d", pt_bins_cand)

    # For MassBinning: use local mass index; for DeltaPhiBinning: 1 mass bin
    if not use_pairs_file:
        # MassBinning: n_mass_local = mass sub-bins per ptCand
        n_mass_local = max(len(me) - 1 for me in mass_edges_per_pt)
    else:
        n_mass_local = 1

    h_v2_delta = [[None] * n_mass_local for _ in range(n_pt_had)]
    h_lm_factor = [None] * n_pt_had

    for i_pt_had in range(n_pt_had):
        for i_mass_local in range(n_mass_local):
            tag = f"PtBinAssoc{i_pt_had + 1}_InvMassBin{i_mass_local + 1}"
            h = ROOT.TH1D(f"hv2Delta_{tag}", "", n_pt_cand, pt_cand_edges_arr)
            h.SetDirectory(ROOT.nullptr)
            h_v2_delta[i_pt_had][i_mass_local] = h

        # LM Factor: single 1D histogram per ptHad (no mass binning)
        h_lm = ROOT.TH1D(f"hLMFactor_PtBinAssoc{i_pt_had + 1}", "", n_pt_cand, pt_cand_edges_arr)
        h_lm.SetDirectory(ROOT.nullptr)
        h_lm_factor[i_pt_had] = h_lm

    # ============== MAIN FIT LOOP ==============
    # For use_pairs_file (DeltaPhiBinning): iterate over ptCand internally
    # For !use_pairs_file (MassBinning): each mass_combos entry carries
    #   its own (i_pt_cand, mass_min, mass_max).
    for i_mass_global, mass_combo in enumerate(mass_combos):
        if len(mass_combo) == 4:
            mass_i_pt, mass_i_local, mass_min, mass_max = mass_combo
        else:
            mass_i_pt, mass_min, mass_max = mass_combo
            mass_i_local = 0
        # Determine which ptCand bins to process in this mass combo
        if use_pairs_file:
            cand_indices = list(range(n_pt_cand))
            mass_range_label = f"[{mass_min}, {mass_max}]"
        else:
            cand_indices = [mass_i_pt]
            mass_range_label = (f"[{mass_min}, {mass_max}]  "
                                f"PtCand[{pt_bins_cand[mass_i_pt]}, {pt_bins_cand[mass_i_pt+1]}]")

        for i_pt_cand in cand_indices:
            pt_cand_min = pt_bins_cand[i_pt_cand]
            pt_cand_max = pt_bins_cand[i_pt_cand + 1]
            print(f"\\n[INFO] Mass[{mass_i_local + 1}/{n_mass_local}] "
                  f"PtCand[{pt_cand_min}, {pt_cand_max}]  {mass_range_label}")

            for i_pt_had, (pt_had_min, pt_had_max) in enumerate(
                    zip(pt_bins_had[:-1], pt_bins_had[1:])):
                print(f"[INFO] PtHad: {pt_had_min} - {pt_had_max}")

                # ---- Build histogram path --------------------------------
                pc_dir = pt_cand_dir_name(pt_cand_min, pt_cand_max)
                ph_dir = pt_had_dir_name(pt_had_min, pt_had_max)

                if use_pairs_file:
                    hist_path = f"{pc_dir}/{ph_dir}/hPairsYields_vs_DeltaPhi"
                else:
                    im_dir = (f"InvMassBin_{int(mass_min * 1000):.0f}_"
                              f"{int(mass_max * 1000):.0f}")
                    hist_path = f"{pc_dir}/{ph_dir}/{im_dir}/hCorrectedCorrel"

                h_corr = in_file.Get(hist_path)
                if not h_corr and not use_pairs_file:
                    base_dir = f"{pc_dir}/{ph_dir}"
                    dir_obj = in_file.Get(base_dir)
                    if dir_obj and dir_obj.IsFolder():
                        for sk in dir_obj.GetListOfKeys():
                            sub_path = f"{base_dir}/{sk.GetName()}/hCorrectedCorrel"
                            h_corr = in_file.Get(sub_path)
                            if h_corr:
                                print(f"    Found: {sub_path}")
                                break
                if not h_corr:
                    print(f"    [WARNING] Histogram not found: {hist_path}, skip")
                    continue
                print(f"    Histogram: {hist_path}")

                # ---- LM template -----------------------------------------
                h_corr_lm = None
                if in_file_lm:
                    h_corr_lm = in_file_lm.Get(hist_path)
                    if not h_corr_lm:
                        alt_path = (f"{pc_dir}/{ph_dir}/hPairsYields_vs_DeltaPhi")
                        h_corr_lm = in_file_lm.Get(alt_path)
                        if h_corr_lm:
                            print(f"    LM template from: {alt_path}")
                        else:
                            print(f"    [WARNING] LM template not found: {alt_path}")
                use_template = h_corr_lm is not None

                # ---- Read ry_trigger --------------------------------------
                ry_val = 1.0
                ry_err = 0.0
                h_ry = in_file.Get(f"{pc_dir}/{ph_dir}/ry_trigger")
                if h_ry:
                    ry_val = h_ry.GetBinContent(1)
                    ry_err = h_ry.GetBinError(1)
                    print(f"    ry_trigger = {ry_val} +/- {ry_err}")
                lm_ry_val = 1.0
                lm_ry_err = 0.0
                if in_file_lm:
                    h_lm_ry = in_file_lm.Get(f"{pc_dir}/{ph_dir}/ry_trigger")
                    if h_lm_ry:
                        lm_ry_val = h_lm_ry.GetBinContent(1)
                        lm_ry_err = h_lm_ry.GetBinError(1)
                        print(f"    LM ry_trigger = {lm_ry_val} +/- {lm_ry_err}")
                    else:
                        h_lm_ry = in_file_lm.Get(f"{pc_dir}/{ph_dir}/ry_trigger")
                        if h_lm_ry:
                            lm_ry_val = h_lm_ry.GetBinContent(1)
                            lm_ry_err = h_lm_ry.GetBinError(1)

                # ---- Convert to TH1F for the fitter ---------------------
                h_fit = ROOT.TH1F(
                    f"hFit_{h_corr.GetName()}", h_corr.GetTitle(),
                    h_corr.GetNbinsX(),
                    h_corr.GetXaxis().GetXmin(),
                    h_corr.GetXaxis().GetXmax())
                for i_bin in range(1, h_corr.GetNbinsX() + 1):
                    h_fit.SetBinContent(i_bin, h_corr.GetBinContent(i_bin))
                    h_fit.SetBinError(i_bin, h_corr.GetBinError(i_bin))
                ROOT.SetOwnership(h_fit, False)
                set_th1_style(h_fit, "",
                                "#Delta#phi [rad]",
                                "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")

                # ---- Create DhCorrelationFitter -------------------------
                corr_fitter = DhCorrelationFitter(h_fit, f_min, f_max)
                ROOT.SetOwnership(corr_fitter, True)

                corr_fitter.SetHistoIsReflected(False)

                if use_template:
                    corr_fitter.SetWithPedLM(with_ped_lm)
                    corr_fitter.SetFixLMFactor(fix_lm_factor)
                    corr_fitter.SetFixBaseline(fix_baseline)
                    corr_fitter.SetBaselineUpOrDown(shift_base_up, shift_base_down)
                    if n_baseline_points > 0:
                        baseline_arr = (c_int * n_baseline_points)(*points_for_baseline)
                        corr_fitter.SetPointsForBaseline(n_baseline_points, baseline_arr)
                else:
                    corr_fitter.SetFixBaseline(0)
                    corr_fitter.SetBaselineUpOrDown(False, False)

                corr_fitter.SetReflectedCorrHisto(not is_reflected)
                corr_fitter.SetFixMean(fix_mean)
                corr_fitter.SetPtRanges(pt_cand_min, pt_cand_max,
                                        pt_had_min, pt_had_max)

                # Pass ry_trigger for raw-yield-scaled F initial value & limits
                corr_fitter.SetRyTrigger(ry_val, ry_err)
                if use_template:
                    corr_fitter.SetLMRyTrigger(lm_ry_val, lm_ry_err)

                if h_corr_lm:
                    corr_fitter.SetLMTemplate(h_corr_lm)

                if par_vals:
                    npars = len(par_vals)
                    vals = (c_double * npars)(*par_vals)
                    lows = (c_double * npars)(*par_low)
                    ups = (c_double * npars)(*par_up)
                    corr_fitter.SetExternalValsAndBounds(npars, vals, lows, ups)

                # ---- Fit -------------------------------------------------
                func_type = DhCorrelationFitter.FunctionType(fit_functions[i_pt_cand])
                corr_fitter.SetFuncType(func_type)

                mass_idx_label = mass_i_local + 1

                # create canvas FIRST then Fitting()
                # draws fit + split-term components onto it.
                canvas_name = (f"CanvasCorrPhi_PtBinCand{i_pt_cand + 1}_"
                               f"PtBinAssoc{i_pt_had + 1}_InvMassBin{mass_idx_label}")
                canvas_title = (f"CorrPhi{dmeson_name}_PtBinCand{i_pt_cand + 1}_"
                                f"PtBinAssoc{i_pt_had + 1}_InvMassBin{mass_idx_label}")
                canvas = ROOT.TCanvas(canvas_name, canvas_title, 1840, 1126)
                canvas.SetBottomMargin(0.08)
                canvas.SetLeftMargin(0.12)
                canvas.SetRightMargin(0.02)
                canvas.SetTopMargin(0.1)
                ROOT.SetOwnership(canvas, False)
                canvas.SetTickx()
                canvas.SetTicky()
                canvas.cd()

                # Fitting(drawSplitTerm=kTRUE, useExternalPars=kTRUE)
                corr_fitter.Fitting(True, True)
                # canvas.Update()
                h_fit.GetYaxis().SetRangeUser(h_fit.GetMinimum()*0.96, h_fit.GetMaximum() * 1.05)
                set_th1_style(h_fit, f"",
                                "#Delta#phi [rad]",
                                "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")
                canvas.Update()

                # ---- Get v2_delta result ---------------------------------
                v2_val = corr_fitter.Getv2Delta()
                v2_err = corr_fitter.Getv2DeltaError()
                bin_idx = i_pt_cand + 1

                print(f"      fit: v2_delta={v2_val:.6f} +/- {v2_err:.6f}")

                # ---- Fill output histogram -------------------------------
                mass_idx = mass_i_local if not use_pairs_file else i_mass_global
                h_v2_delta[i_pt_had][mass_idx].SetBinContent(bin_idx, v2_val)
                h_v2_delta[i_pt_had][mass_idx].SetBinError(bin_idx, v2_err)

                # ---- Fill LM Factor histogram (par 0) ---------------------
                if use_template and mass_i_local == 0:
                    lm_val = corr_fitter.GetLMFactor()
                    lm_err = corr_fitter.GetLMFactorError()
                    print(f"      LM Factor = {lm_val:.6f} +/- {lm_err:.6f}")
                    h_lm_factor[i_pt_had].SetBinContent(bin_idx, lm_val)
                    h_lm_factor[i_pt_had].SetBinError(bin_idx, lm_err)

                # ---- Finish drawing the canvas ---------------------------
                set_th1_style(h_corr, "",
                              "#Delta#phi [rad]",
                              "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")
                h_corr.SetStats(0)
                h_corr.SetMinimum(0)
                h_corr.Draw("same")

                pt_text = ROOT.TPaveText(0.15, 0.9, 0.85, 0.95, "NDC")
                pt_text.SetFillStyle(0)
                pt_text.SetBorderSize(0)
                pt_text.AddText(
                    0., 0.8,
                    f"{pt_cand_min:.1f} < p_{{T}}^{{{dmeson_label}}} < "
                    f"{pt_cand_max:.1f} GeV/c, "
                    f"{pt_had_min:.1f} < p_{{T}}^{{assoc}} < {pt_had_max:.1f} GeV/c, "
                    f"Mass[{mass_min:.3f}, {mass_max:.3f}] GeV/c^{{2}}")
                pt_text.SetTextAlign(22)
                pt_text.Draw("same")

                png_path = os.path.join(
                    out_png_dir,
                    f"CorrPhi{dmeson_name}_PtBinCand{i_pt_cand + 1}_"
                    f"PtBinAssoc{i_pt_had + 1}_InvMassBin{mass_idx_label}.png")
                root_canvas_path = os.path.join(
                    out_root_dir,
                    f"CorrPhi{dmeson_name}_PtBinCand{i_pt_cand + 1}_"
                    f"PtBinAssoc{i_pt_had + 1}_InvMassBin{mass_idx_label}.root")
                canvas.SaveAs(png_path)
                canvas.SaveAs(root_canvas_path)
                canvas.Close()

                del corr_fitter, h_fit, canvas

    # ---- Close input files ---------------------------------------------
    in_file.Close()
    if in_file_lm:
        in_file_lm.Close()

    # ============== SAVE FINAL PLOTS ==============
    final_plot_path = os.path.join(out_root_dir,
                                   f"CorrPhi{dmeson_name}_FinalPlots.root")
    print(f"\n[INFO] Saving final plots to: {final_plot_path}")

    out_file = ROOT.TFile(final_plot_path, "RECREATE")
    out_file.cd()

    n_mass_out = n_mass_local
    for i_mass_out in range(n_mass_out):
        for i_pt_had in range(n_pt_had):
            h = h_v2_delta[i_pt_had][i_mass_out]
            h.SetDirectory(out_file)
            out_file.cd()
            h.Write()
    # Write LM Factor histograms (flat — one per ptHad, no mass binning)
    for i_pt_had in range(n_pt_had):
        h_lm = h_lm_factor[i_pt_had]
        h_lm.SetDirectory(out_file)
        out_file.cd()
        h_lm.Write()
    # todo: to be removed
    for i_mass_out in range(n_mass_out):
        for i_pt_had in range(n_pt_had):
            src = h_v2_delta[i_pt_had][i_mass_out]
            for name_template in [
                "hBaselin_PtBinAssoc{}_InvMassBin{}",
                "hNSYield_PtBinAssoc{}_InvMassBin{}",
                "hNSSigma_PtBinAssoc{}_InvMassBin{}",
                "hASYield_PtBinAssoc{}_InvMassBin{}",
                "hASSigma_PtBinAssoc{}_InvMassBin{}",
                "hBeta_PtBinAssoc{}_InvMassBin{}",
                "hNSYieldBinCount_PtBinAssoc{}_InvMassBin{}",
                "hASYieldBinCount_PtBinAssoc{}_InvMassBin{}",
            ]:
                h_name = name_template.format(i_pt_had + 1, i_mass_out + 1)
                h_pl = ROOT.TH1D(h_name, "", n_pt_cand, pt_cand_edges_arr)
                for i_bin in range(1, n_pt_cand + 1):
                    h_pl.SetBinContent(i_bin, src.GetBinContent(i_bin))
                    h_pl.SetBinError(i_bin, src.GetBinError(i_bin))
                h_pl.SetDirectory(out_file)
                out_file.cd()
                h_pl.Write()
                del h_pl

    out_file.Close()
    print("[INFO] Done!")
    print(f"[INFO] Final plots saved to: {final_plot_path}")


# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit azimuthal correlations (rewrite of FitCorrel.C)")
    parser.add_argument("config",
                        nargs="?",
                        default="config_CorrAnalysis_v2_010_negDeta.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    fit_correl(args.config)
