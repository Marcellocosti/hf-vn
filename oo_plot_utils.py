import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import ROOT
import array
import sys
import os
import re
import subprocess
from ROOT import TFile, gROOT, TGaxis, gStyle


def get_langevin_prediction(filename):
    """
    Get the Langevin prediction from a .dat file.

    Args:
        filename: Path to the .dat file.

    Returns:
        ROOT.TGraph: Graph containing the Langevin prediction.
    """
    data = pd.read_csv(filename, delim_whitespace=True, header=None)
    x = data[0].values
    y = data[1].values
    graph = ROOT.TGraph(len(x), x, y)
    return graph

def get_katz_prediction(filename):
    """
    Get the Katz prediction from a .dat file.

    Args:
        filename: Path to the .dat file.

    Returns:
        ROOT.TGraph: Graph containing the Katz prediction.
    """
    data = pd.read_csv(filename, delim_whitespace=True, header=None)
    x = data[0].values
    y = data[1].values
    graph = ROOT.TGraph(len(x), x, y)
    return graph

def get_empty_marker_style(marker_style):
    """
    Get an empty marker style based on the original marker style.
    
    Args:
        marker_style: Original marker style (ROOT constant)
    Returns:
        New marker style with fillstyle set to 0 (empty)
    """
    # Create a new marker style with the same shape but empty fill
    if marker_style == ROOT.kFullCircle:
        return ROOT.kOpenCircle
    elif marker_style == ROOT.kFullSquare:
        return ROOT.kOpenSquare
    elif marker_style == ROOT.kFullTriangleUp:
        return ROOT.kOpenTriangleUp
    elif marker_style == ROOT.kFullTriangleDown:
        return ROOT.kOpenTriangleDown
    elif marker_style == ROOT.kFullDiamond:
        return ROOT.kOpenDiamond
    elif marker_style == ROOT.kFullCross:
        return ROOT.kOpenCross
    elif marker_style == ROOT.kFullStar:
        return ROOT.kOpenStar
    elif marker_style == ROOT.kFullCrossX:
        return ROOT.kOpenCrossX
    elif marker_style == ROOT.kFullDoubleDiamond:
        return ROOT.kOpenDoubleDiamond
    else:
        # Default to open circle if unrecognized
        print(f"Warning: Unrecognized marker style {marker_style}, defaulting to open circle")
        return ROOT.kOpenCircle

def create_empty_clone(hist):
    """
    Create an empty clone of the histogram/graph for error representation.
    
    Args:
        hist: List of histograms/graphs (e.g. [stat_graph, syst_graph])
    Returns:
        List of histograms/graphs with an empty clone added (e.g. [stat_graph, syst_graph, empty_clone])
    """
    
    stat_hist = hist[0] if isinstance(hist, list) else hist
    empty_clone = stat_hist.Clone(stat_hist.GetName() + "_empty")
    empty_clone.SetTitle("")
    marker_style = get_empty_marker_style(stat_hist.GetMarkerStyle())
    SetObjectStyle(empty_clone, markerstyle=marker_style, fillstyle=0, markercolor=ROOT.kBlack, markersize=stat_hist.GetMarkerSize(), linecolor=stat_hist.GetLineColor(), linewidth=stat_hist.GetLineWidth())
    
    if isinstance(hist, list):
        hist.append(empty_clone)
        return hist
    else:
        return empty_clone

def get_root_constant(name):
    """
    Get ROOT constant with support for arithmetic expressions (e.g. "kGray+2")
    
    Args:
        name: Constant name (str) or non-string type
        
    Returns:
        Calculated constant value
    """
    if not isinstance(name, str):
        return name

    # Parse constant name and expression
    match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)([+\-*/%].*)?$', name.strip())
    if not match:
        raise ValueError(f"Invalid format: {name}")

    const_name, expr = match.groups()
    base_val = getattr(ROOT, const_name)
    
    # Return base value or calculate expression
    return base_val if not expr else eval(f"{base_val}{expr}", {}, {})

def LoadGraphAndSyst(path_to_file, graph_name, syst_name, color, marker, markersize=2, alpha=1,
                      syst_fd=False):
    """
    Loads graph and systematic uncertainties from a ROOT file.

    Args:
        path_to_file (str): Path to ROOT file.
        graph_name (str): Name of the graph.
        syst_name (str): Name of systematic uncertainty graph.
        syst_fd (str): Additional systematic uncertainty graph.
        color (int): Color of the graph.
        marker (int): Marker style.
        markersize (float): Marker size.

    Returns:
        tuple: (graph, syst, syst_fd) if both systematic uncertainties are provided,
               (graph, syst) if only one is provided, or (graph) otherwise.
    """
    infile = ROOT.TFile.Open(path_to_file)
    graph = infile.Get(graph_name)
    color = get_root_constant(color)
    color = ROOT.TColor.GetColorTransparent(color, alpha)
    marker = get_root_constant(marker)

    SetObjectStyle(graph, markerstyle=marker, markercolor=color, markersize=markersize, linecolor=color, linewidth=2)

    if syst_name:
        gsyst = infile.Get(syst_name)
        SetObjectStyle(gsyst, markerstyle=marker, markercolor=color, markersize=markersize, linecolor=color, linewidth=2, fillalpha=0, fillstyle=0)
        gsyst = SystX(graph, gsyst)

        if syst_fd:
            gsyst_fd = infile.Get(syst_fd)
            SetObjectStyle(gsyst_fd, markerstyle=marker, markercolor=color, markersize=markersize, linecolor=color, linewidth=1, fillcolor=color, fillalpha=0.8, fillstyle=3145)
            gsyst_fd = SystX(graph, gsyst_fd, 0.1)
            return graph, gsyst, gsyst_fd

        return graph, gsyst
    return graph

def SetGlobalStyle(**kwargs):
    '''
    Method to set global style.

    Parameters
    ----------

    - padrightmargin (float), default = 0.035
    - padleftmargin (float), default = 0.12
    - padtopmargin (float), default = 0.035
    - padbottommargin (float), default = 0.1

    - titlesize (float), default = 0.050
    - titlesizex (float), default = 0.050
    - titlesizey (float), default = 0.050
    - titlesizez (float), default = 0.050

    - labelsize (float), default = 0.045
    - labelsizex (float), default = 0.045
    - labelsizey (float), default = 0.045
    - labelsizez (float), default = 0.045

    - titleoffset (float), default = 1.2
    - titleoffsetx (float), default = 1.2
    - titleoffsey (float), default = 1.2
    - titleoffsetz (float), default = 1.2

    - opttitle (int), default = 0
    - optstat (int), default = 0

    - padtickx (int), default = 1
    - padticky (int), default = 1

    - maxdigits (int), default no max value

    - palette (int), default kBird
    '''

    # pad margins
    if 'padrightmargin' in kwargs:
        gStyle.SetPadRightMargin(kwargs['padrightmargin'])
    else:
        gStyle.SetPadRightMargin(0.035)

    if 'padleftmargin' in kwargs:
        gStyle.SetPadLeftMargin(kwargs['padleftmargin'])
    else:
        gStyle.SetPadLeftMargin(0.12)

    if 'padtopmargin' in kwargs:
        gStyle.SetPadTopMargin(kwargs['padtopmargin'])
    else:
        gStyle.SetPadTopMargin(0.035)

    if 'padbottommargin' in kwargs:
        gStyle.SetPadBottomMargin(kwargs['padbottommargin'])
    else:
        gStyle.SetPadBottomMargin(0.1)

    # title sizes
    if 'titlesize' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesize'], 'xyz')
    else:
        gStyle.SetTitleSize(0.050, 'xyz')

    if 'titlesizex' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'x')
    if 'titlesizey' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'y')
    if 'titlesizez' in kwargs:
        gStyle.SetTitleSize(kwargs['titlesizex'], 'z')

    # label sizes
    if 'labelsize' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsize'], 'xyz')
    else:
        gStyle.SetLabelSize(0.045, 'xyz')

    if 'labelsizex' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizex'], 'x')
    if 'labelsizey' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizey'], 'y')
    if 'labelsizez' in kwargs:
        gStyle.SetLabelSize(kwargs['labelsizez'], 'z')

    # title offsets
    if 'titleoffset' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffset'], 'xyz')
    else:
        gStyle.SetTitleOffset(1.2, 'xyz')

    if 'titleoffsetx' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsetx'], 'x')
    if 'titleoffsety' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsety'], 'y')
    if 'titleoffsetz' in kwargs:
        gStyle.SetTitleOffset(kwargs['titleoffsetz'], 'z')

    # other options
    if 'opttitle' in kwargs:
        gStyle.SetOptTitle(kwargs['opttitle'])
    else:
        gStyle.SetOptTitle(0)

    if 'optstat' in kwargs:
        gStyle.SetOptStat(kwargs['optstat'])
    else:
        gStyle.SetOptStat(0)

    if 'padtickx' in kwargs:
        gStyle.SetPadTickX(kwargs['padtickx'])
    else:
        gStyle.SetPadTickX(1)

    if 'padticky' in kwargs:
        gStyle.SetPadTickY(kwargs['padticky'])
    else:
        gStyle.SetPadTickY(1)

    gStyle.SetLegendBorderSize(0)

    if 'maxdigits' in kwargs:
        TGaxis.SetMaxDigits(kwargs['maxdigits'])

    if 'palette' in kwargs:
        gStyle.SetPalette(kwargs['palette'])

    gROOT.ForceStyle()


def SetObjectStyle(obj, **kwargs):
    '''
    Method to set root object style.

    Parameters
    ----------

    - obj: object to set style

    - linecolor (int) default 1 (black)
    - linealpha (float) default 1
    - linewidth (int) default 2
    - linestyle (int) default 1

    - markercolor (int) default 1 (black)
    - markeralpha (float) default 1
    - markerstyle (int) default 20 (full circle)
    - markersize (int) default 20 (full circle)

    - fillcolor (int) default no filling
    - fillalpha (float) default 1
    - fillstyle (int) default 0 (no style)

    - color (int) sets same color for line, marker and fill
    - alpha (float) sets same alpha for line, marker and fill
    '''

    # alpha parameters
    lalpha = kwargs.get('linealpha', 1)
    malpha = kwargs.get('markeralpha', 1)
    falpha = kwargs.get('fillalpha', 1)
    if 'alpha' in kwargs:
        lalpha = kwargs['alpha']
        malpha = kwargs['alpha']
        falpha = kwargs['alpha']
    if 'linealpha' in kwargs:
        lalpha = kwargs['linealpha']
    if 'markeralpha' in kwargs:
        malpha = kwargs['markeralpha']
    if 'fillalpha' in kwargs:
        falpha = kwargs['fillalpha']

    # line styles
    if 'linecolor' in kwargs:
        if lalpha < 1:
            obj.SetLineColorAlpha(kwargs['linecolor'], lalpha)
        else:
            obj.SetLineColor(kwargs['linecolor'])
    else:
        if lalpha < 1:
            obj.SetLineColorAlpha(1, lalpha)
        else:
            obj.SetLineColor(1)

    if 'linewidth' in kwargs:
        obj.SetLineWidth(kwargs['linewidth'])
    else:
        obj.SetLineWidth(2)

    if 'linestyle' in kwargs:
        obj.SetLineStyle(kwargs['linestyle'])
    else:
        obj.SetLineStyle(1)

    # marker styles
    if 'markercolor' in kwargs:
        if malpha < 1:
            obj.SetMarkerColorAlpha(kwargs['markercolor'], malpha)
        else:
            obj.SetMarkerColor(kwargs['markercolor'])
    else:
        if malpha < 1:
            obj.SetMarkerColorAlpha(1, malpha)
        else:
            obj.SetMarkerColor(1)

    if 'markersize' in kwargs:
        obj.SetMarkerSize(kwargs['markersize'])
    else:
        obj.SetMarkerSize(1)

    if 'markerstyle' in kwargs:
        obj.SetMarkerStyle(kwargs['markerstyle'])
    else:
        obj.SetMarkerStyle(20)

    # fill styles
    if 'fillcolor' in kwargs:
        if falpha < 1:
            obj.SetFillColorAlpha(kwargs['fillcolor'], falpha)
        else:
            obj.SetFillColor(kwargs['fillcolor'])

    if 'fillstyle' in kwargs:
        obj.SetFillStyle(kwargs['fillstyle'])

    # global color
    if 'color' in kwargs:
        if lalpha < 1:
            obj.SetLineColorAlpha(kwargs['color'], lalpha)
        else:
            obj.SetLineColor(kwargs['color'])
        if malpha < 1:
            obj.SetMarkerColorAlpha(kwargs['color'], malpha)
        else:
            obj.SetMarkerColor(kwargs['color'])
        if falpha < 1:
            obj.SetFillColorAlpha(kwargs['color'], falpha)
        else:
            obj.SetFillColor(kwargs['color'])


def GetInvMassHistAndFit(infile, ptmin, ptmax, nbin, hasfeflections=False):
    print(f"Loading invariant mass histogram and fit from file: {infile}")
    inFile = ROOT.TFile.Open(infile)

    print(f"Looking for canvas: cSimFit_pt{ptmin}_{ptmax}")
    cMassVsV2 = inFile.Get(f'cSimFit_pt{ptmin}_{ptmax}')
    hInvMass = cMassVsV2.GetPad(1).GetListOfPrimitives().FindObject(f'MassForFit{nbin}')
    fMassTot = cMassVsV2.GetPad(1).GetListOfPrimitives().FindObject('fMassTotFunc')
    fMassBkg = cMassVsV2.GetPad(1).GetListOfPrimitives().FindObject('fMassBkgFunc')
    if hasfeflections:
        fMassRefl = cMassVsV2.GetPad(1).GetListOfPrimitives().FindObject('fMassRflFunc')
        SetObjectStyle(fMassRefl, linewidth=1, color=ROOT.kGreen+2, fillalpha=0.3, fillstyle=1001)
    hV2VsMass = cMassVsV2.GetPad(2).GetListOfPrimitives().FindObject('hDummy')
    fV2Tot = cMassVsV2.GetPad(2).GetListOfPrimitives().FindObject('fVnTotFunc')
    fV2Bkg = cMassVsV2.GetPad(2).GetListOfPrimitives().FindObject('fVnBkgFunc')

    SetObjectStyle(hInvMass, color=ROOT.kBlack)
    SetObjectStyle(fMassTot, linewidth=3, linecolor=ROOT.kAzure+4)
    SetObjectStyle(fMassBkg, linewidth=3, linecolor=ROOT.kRed-4, linestyle=2)
    SetObjectStyle(hV2VsMass, color=ROOT.kBlack)
    SetObjectStyle(fV2Tot, linewidth=3, linecolor=ROOT.kAzure+4)
    SetObjectStyle(fV2Bkg, linewidth=3, linecolor=ROOT.kRed-4, linestyle=2)

    fMassTot.SetNpx(1000)
    fMassBkg.SetNpx(1000)
    fV2Tot.SetNpx(1000)
    fV2Bkg.SetNpx(1000)

    if hasfeflections:
        return hInvMass, fMassTot, fMassBkg, hV2VsMass, fV2Tot, fV2Bkg, fMassRefl
    else:
        return hInvMass, fMassTot, fMassBkg, hV2VsMass, fV2Tot, fV2Bkg


def GetV2HistAndFit(infile, dir, ptmin, ptmax, nbin, hasfeflections=False):
    inFile = ROOT.TFile.Open(infile)
    if dir:
        cMassVsV2 = inFile.Get(f'{dir}/cFrac_{ptmin}_{ptmax}')
    else:
        cMassVsV2 = inFile.Get(f'cFrac_{ptmin}_{ptmax}')
    gv2 = cMassVsV2.GetListOfPrimitives().FindObject('Graph')
    hV2VsFrac = cMassVsV2.GetListOfPrimitives().FindObject(f'hV2VsFrac_{nbin}')
    tf1 = gv2.GetFunction("linear")

    SetObjectStyle(gv2, color=ROOT.kBlack, markersize=1.2, linewidth=1)
    SetObjectStyle(hV2VsFrac, linewidth=1, linecolor=ROOT.kAzure+4, markersize=0, alpha=0.2)
    SetObjectStyle(tf1, linewidth=1, linecolor=ROOT.kRed-4, markersize=0, linestyle=9)

    return gv2, hV2VsFrac, tf1


def HepDataHandeler(path, table, histonumber, **kwargs):
    '''
    Helper function to read the HepData table and extract the data.

    Parameters
    ----------
    - path: path to the HepData file
    - table: table number
    - histonumber: histogram index
    - kwargs:
        - stat: statistical uncertainty suffix (default: 'e1')
        - syst: systematic uncertainty suffix (default: 'e2')
        - syst_min: asymmetric lower systematic suffix
        - syst_max: asymmetric upper systematic suffix

    Returns
    ----------
    - graph (stat errors), graph_syst (syst errors)
    '''
    file = ROOT.TFile.Open(path)
    histo_name = f'{table}/Hist1D_y{histonumber}'
    graph_name = f'{table}/Graph1D_y{histonumber}'

    histo = file.Get(f'{histo_name}')
    histo.SetDirectory(0)

    if kwargs.get('stat'):
        histo_unc = file.Get(f'{histo_name}_{kwargs.get("stat")}')

    if kwargs.get('syst'):
        histo_syst = file.Get(f'{histo_name}_{kwargs.get("syst")}')
    else:
        histo_syst = file.Get(f'{histo_name}_e2minus')

    if kwargs.get('syst_min'):
        histo_syst_min = file.Get(f'{histo_name}_{kwargs.get("syst_min")}')
    else:
        histo_syst_min = file.Get(f'{histo_name}_e2minus')

    if kwargs.get('syst_max'):
        histo_syst_max = file.Get(f'{histo_name}_{kwargs.get("syst_max")}')
    else:
        histo_syst_max = file.Get(f'{histo_name}_e2plus')

    graph_syst = file.Get(graph_name)
    graph = graph_syst.Clone('graph')

    for i in range(1, histo.GetNbinsX()+1):
        if kwargs.get('stat'):
            graph.SetPointEYlow(i-1, histo_unc.GetBinContent(i))
            graph.SetPointEYhigh(i-1, histo_unc.GetBinContent(i))

        if kwargs.get('syst_min'):
            graph_syst.SetPointEYlow(i, np.abs(histo_syst_min.GetBinContent(i)))
        else:
            graph_syst.SetPointEYlow(i-1, np.abs(histo_syst.GetBinContent(i)))

        if kwargs.get('syst_max'):
            graph_syst.SetPointEYhigh(i, np.abs(histo_syst_max.GetBinContent(i)))
        else:
            graph_syst.SetPointEYhigh(i-1, np.abs(histo_syst.GetBinContent(i)))

    graph_syst = SystX(graph, graph_syst)
    SetObjectStyle(graph_syst, fillalpha=0, fillstyle=1001, fillcolor=ROOT.kWhite)

    return graph, graph_syst


def GetJPsiGraph(filename, color, marker):
    x_vals, ex_vals, y_vals, stat_errs, syst_errs = [], [], [], [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                x_min, x_max, val, stat, syst = map(float, parts)
            except ValueError:
                continue

            x = (x_min + x_max) / 2
            ex = (x_max - x_min) / 2
            x_vals.append(x)
            ex_vals.append(ex)
            y_vals.append(val)
            stat_errs.append(stat)
            syst_errs.append(syst)

    graph_stat = ROOT.TGraphAsymmErrors(len(x_vals))
    graph_syst = ROOT.TGraphAsymmErrors(len(x_vals))

    for i in range(len(x_vals)):
        graph_stat.SetPoint(i, x_vals[i], y_vals[i])
        graph_stat.SetPointEXlow(i, ex_vals[i])
        graph_stat.SetPointEXhigh(i, ex_vals[i])
        graph_stat.SetPointEYlow(i, stat_errs[i])
        graph_stat.SetPointEYhigh(i, stat_errs[i])

        graph_syst.SetPoint(i, x_vals[i], y_vals[i])
        graph_syst.SetPointEXlow(i, ex_vals[i])
        graph_syst.SetPointEXhigh(i, ex_vals[i])
        graph_syst.SetPointEYlow(i, syst_errs[i])
        graph_syst.SetPointEYhigh(i, syst_errs[i])

    SetObjectStyle(graph_stat, color=color, markerstyle=marker, fillalpha=0)
    SetObjectStyle(graph_syst, linecolor=color, markercolor=color, markerstyle=marker, fillalpha=0, fillcolor=ROOT.kWhite, fillstyle=0)
    graph_syst = SystX(graph_stat, graph_syst)
    return graph_stat, graph_syst


def GetDeuGraph(filename, color, marker):
    x_vals, y_vals, stat_low_errs, stat_high_errs, syst_low_errs, syst_high_errs = [], [], [], [], [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                print('por')
                continue
            try:
                xcent, val, statlow, stathigh, systlow, systhigh = map(float, parts)
            except ValueError:
                print('codio')
                continue

            x_vals.append(xcent)
            y_vals.append(val)
            stat_low_errs.append(statlow)
            stat_high_errs.append(statlow)
            syst_low_errs.append(systlow)
            syst_high_errs.append(systlow)

    graph_stat = ROOT.TGraphAsymmErrors(len(x_vals))
    graph_syst = ROOT.TGraphAsymmErrors(len(x_vals))

    print(x_vals, y_vals)
    for i in range(len(x_vals)):
        graph_stat.SetPoint(i, x_vals[i], y_vals[i])
        graph_stat.SetPointEXlow(i, 0.2)
        graph_stat.SetPointEXhigh(i, 0.2)
        graph_stat.SetPointEYlow(i, stat_low_errs[i])
        graph_stat.SetPointEYhigh(i, stat_high_errs[i])

        graph_syst.SetPoint(i, x_vals[i], y_vals[i])
        graph_syst.SetPointEXlow(i, 0.2)
        graph_syst.SetPointEXhigh(i, 0.2)
        graph_syst.SetPointEYlow(i, syst_low_errs[i])
        graph_syst.SetPointEYhigh(i, syst_high_errs[i])

    SetObjectStyle(graph_stat, color=color, markerstyle=marker)
    SetObjectStyle(graph_syst, linecolor=color, markercolor=color, markerstyle=marker, fillalpha=0, fillstyle=1001)
    graph_syst = SystX(graph_stat, graph_syst)

    return graph_stat, graph_syst


def scale_ncq(graph, ncq):
    """
    Scale both pt (x) and v2 (y) of a TGraphErrors by the number of constituent quarks.

    Args:
        graph (ROOT.TGraphErrors): Input graph with stat errors.
        ncq (int): Number of constituent quarks.

    Returns:
        ROOT.TGraphErrors: New graph with x -> x/ncq, y -> y/ncq, errors scaled accordingly.
    """
    gscaled = ROOT.TGraphErrors()
    for i in range(graph.GetN()):
        gscaled.AddPoint(graph.GetX()[i] / ncq, graph.GetY()[i] / ncq)
        gscaled.SetPointError(i, graph.GetErrorX(i) / ncq, graph.GetErrorY(i) / ncq)
    return gscaled


def scale_ncq_hist(hist, ncq):
    """
    Convert a TH1 histogram to a TGraphErrors scaled by the number of constituent quarks.
    x -> bin_center/ncq, y -> bin_content/ncq, x_err -> 0.2 * bin_width / ncq.

    Args:
        hist (ROOT.TH1): Input histogram.
        ncq (int): Number of constituent quarks.

    Returns:
        ROOT.TGraphErrors: Scaled graph.
    """
    graph = ROOT.TGraphErrors()
    for i in range(hist.GetXaxis().GetNbins()):
        x = hist.GetXaxis().GetBinCenter(i + 1)
        y = hist.GetBinContent(i + 1)
        ex = 0.2 * hist.GetXaxis().GetBinWidth(i + 1)
        ey = hist.GetBinError(i + 1)
        graph.SetPoint(i, x / ncq, y / ncq)
        graph.SetPointError(i, ex / ncq, ey / ncq)
    return graph


def GetCanvas3sub(name, axisname):
    """
    Creates a canvas with three adjacent subpads sharing the y-axis.

    Args:
        name (str): Name of the canvas.
        axisname (str): Axis title string.

    Returns:
        tuple: (canvas, frames)
    """
    canvas = ROOT.TCanvas(name, name, 1600, 600)
    canvas.Divide(3, 1, 0, 0)

    frames = []
    for i in range(3):
        canvas.cd(i + 1)
        pad = ROOT.gPad
        pad.SetLeftMargin(0.18 if i == 0 else 0.0)
        pad.SetRightMargin(0.05 if i == 2 else 0.0)

        frame = pad.DrawFrame(-0.5, -0.20, 40, 0.62, axisname)
        frame.SetTitle("")

        if i > 0:
            frame.GetYaxis().SetLabelSize(0)
            frame.GetYaxis().SetTickLength(0.040)
            frame.GetXaxis().SetTickLength(0.025)
        else:
            frame.GetYaxis().SetTitleSize(0.05)
            frame.GetYaxis().SetTitleOffset(1.6)
            frame.GetYaxis().SetDecimals()

        frames.append(frame)

    return canvas, frames


def PlotEmptyClone(graph, leg):
    """
    Creates an empty clone of a graph with adjusted marker style for legend entry.

    Args:
        graph (ROOT.TGraph): The input graph.
        leg (ROOT.TLegend): Legend to add the entry to.
    """
    clone = create_empty_clone(graph)
    clone.Draw('PZ same')
    if leg:
        leg.AddEntry(clone, leg.GetEntry(leg.GetNRows()).GetLabel(), 'P')


def DrawStatSystEmpty(graph, gsyst, gfd, leg):
    gsyst.Draw('5 same')
    if gfd:
        gfd.Draw('5 same')
    graph.Draw('PZ same')
    PlotEmptyClone(graph, leg)


def SaveCanvas(canv, title, suffix='', formats=['pdf']):
    """
    Saves canvas in multiple formats.

    Args:
        canv (ROOT.TCanvas): Canvas to save.
        title (str): Output file name.
        suffix (str): Additional suffix.
        formats (tuple): File formats.
    """
    for form in formats:
        canv.SaveAs(f'{title}{suffix}.{form}')


def GetCanvas(name, axisname, xmin=0.4, xmax=40, ymin=-0.20, ymax=0.62, setlogx=True):
    """
    Creates a ROOT canvas with optional log-x scale and a formatted frame.

    Args:
        name (str): Name of the canvas.
        axisname (str): Axis title string.
        xmin, xmax, ymin, ymax (float): Frame range.
        setlogx (bool): Enable log-x scale.

    Returns:
        tuple: (canvas, frame)
    """
    canv = ROOT.TCanvas(name, name, 800, 800)
    if setlogx:
        canv.SetLogx()

    hframe = canv.DrawFrame(xmin, ymin, xmax, ymax, axisname)
    hframe.GetYaxis().SetDecimals()
    hframe.GetYaxis().SetTitleOffset(1.6)
    hframe.GetXaxis().SetMoreLogLabels()

    return canv, hframe


def GetCanvas4sub(name, xmins, xmaxs, ymins_mass, ymaxs_mass, ymins_v2, ymaxs_v2, axisnametop, axisnamebottom):
    """
    Creates a 2x2 canvas: mass (top-left), raw yield (top-right),
    v2 (bottom-left), v2 vs FD fraction (bottom-right).

    Returns:
        tuple: (canvas, frames)
    """
    canvas = ROOT.TCanvas(name, name, 1200, 1100)
    canvas.Divide(2, 2)

    frames = []
    for i in range(4):
        canvas.cd(i + 1)
        pad = ROOT.gPad
        if i == 0:
            frame = pad.DrawFrame(xmins, ymins_mass, xmaxs, ymaxs_mass, axisnametop)
        elif i == 1:
            frame = pad.DrawFrame(0.5, 0, 20.5, 20000, ';Minimum BDT score for prompt #Lambda_{c}^{+}; raw yield')
        elif i == 2:
            frame = pad.DrawFrame(xmins, ymins_v2, xmaxs, ymaxs_v2, axisnamebottom)
        elif i == 3:
            frame = pad.DrawFrame(0, 0, 1.05, 0.3, ';Non-prompt fraction; #it{v}_{2}^{obs.}{SP, |#Delta#it{#eta}| > 1.3}')
        frame.SetTitle("")
        frames.append(frame)

    return canvas, frames


def GetCanvas2sub(name, xmins, xmaxs, ymins_mass, ymaxs_mass, ymins_v2, ymaxs_v2, axisnameleft, axisnameright, n_divisions):
    """
    Creates a canvas with 2 vertical subpads (top and bottom) sharing x-axis.

    Returns:
        tuple: (canvas, frames)
    """
    canvas = ROOT.TCanvas(name, name, 600, 1000)
    canvas.Divide(1, 2, 0.0, 0.0)
    frames = []
    for i in range(2):
        canvas.cd(i + 1)
        pad = ROOT.gPad
        if i == 0:
            canvas.cd(i + 1).SetTopMargin(0.08)
            frame = pad.DrawFrame(xmins, ymins_mass, xmaxs, ymaxs_mass, axisnameleft)
            frame.SetTitle("")
            #frame.GetYaxis().SetTickLength(0.040)
            frame.GetYaxis().SetDecimals()
            frame.GetYaxis().SetTitleSize(0.05)
            frame.GetYaxis().SetTitleOffset(1.6)
            frame.GetYaxis().SetLabelSize(0.058)

            frame.GetYaxis().SetMaxDigits(3)
            frame.GetXaxis().SetTickLength(0.025)
        elif i == 1:
            frame = pad.DrawFrame(xmins, ymins_v2, xmaxs, ymaxs_v2, axisnameright)
            frame.GetYaxis().SetDecimals()
            if n_divisions != -1:
                frame.GetXaxis().SetNdivisions(n_divisions) #, False)   # control label density
                # frame.GetXaxis().SetLabelSize(0.045)  # smaller labels
                # frame.GetXaxis().SetTitleSize(0.05)
                # frame.GetXaxis().SetTitleOffset(1.0)

        frame.SetTitle("")
        frames.append(frame)

    return canvas, frames


def GetLegend(xmin=0.19, ymin=0.62, xmax=0.75, ymax=0.77, textsize=0.04, ncolumns=2, header=' ', fillstyle=0):
    """
    Creates a formatted legend.

    Args:
        xmin, ymin, xmax, ymax (float): Legend position (NDC).
        textsize (float): Text size.
        ncolumns (int): Number of columns.
        header (str): Header text.
        fillstyle (int): Fill style.

    Returns:
        ROOT.TLegend: Configured legend.
    """
    leg = ROOT.TLegend(xmin, ymin, xmax, ymax)
    leg.SetTextSize(textsize)
    leg.SetNColumns(ncolumns)
    leg.SetFillStyle(fillstyle)
    leg.SetHeader(header)
    return leg


def DrawLineAt0(min, max, title=False):
    line = ROOT.TLine(min, 0, max, 0)
    line.SetLineStyle(9)
    line.SetLineWidth(2)
    line.SetLineColor(ROOT.kGray+2)
    return line


def SystX(graph, syst, percentage=0.2):
    if isinstance(graph, ROOT.TGraphAsymmErrors) and isinstance(syst, ROOT.TGraphAsymmErrors):
        for i in range(graph.GetN()):
            stat = 2 * graph.GetErrorXlow(i)
            syst.SetPointEXlow(i, stat * percentage)
            syst.SetPointEXhigh(i, stat * percentage)
    
    if isinstance(graph, ROOT.TGraphErrors) and isinstance(syst, ROOT.TGraphAsymmErrors):
        for i in range(graph.GetN()):
            stat = 2 * graph.GetErrorX(i)
            syst.SetPointEXlow(i, stat * percentage)
            syst.SetPointEXhigh(i, stat * percentage)

    if isinstance(graph, ROOT.TGraphAsymmErrors) and isinstance(syst, ROOT.TGraphErrors):
        for i in range(graph.GetN()):
            stat = 2 * graph.GetErrorXlow(i)
            syst.SetPointError(i, stat * percentage, stat * percentage)

    return syst


def GetPrediction(file, xmin, xmax, scale, isUnc, model='tamu'):
    if model == 'tamu':
        splinev2, _, _, _ = ReadTAMUv2(file)
        gpred = ROOT.TGraphAsymmErrors(1)
        for ipt in range(xmin, xmax):
            index = ipt
            ipt *= scale

            gpred.AddPoint(ipt, splinev2['yCent'](ipt))
            if isUnc:
                gpred.SetPointError(index, 0., 0.,
                                    float(splinev2['yCent'](ipt) - splinev2['yMin'](ipt)),
                                    float(splinev2['yMax'](ipt) - splinev2['yCent'](ipt)))
    elif model == 'lido':
        splinev2, _, _, _ = ReadLIDOV2(file)
        gpred = ROOT.TGraphAsymmErrors(1)
        for ipt in range(xmin, xmax):
            ipt *= scale

            gpred.AddPoint(ipt, splinev2['yCent'](ipt))
            if isUnc:
                gpred.SetPointError(ipt, 0., 0.,
                                    float(splinev2['yCent'](ipt) - splinev2['yMin'](ipt)),
                                    float(splinev2['yMax'](ipt) - splinev2['yCent'](ipt)))
    elif model == 'lgr':
        splinev2, _, _, _ = ReadLGRV2(file)
        gpred = ROOT.TGraphAsymmErrors(0)
        index = 0
        for ipt in range(xmin, xmax):
            ipt *= scale

            gpred.AddPoint(ipt, splinev2['yCent'](ipt))
            if isUnc:
                index += 1
                gpred.SetPointError(index, 0., 0.,
                                    float(splinev2['yCent'](ipt) - splinev2['yMin'](ipt)),
                                    float(splinev2['yMax'](ipt) - splinev2['yCent'](ipt)))
    elif model == 'phsd':
        splinePHSD3040, splinePHSD4050, splinePHSD6080, _, _, _ = ReadPHSDV2(file)
        gpred3040 = ROOT.TGraphAsymmErrors(0)
        index = 0
        for ipt in range(xmin, xmax):
            ipt *= scale
            gpred3040.AddPoint(ipt, splinePHSD3040['yCent'](ipt))
            if isUnc:
                index += 1
                gpred3040.SetPointError(index, 0., 0.,
                                        float(splinePHSD3040['yCent'](ipt) - splinePHSD3040['yMin'](ipt)),
                                        float(splinePHSD3040['yMax'](ipt) - splinePHSD3040['yCent'](ipt)))
        gpred3040.RemovePoint(0)
        gpred3040.RemovePoint(0)

        gpred4050 = ROOT.TGraphAsymmErrors(0)
        index = 0
        for ipt in range(xmin, xmax):
            ipt *= scale
            gpred4050.AddPoint(ipt, splinePHSD4050['yCent'](ipt))
            if isUnc:
                index += 1
                gpred4050.SetPointError(index, 0., 0.,
                                        float(splinePHSD4050['yCent'](ipt) - splinePHSD4050['yMin'](ipt)),
                                        float(splinePHSD4050['yMax'](ipt) - splinePHSD4050['yCent'](ipt)))
        gpred4050.RemovePoint(0)
        gpred4050.RemovePoint(0)

        gpred6080 = ROOT.TGraphAsymmErrors(0)
        index = 0
        for ipt in range(xmin, xmax):
            ipt *= scale
            gpred6080.AddPoint(ipt, splinePHSD6080['yCent'](ipt))
            if isUnc:
                index += 1
                gpred6080.SetPointError(index, 0., 0.,
                                        float(splinePHSD6080['yCent'](ipt) - splinePHSD6080['yMin'](ipt)),
                                        float(splinePHSD6080['yMax'](ipt) - splinePHSD6080['yCent'](ipt)))
        gpred6080.RemovePoint(0)
        gpred6080.RemovePoint(0)

        return gpred3040, gpred4050, gpred6080
    return gpred


def compute_central_graph(graph_upper, graph_lower):
    """
    Given two TGraphAsymmErrors (upper and lower limits), compute the central values
    and asymmetric uncertainties.

    Returns:
        ROOT.TGraphAsymmErrors: Graph with central values and uncertainties.
    """
    x_upper = np.array([graph_upper.GetPointX(i) for i in range(graph_upper.GetN())])
    y_upper = np.array([graph_upper.GetPointY(i) for i in range(graph_upper.GetN())])
    x_lower = np.array([graph_lower.GetPointX(i) for i in range(graph_lower.GetN())])
    y_lower = np.array([graph_lower.GetPointY(i) for i in range(graph_lower.GetN())])

    x_common = np.sort(np.unique(np.concatenate((x_upper, x_lower))))
    y_upper_interp = np.interp(x_common, x_upper, y_upper, left=y_upper[0], right=y_upper[-1])
    y_lower_interp = np.interp(x_common, x_lower, y_lower, left=y_lower[0], right=y_lower[-1])

    y_central = 0.5 * (y_upper_interp + y_lower_interp)
    y_err_low = np.abs(y_central - y_lower_interp)
    y_err_high = np.abs(y_upper_interp - y_central)

    graph_central = ROOT.TGraphAsymmErrors(len(x_common))
    for i, x in enumerate(x_common):
        graph_central.SetPoint(i, x, y_central[i])
        graph_central.SetPointError(i, 0, 0, y_err_low[i], y_err_high[i])

    return graph_central


def ReadPHSDV2(fileName):
    '''Read PHSD v2 predictions from file. Returns splines for 30-40, 40-50, 60-80 centralities.'''
    dfPHSD = pd.read_csv(fileName, sep=' ', comment='#')
    dfPHSD['3040_min'] = dfPHSD['3040'] - dfPHSD['3040_unc']
    dfPHSD['3040_max'] = dfPHSD['3040'] + dfPHSD['3040_unc']
    dfPHSD['4050_min'] = dfPHSD['4050'] - dfPHSD['4050_unc']
    dfPHSD['4050_max'] = dfPHSD['4050'] + dfPHSD['4050_unc']
    dfPHSD['6080_min'] = dfPHSD['6080'] - dfPHSD['6080_unc']
    dfPHSD['6080_max'] = dfPHSD['6080'] + dfPHSD['6080_unc']

    splinePHSD3040, ptMin, ptMax = InterpolateModel(dfPHSD['pT'], dfPHSD['3040'], dfPHSD['3040_max'], dfPHSD['3040_min'])
    splinePHSD4050, _, _ = InterpolateModel(dfPHSD['pT'], dfPHSD['4050'], dfPHSD['4050_max'], dfPHSD['4050_min'])
    splinePHSD6080, _, _ = InterpolateModel(dfPHSD['pT'], dfPHSD['6080'], dfPHSD['6080_max'], dfPHSD['6080_min'])

    return splinePHSD3040, splinePHSD4050, splinePHSD6080, dfPHSD, ptMin, ptMax


def ReadLIDOV2(fileName):
    '''Read LIDO v2 predictions from file.'''
    dfLIDO = pd.read_csv(fileName, sep=' ', comment='#')
    dfLIDO['v2_min'] = dfLIDO['v2'] - dfLIDO['v2-error']
    dfLIDO['v2_max'] = dfLIDO['v2'] + dfLIDO['v2-error']

    splineLIDO, ptMin, ptMax = InterpolateModel(dfLIDO['pT'], dfLIDO['v2'], dfLIDO['v2_max'], dfLIDO['v2_min'])

    return splineLIDO, dfLIDO, ptMin, ptMax


def ReadLGRV2(fileName):
    '''Read LGR v2 predictions from file.'''
    dfLGR = pd.read_csv(fileName, sep=' ', comment='#')
    splineLGR, ptMin, ptMax = InterpolateModel(dfLGR['pT'], dfLGR['v2'], dfLGR['v2_min'], dfLGR['v2_max'])

    return splineLGR, dfLGR, ptMin, ptMax


def InterpolateModel(ptCent, yCent, yMin=None, yMax=None):
    '''
    Interpolate model predictions using univariate splines.

    Returns:
        dict of splines {yCent, yMin, yMax}, ptMin, ptMax
    '''
    splinesAll = {}
    splinesAll['yCent'] = InterpolatedUnivariateSpline(ptCent, yCent, ext='raise', check_finite=True)

    if yMin is not None and yMin.any():
        splinesAll['yMin'] = InterpolatedUnivariateSpline(ptCent, yMin, ext='raise', check_finite=True)
    if yMax is not None and yMax.any():
        splinesAll['yMax'] = InterpolatedUnivariateSpline(ptCent, yMax, ext='raise', check_finite=True)

    return splinesAll, min(ptCent), max(ptCent)


def ReadTAMUv2(fileNameTAMUv2):
    '''Read TAMU v2 predictions from txt file.'''
    dfTAMU = pd.read_csv(fileNameTAMUv2, sep=' ', comment='#')
    if 'v2min' in dfTAMU and 'v2max' in dfTAMU:
        dfTAMU['v2'] = (dfTAMU['v2min'] + dfTAMU['v2max']) / 2
        splineTAMU, ptMin, ptMax = InterpolateModel(dfTAMU['pT'], dfTAMU['v2'],
                                                    dfTAMU['v2min'], dfTAMU['v2max'])
    else:
        splineTAMU, ptMin, ptMax = InterpolateModel(dfTAMU['pT'], dfTAMU['v2'])

    return splineTAMU, dfTAMU, ptMin, ptMax


def get_particle_info(particleName):
    '''Return particle title, mass axis title, decay string, and PDG mass.'''
    if particleName == 'Dplus':
        particleTit = 'D^{+}'
        massAxisTit = '#it{M}(K#pi#pi) (GeV/#it{c}^{2})'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(411).Mass()
        decay = 'D^{+} #rightarrow K^{#minus}#pi^{+}#pi^{+}'
    elif particleName == 'Ds':
        particleTit = 'D_{s}^{+}'
        massAxisTit = '#it{M}(KK#pi) (GeV/#it{c}^{2})'
        decay = 'D_{s}^{+} #rightarrow #phi#pi^{+} #rightarrow K^{+}K^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(431).Mass()
    elif particleName == 'LctopKpi':
        particleTit = '#Lambda_{c}^{+}'
        massAxisTit = '#it{M}(pK#pi) (GeV/#it{c}^{2})'
        decay = '#Lambda_{c}^{+} #rightarrow pK^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(4122).Mass()
    elif particleName == 'LctopK0s':
        massAxisTit = '#it{M}(pK^{0}_{s}) (GeV/#it{c}^{2})'
        decay = '#Lambda_{c}^{+} #rightarrow pK^{0}_{s}'
        massForFit = 2.25
    elif particleName == 'Dstar':
        particleTit = 'D^{*+}'
        massAxisTit = '#it{M}(K#pi#pi) - #it{M}(K#pi) (GeV/#it{c}^{2})'
        decay = 'D^{*+} #rightarrow D^{0}#pi^{+} #rightarrow K^{#minus}#pi^{+}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(413).Mass() - ROOT.TDatabasePDG.Instance().GetParticle(421).Mass()
    elif particleName == 'Dzero':
        particleTit = 'D^{0}'
        massAxisTit = '#it{M}(K#pi) (GeV/#it{c}^{2})'
        decay = 'D^{0} #rightarrow K^{#minus}#pi^{+}'
        massForFit = ROOT.TDatabasePDG.Instance().GetParticle(421).Mass()
    else:
        print(f'ERROR: particle "{particleName}" not supported. Choose: Dzero, Dplus, Ds, Dstar, Lc. Exit!')
        sys.exit()

    return particleTit, massAxisTit, decay, massForFit


def read_txt(file_path, sep=",", header=None, nrows=None):
    return pd.read_csv(file_path, sep=sep, header=header, nrows=nrows, engine='python').astype('float64')


def kEt(m, pt):
    '''kEt = sqrt(m^2 + pt^2) - m'''
    if isinstance(pt, list):
        return [np.sqrt(m**2 + ipt**2) - m for ipt in pt]
    else:
        return np.sqrt(m**2 + pt**2) - m


def nq_scaling(x, nq):
    if isinstance(x, list):
        return [ix / nq for ix in x]
    else:
        return x / nq


def fit(ptCent, yCent, med, getParams=False):
    '''Fit descending portion of v2(pT) with exponential decay to extend model to high pT.'''
    peak_idx = np.argmax(yCent)
    x_down = ptCent[peak_idx:]
    y_down = yCent[peak_idx:]

    def exp_decay(x, A, alpha):
        return A * np.exp(-alpha * x)

    p0 = [y_down[0], 0.001]
    params, cov = curve_fit(exp_decay, x_down, y_down, p0=p0)
    A_fit, alpha_fit = params
    if getParams:
        return A_fit, alpha_fit

    x_extended = np.linspace(med, 24, 100)
    y_extended = exp_decay(x_extended, A_fit, alpha_fit)
    return x_extended, y_extended


def preprocess(file_path, do_interp=False, do_fit_extend=False, catania=False, sep=" ", do_ncq=False, header=None, nrows=None):
    '''Preprocess model data: optional interpolation and exponential fit extension to high pT.'''
    df = read_txt(file_path, sep, header=header, nrows=nrows)
    if do_ncq:
        return df
    med = 11
    if not do_interp and not do_fit_extend:
        if catania:
            return df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
        return df.iloc[:, 0], df.iloc[:, 1]
    if do_interp:
        pchip = PchipInterpolator(df.iloc[:, 0], df.iloc[:, 1])
        if catania:
            pchip2 = PchipInterpolator(df.iloc[:, 0], df.iloc[:, 2])
        x_interp = np.linspace(1, max(df.iloc[:, 0]), 300)
        y_pchip = pchip(x_interp)
    if not do_fit_extend:
        if catania:
            return max(df.iloc[:, 0]), pchip, pchip2
        return max(df.iloc[:, 0]), pchip
    x_extended, y_extended = fit(x_interp, y_pchip, med, getParams=False)
    return np.concatenate((x_interp, x_extended)), np.concatenate((y_pchip, y_extended))


def preprocess_data(data_file_path, get_source_data=False, compine_syst_stat=True, header=9):
    '''Preprocess light-flavour data; weighted average to combine centrality bins.'''
    v2_index = 1
    data_columns = ['PT', 'v2', 'Stat +', 'Stat -', 'Syst +', 'Syst -']
    if header == 12:
        data_columns = ['PT', 'v2', 'stat +', 'stat -', 'sys +', 'sys -']
        v2_index = 3
    if not isinstance(data_file_path, list):
        data_file_path = list(data_file_path)
    if get_source_data:
        df = read_txt(data_file_path[0], header=header)
        return df
    df1 = read_txt(data_file_path[0], header=header)
    df2 = read_txt(data_file_path[1], header=header)
    weight1 = 1 / (df1[data_columns[2]] ** 2)
    weight2 = 1 / (df2[data_columns[2]] ** 2)
    weighted_avg = (df1.iloc[:, v2_index] * weight1 + df2.iloc[:, v2_index] * weight2) / (weight1 + weight2)
    df_new = df1.copy()
    columns = df_new.columns.tolist()
    columns[v2_index] = 'v2'
    columns[0] = 'PT [GeV/c]'
    df_new.columns = columns
    df_new['v2'] = weighted_avg.values
    if compine_syst_stat:
        df_new['Total Error'] = np.sqrt(df_new[data_columns[2]]**2 + df_new[data_columns[4]]**2)
    return df_new


def preprocess_ncq(data, do_ket_nq=False, do_pt_nq=True, ismodel=False):
    '''Scale data or model predictions by number of constituent quarks (NCQ).'''
    pdg_db = ROOT.TDatabasePDG.Instance()
    if ismodel:
        for key in data.keys():
            for subkey in data[key].keys():
                if subkey == 'lc':
                    _, _, _, mass = get_particle_info('LctopKpi')
                    nq = 3
                elif subkey == 'd0':
                    _, _, _, mass = get_particle_info('Dzero')
                    nq = 2
                for i in range(len(data[key][subkey])):
                    if do_ket_nq:
                        data[key][subkey][i].iloc[:, 0] = data[key][subkey][i].iloc[:, 0].apply(lambda x: kEt(mass, x))
                        data[key][subkey][i].iloc[:, 0] = data[key][subkey][i].iloc[:, 0].apply(lambda x: nq_scaling(x, nq))
                    else:
                        data[key][subkey][i].iloc[:, 0] = data[key][subkey][i].iloc[:, 0].apply(lambda x: nq_scaling(x, nq))
                    data[key][subkey][i].iloc[:, 1] = data[key][subkey][i].iloc[:, 1].apply(lambda x: nq_scaling(x, nq))
                    if key == "catania":
                        data[key][subkey][i].iloc[:, 2] = data[key][subkey][i].iloc[:, 2].apply(lambda x: nq_scaling(x, nq))
    else:
        for key in data.keys():
            if key == "lambda":
                mass = pdg_db.GetParticle(3122).Mass()
                nq = 3
            elif key == "ks":
                mass = pdg_db.GetParticle(310).Mass()
                nq = 2
            if do_ket_nq:
                data[key]['kEt'] = data[key]["PT [GeV/c]"].apply(lambda x: kEt(mass, x))
                data[key]['kEt/nq'] = data[key]["kEt"].apply(lambda x: nq_scaling(x, nq))
            else:
                data[key]['pt/nq'] = data[key]["PT [GeV/c]"].apply(lambda x: nq_scaling(x, nq))
            data[key]['v2/nq'] = data[key]["v2"].apply(lambda x: nq_scaling(x, nq))
            data[key]['Total Error/nq'] = data[key]["Total Error"].apply(lambda x: nq_scaling(x, nq))
    return data


def read_hists(file_path, markerstyle, markersize=1, colors=[], gname=['gV2PromptStat', 'gSystTotPrompt']):
    '''Read stat and syst error graphs from ROOT file.'''
    if not isinstance(colors, list):
        colors = [colors]
    file = TFile.Open(file_path)
    if not file:
        print('error: failed to open file')
        return
    h_prompt_cent = file.Get(gname[0])
    SetObjectStyle(h_prompt_cent, color=colors[0], markerstyle=markerstyle, markersize=markersize, linewidth=2, fillalpha=0.2)
    h_prompt_systtot = file.Get(gname[1])
    SetObjectStyle(h_prompt_systtot, color=colors[0], linewidth=2)
    h_prompt_systtot.SetFillStyle(0)
    if not h_prompt_cent or not h_prompt_systtot:
        print('failed to get hist')
        return
    return [h_prompt_cent, h_prompt_systtot]


def get_band(low_x, high_x, low_y, high_y, color, doxlim=False):
    '''Create and fill a TPolyLine band between two curves.'''
    graph_high = ROOT.TGraph(len(high_x), array.array('d', high_x), array.array('d', high_y))
    graph_high.SetLineColor(color)
    graph_high.SetLineWidth(0)
    graph_low = ROOT.TGraph(len(low_x), array.array('d', low_x), array.array('d', low_y))
    graph_low.SetLineColor(color)
    graph_low.SetLineWidth(0)

    x_min = max(min(low_x), min(high_x))
    x_max = min(max(low_x), max(high_x))
    x_min = 0.5
    if doxlim:
        x_max = 3.3

    polyline_x = []
    polyline_y = []
    for x, y in zip(low_x, low_y):
        if x >= x_min and x <= x_max:
            polyline_x.append(x)
            polyline_y.append(y)
    for x, y in zip(reversed(high_x), reversed(high_y)):
        if x >= x_min and x <= x_max:
            polyline_x.append(x)
            polyline_y.append(y)
    polyline = ROOT.TPolyLine(len(polyline_x), array.array('d', polyline_x), array.array('d', polyline_y))
    polyline.SetFillColor(color)
    polyline.SetFillStyle(1001)
    polyline.SetLineWidth(0)

    return polyline


def get_latex():
    lat_large = ROOT.TLatex()
    lat_large.SetNDC()
    lat_large.SetTextFont(42)
    lat_large.SetTextSize(0.05)
    lat_mid = ROOT.TLatex()
    lat_mid.SetNDC()
    lat_mid.SetTextFont(42)
    lat_mid.SetTextSize(0.045)
    latLabel = ROOT.TLatex()
    latLabel.SetNDC()
    latLabel.SetTextFont(42)
    latLabel.SetTextSize(0.04)
    return lat_large, lat_mid, latLabel


def get_edges(df):
    '''Infer bin edges from bin center values.'''
    centers = np.array(df["PT [GeV/c]"])
    d = np.diff(centers) / 2
    edges = np.concatenate([[centers[0] - d[0]], centers[:-1] + d, [centers[-1] + d[-1]]])
    edges = np.round(edges / 0.125) * 0.125
    return edges


def fill_hist(data, hist='', columns=["PT [GeV/c]", "v2", "Stat +"]):
    '''Fill a ROOT TH1 from a pandas DataFrame.'''
    if not hist:
        hist = ROOT.TH1F("hist", "Histogram", 100, 0, 10)
    for i in range(len(data[columns[0]])):
        x = data[columns[0]][i]
        y = data[columns[1]][i]
        yerr = data[columns[2]][i]
        hist.SetBinContent(i + 1, y)
        hist.SetBinError(i + 1, yerr)
    hist.SetStats(ROOT.kFALSE)


def fill_graph(data, columns=["PT [GeV/c]", "v2", "Total Error"], compine_syst_stat=True):
    '''Fill a TGraphAsymmErrors from a pandas DataFrame.'''
    n_points = len(data[columns[0]])
    graph = ROOT.TGraphAsymmErrors(n_points)

    for i in range(n_points):
        x = data[columns[0]][i]
        y = data[columns[1]][i]
        yerr = data[columns[2]][i] if compine_syst_stat else 0
        graph.SetPoint(i, x, y)
        dx = 0
        if 'syst' in columns[2]:
            dx = x / 10 * 0.8
            if dx < 0.4:
                dx = 0.25 * 0.8
        graph.SetPointError(i, dx, dx, yerr, yerr)

    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.2)
    graph.SetLineColor(ROOT.kBlue)
    graph.SetLineWidth(2)

    return graph


def graph_to_hist_with_errors(graph, hist_name, pt_bins, title="", use_syst_errors=False, graph_hist=''):
    '''Convert TGraph to TH1F with errors.'''
    if not isinstance(graph, (ROOT.TGraph, ROOT.TGraphErrors)):
        raise TypeError("Input must be a TGraph or TGraphErrors object")
    if len(pt_bins) < 2:
        raise ValueError("pt_bins must contain at least two boundary values")
    if use_syst_errors and not graph_hist:
        raise ValueError("If using systematic errors, must provide graph histogram")

    nbins = len(pt_bins) - 1
    hist = ROOT.TH1F(hist_name, title, nbins, np.asarray(pt_bins, dtype='d'))
    hist.Sumw2()

    n_points = graph.GetN()
    has_errors = isinstance(graph, ROOT.TGraphAsymmErrors)
    y_errs = graph.GetEYhigh() if has_errors else graph.GetEY()

    for ibin in range(n_points):
        hist.SetBinContent(ibin + 1, graph.GetY()[ibin])
        hist.SetBinError(ibin + 1, graph.GetEYhigh()[ibin])
    return hist


def get_interp_hist(hists_target, x_max, input_df=[], name='', cent=True):
    '''Interpolate model predictions at experiment data bin centres for chi2 calculation.'''
    target_bins = get_edges_from_hist(hists_target)
    interpolate_bins = interpolate_pt_bins(target_bins)
    new_hist = create_hist_safely(name, name, interpolate_bins)
    if cent:
        new_hist = hists_target.Clone(name)
    for iPt in range(1, new_hist.GetNbinsX() + 1):
        ptCent = new_hist.GetBinCenter(iPt)
        ptmax = ptCent if cent else new_hist.GetBinLowEdge(iPt + 10)
        if ptmax < x_max:
            if len(input_df) == 1:
                new_hist.SetBinContent(iPt, input_df[0](ptCent))
            elif len(input_df) == 2:
                new_hist.SetBinContent(iPt, np.mean([input_df[0](ptCent), input_df[1](ptCent)]))
        else:
            new_hist.SetBinContent(iPt, 1e-10)
        new_hist.SetBinError(iPt, 0)
    return new_hist


def get_edges_from_hist(hist):
    n_bins = hist.GetNbinsX()
    bin_edges = [hist.GetBinLowEdge(i) for i in range(1, n_bins + 2)]
    return np.array(bin_edges, 'd')


def create_hist_safely(name, title, bin_edges):
    '''Safely create a variable-bin-width TH1F histogram.'''
    if not hasattr(bin_edges, '__iter__'):
        raise ValueError("bin_edges must be an iterable object")
    try:
        bin_edges_array = np.asarray(bin_edges, dtype='d')
    except Exception:
        raise ValueError("Failed to convert bin_edges to a float array")
    if len(bin_edges_array) < 2:
        raise ValueError("bin_edges must contain at least 2 elements")
    if not np.all(np.diff(bin_edges_array) > 0):
        raise ValueError("bin_edges must be strictly monotonically increasing")
    n_bins = len(bin_edges_array) - 1
    hist = ROOT.TH1F(name, title, n_bins, bin_edges_array)
    return hist


def rebin_safely(hist, new_name, new_bin_edges, is_density_hist=False, fixed_rebin=False):
    '''Safely rebin a histogram, supporting density and non-density modes.'''
    hist_clone = hist.Clone(f"{hist.GetName()}_clone")
    if not new_name:
        new_name = f"{hist.GetName()}_rebin"
    if is_density_hist:
        for ibin in range(1, hist_clone.GetNbinsX() + 1):
            old_width = hist_clone.GetBinWidth(ibin)
            content = hist_clone.GetBinContent(ibin)
            error = hist_clone.GetBinError(ibin)
            hist_clone.SetBinContent(ibin, content * old_width)
            hist_clone.SetBinError(ibin, error * old_width)

    n_new_bins = len(new_bin_edges) - 1
    hist_rebin = hist_clone.Rebin(n_new_bins, new_name, np.array(new_bin_edges, 'd'))

    if is_density_hist:
        for ibin in range(1, hist_rebin.GetNbinsX() + 1):
            new_width = hist_rebin.GetBinWidth(ibin)
            content = hist_rebin.GetBinContent(ibin)
            error = hist_rebin.GetBinError(ibin)
            hist_rebin.SetBinContent(ibin, content / new_width)
            hist_rebin.SetBinError(ibin, error / new_width)
    elif fixed_rebin:
        for ibin in range(1, hist_rebin.GetNbinsX() + 1):
            new_width = fixed_rebin
            content = hist_rebin.GetBinContent(ibin)
            error = hist_rebin.GetBinError(ibin)
            hist_rebin.SetBinContent(ibin, content / new_width)
            hist_rebin.SetBinError(ibin, error / new_width)
    return hist_rebin


def interpolate_pt_bins(pt_bins, points_per_interval=9):
    '''Insert interpolated points between each adjacent pair in pt_bins.'''
    interpolated = []
    for i in range(len(pt_bins) - 1):
        a = pt_bins[i]
        b = pt_bins[i + 1]
        interval_points = np.linspace(a, b, points_per_interval + 2)[1:]
        if i == 0:
            interpolated.append(a)
        elif i == len(pt_bins) - 2:
            interval_points = interval_points[:-1]
        interpolated.extend(interval_points.tolist())
    interpolated.append(pt_bins[-1])
    interpolated = [round(x, 2) for x in interpolated]
    return interpolated


def merge_asymmetric_errors(graph_stat, graph_syst):
    '''Combine statistical and systematic errors in quadrature into a single TGraphAsymmErrors.'''
    n_stat = graph_stat.GetN()
    n_syst = graph_syst.GetN()
    if n_stat != n_syst:
        raise ValueError(f"Mismatched number of points: {n_stat} vs {n_syst}")

    merged_graph = graph_stat.Clone(f"merged_{graph_stat.GetName()}")
    merged_graph.SetTitle(f"Merged Errors: {graph_stat.GetTitle()}")

    for i in range(n_stat):
        x_stat = graph_stat.GetX()[i]
        x_syst = graph_syst.GetX()[i]
        if abs(x_stat - x_syst) > 1e-6:
            raise ValueError(f"X coordinate mismatch at point {i}: {x_stat:.4f} vs {x_syst:.4f}")

        exl_stat = graph_stat.GetErrorXlow(i)
        exh_stat = graph_stat.GetErrorXhigh(i)
        eyl_stat = graph_stat.GetErrorYlow(i)
        eyh_stat = graph_stat.GetErrorYhigh(i)
        eyl_syst = graph_syst.GetErrorYlow(i)
        eyh_syst = graph_syst.GetErrorYhigh(i)

        total_eyl = np.sqrt(eyl_stat**2 + eyl_syst**2)
        total_eyh = np.sqrt(eyh_stat**2 + eyh_syst**2)

        merged_graph.SetPointError(i, exl_stat, exh_stat, total_eyl, total_eyh)

    return merged_graph


def model_chi2(data_asymm, h_model, ndf=0):
    '''Compute chi2 between data (TGraphAsymmErrors or TH1F) and model (TH1F).'''
    print(h_model.GetName())
    chi2 = 0.0
    if not ndf:
        for ibin in range(1, h_model.GetNbinsX() + 1):
            if h_model.GetBinContent(ibin) <= 1e-9:
                h_model.SetBinContent(ibin, 0)
                ndf = ibin - 1
                break
    residuals = []
    if isinstance(data_asymm, ROOT.TH1F):
        for ibin in range(1, ndf + 1):
            residual = data_asymm.GetBinContent(ibin) - h_model.GetBinContent(ibin)
            residuals.append(residual)
            chi2 += (residual ** 2) / (data_asymm.GetBinError(ibin) ** 2)
    else:
        x_vals = list(data_asymm.GetX())
        y_vals = list(data_asymm.GetY())
        model_y_vals = []
        y_errs = []
        chi2_list = []
        for i in range(ndf):
            x_data = x_vals[i]
            y_data = y_vals[i]
            y_err_low = data_asymm.GetErrorYlow(i)
            y_err_high = data_asymm.GetErrorYhigh(i)
            x_model = h_model.GetBinCenter(i + 1)
            y_model = h_model.GetBinContent(i + 1)
            model_y_vals.append(y_model)
            if abs(x_data - x_model) > 1e-6:
                raise ValueError(f"pt mismatch at point {i}: data={x_data}, model={x_model}")
            residual = y_data - y_model
            residuals.append(residual)
            if residual > 0:
                if y_err_high < 1e-10:
                    ndf -= 1
                    continue
                sigma = y_err_high
            else:
                if y_err_low < 1e-10:
                    ndf -= 1
                    continue
                sigma = y_err_low
            y_errs.append(sigma)
            chi2 += (residual ** 2) / (sigma ** 2)
            chi2_list.append((residual ** 2) / (sigma ** 2))
    print(f'chi2:{chi2:.2f}; ndf: {ndf}; chi2/ndf: {chi2/ndf:.2f}')
    return chi2, ndf, chi2 / ndf


def compute_ratio_graph(graph_num, graph_den):
    '''Compute ratio of two TGraphAsymmErrors with error propagation.'''
    if isinstance(graph_num, ROOT.TH1F):
        graph_num = ROOT.TGraphAsymmErrors(graph_num)
    if isinstance(graph_den, ROOT.TH1F):
        graph_den = ROOT.TGraphAsymmErrors(graph_den)
    if not (isinstance(graph_num, ROOT.TGraphAsymmErrors) and
            isinstance(graph_den, ROOT.TGraphAsymmErrors)):
        raise TypeError("Inputs must be of type TGraphAsymmErrors")

    n_points = graph_num.GetN()
    if n_points != graph_den.GetN():
        raise ValueError(f"Mismatch in number of points: {n_points} vs {graph_den.GetN()}")

    x = [graph_num.GetX()[i] for i in range(n_points)]
    x_err_low = [graph_num.GetErrorXlow(i) for i in range(n_points)]
    x_err_high = [graph_num.GetErrorXhigh(i) for i in range(n_points)]
    y_num = [graph_num.GetY()[i] for i in range(n_points)]
    y_den = [graph_den.GetY()[i] for i in range(n_points)]
    y_num_err_low = [graph_num.GetErrorYlow(i) for i in range(n_points)]
    y_num_err_high = [graph_num.GetErrorYhigh(i) for i in range(n_points)]

    ratio_y, ratio_err_low, ratio_err_high = [], [], []
    for i in range(n_points):
        if y_den[i] == 0:
            raise ZeroDivisionError(f"Denominator is zero at point {i}")
        ratio = y_num[i] / y_den[i]
        ratio_y.append(ratio)
        ratio_err_low.append(y_num_err_low[i] / y_den[i])
        ratio_err_high.append(y_num_err_high[i] / y_den[i])

    ratio_graph = ROOT.TGraphAsymmErrors(
        n_points,
        np.array(x, dtype=float), np.array(ratio_y, dtype=float),
        np.array(x_err_low, dtype=float), np.array(x_err_high, dtype=float),
        np.array(ratio_err_low, dtype=float), np.array(ratio_err_high, dtype=float)
    )
    return ratio_graph


def scale_x_errors(graph, scale_factor=0.8, target_graph='', target_bins=[]):
    '''Scale x errors of a TGraphAsymmErrors by a given factor.'''
    n_points = graph.GetN()
    for i in range(n_points):
        x = graph.GetX()[i]
        y = graph.GetY()[i]
        x_err_low = graph.GetErrorXlow(i)
        x_err_high = graph.GetErrorXhigh(i)

        if target_graph:
            x_err_low = target_graph.GetErrorXlow(i)
            x_err_high = target_graph.GetErrorXhigh(i)
        elif target_bins:
            x_err_low = (target_bins[i + 1] - target_bins[i]) / 2
            x_err_high = x_err_low

        y_err_low = graph.GetErrorYlow(i)
        y_err_high = graph.GetErrorYhigh(i)
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, x_err_low * scale_factor, x_err_high * scale_factor, y_err_low, y_err_high)

    return graph


def pdf2eps_imagemagick(pdf_paths, target_format='png'):
    '''Convert PDF files to PNG or EPS using ImageMagick.'''
    for pdf in pdf_paths:
        target_path = os.path.splitext(pdf)[0] + f'.{target_format}'
        try:
            subprocess.run([
                'convert', '-trim',
                '-density', '300',
                pdf,
                target_path
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{pdf} ImageMagick convert failed")