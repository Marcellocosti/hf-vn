import shutil
import os
import sys
import ROOT
from ROOT import TH1, TH2, TH3
import numpy as np

def check_dir(dir):

	if not os.path.exists(dir):
		print(f"\033[32m{dir} does not exist, it will be created\033[0m")
		os.makedirs(dir)
	else:
		print(f"\033[33m{dir} already exists, it will be overwritten\033[0m")
		shutil.rmtree(dir)
		os.makedirs(dir)

	return

def make_dir_root_file(dir, file):
    if not file.GetDirectory(dir):
        file.mkdir(dir)

def profile_mass_sp(hist_mass_sp, inv_mass_bins, resolution):
    '''
    Profile the mass sparse to get vn versus mass
    Input:
        - hist_mass_sp:
            THnSparse, input THnSparse object (already projected in centrality and pt)
        - inv_mass_bins:
            list of floats, bin edges for the mass axis
        - resolution:
            float, resolution to normalize the vn values
    Output:
        - hist_vn_vs_mass:
            TH1D, histogram with vn as a function of mass
    '''
    hist_vn_vs_mass = ROOT.TH1D('hist_vn_vs_mass', 'hist_vn_vs_mass', len(inv_mass_bins)-1, np.array(inv_mass_bins))
    hist_vn_vs_mass.SetDirectory(0)
    for i in range(hist_vn_vs_mass.GetNbinsX()):
        bin_low = hist_mass_sp.GetXaxis().FindBin(inv_mass_bins[i])
        bin_high = hist_mass_sp.GetXaxis().FindBin(inv_mass_bins[i+1])
        profile = hist_mass_sp.ProfileY(f'profile_{bin_low}_{bin_high}', bin_low, bin_high)
        mean_sp = profile.GetMean()
        mean_sp_err = profile.GetMeanError()
        hist_vn_vs_mass.SetBinContent(i+1, mean_sp / resolution)
        hist_vn_vs_mass.SetBinError(i+1, mean_sp_err / resolution)
    return hist_vn_vs_mass

def get_vn_versus_mass(thnSparses, resolutions, inv_mass_bins, mass_axis, vn_axis, sampling=-1, debug=False):
    '''
    Project vn versus mass

    Input:
        - thnSparse:
            THnSparse, input THnSparse obeject (already projected in centrality and pt)
        - inv_mass_bins:
            list of floats, bin edges for the mass axis
        - mass_axis:
            int, axis number for mass
        - vn_axis:
            int, axis number for vn
        - debug:
            bool, if True, create a debug file with the projections (default: False)

    Output:
        - hist_mass_proj:
            TH1D, histogram with vn as a function of mass
    '''

    invmass_bins = np.array(inv_mass_bins)

    if sampling != -1:
        print('Sampling vn versus mass to be implemented!')
    else:
        for iThn, ((_, sparse), (_, reso)) in enumerate(zip(thnSparses.items(), resolutions.items())):
            resolution = reso.GetBinContent(1)
            hist_vn_proj_temp = sparse.Projection(vn_axis, mass_axis)
            hist_vn_proj_temp.SetName(f'hist_vn_proj_{iThn}')
            hist_vn_proj_temp.SetDirectory(0)
            
            if iThn == 0:
                hist_vn_proj = hist_vn_proj_temp.Clone('hist_vn_proj')
                hist_vn_proj.SetDirectory(0)
                hist_vn_proj.Reset()

            hist_vn_proj.Add(hist_vn_proj_temp)

        hist_vn_vs_mass = profile_mass_sp(hist_vn_proj, invmass_bins, resolution)

    if debug:
        outfile = ROOT.TFile('debug.root', 'RECREATE')
        hist_vn_proj.Write()
        hist_vn_vs_mass.Write()
        outfile.Close()

    return hist_vn_vs_mass

def get_centrality_bins(centrality):
    '''
    Get centrality bins

    Input:
        - centrality:
            str, centrality class (e.g. 'k3050')

    Output:
        - cent_bins:
            list of floats, centrality bins
        - cent_label:
            str, centrality label
    '''
    if centrality == 'k05':
        return '0_5', [0, 5]
    if centrality == 'k510':
        return '5_10', [5, 10]
    if centrality == 'k010':
        return '0_10', [0, 10]
    if centrality == 'k1015':
        return '10_15', [10, 15]
    if centrality == 'k1520':
        return '15_20', [15, 20]
    if centrality == 'k1020':
        return '10_20', [10, 20]
    if centrality == 'k020':
        return '0_20', [0, 20]
    if centrality == 'k2030':
        return '20_30', [20, 30]
    elif centrality == 'k3040':
        return '30_40', [30, 40]
    elif centrality == 'k3050':
        return '30_50', [30, 50]
    elif centrality == 'k4050':
        return '40_50', [40, 50]
    elif centrality == 'k2060':
        return '20_60', [20, 60]
    elif centrality == 'k4060':
        return '40_60', [40, 60]
    elif centrality == 'k4080':
        return '40_80', [40, 80]
    elif centrality == 'k5060':
        return '50_60', [50, 60]
    elif centrality == 'k5080':
        return '50_80', [50, 80]
    elif centrality == 'k6070':
        return '60_70', [60, 70]
    elif centrality == 'k6080':
        return '60_80', [60, 80]
    elif centrality == 'k7080':
        return '70_80', [70, 80]
    elif centrality == 'k0100':
        return '0_100', [0, 100]
    else:
        print(f"ERROR: cent class \'{centrality}\' is not supported! Exit")
    sys.exit()

def reweight_histo_1D(histo, weights, binned=False):
    for iBin in range(1, histo.GetNbinsX()+1):
        ptCent = histo.GetBinCenter(iBin)
        weight = weights[iBin-1] if binned else weights(ptCent) if weights(ptCent) > 0 else 0
        histo.SetBinContent(iBin, histo.GetBinContent(iBin) * weight)
        histo.SetBinError(iBin, histo.GetBinError(iBin) * weight)
    proj_hist = histo.Clone(histo.GetName())
    return proj_hist

def reweight_histo_2D(histo, weights, binned=False):
    for iBinX in range(1, histo.GetXaxis().GetNbins()+1):
        for iBinY in range(1, histo.GetYaxis().GetNbins()+1):
            if binned:
                weight = weights[iBinY-1]
            else:
                binCentVal = histo.GetYaxis().GetBinCenter(iBinY)
                weight = weights(binCentVal) if weights(binCentVal) > 0 else 0
            weighted_content = histo.GetBinContent(iBinX, iBinY) * weight
            weighted_error = histo.GetBinError(iBinX, iBinY) * weight if weight > 0 else 0
            histo.SetBinContent(iBinX, iBinY, weighted_content)
            histo.SetBinError(iBinX, iBinY, weighted_error)
    proj_hist = histo.ProjectionX(histo.GetName(), 0, histo.GetYaxis().GetNbins()+1, 'e')
    return proj_hist

def reweight_histo_3D(histo, weightsY, weightsZ):
    for iBinX in range(1, histo.GetXaxis().GetNbins()+1):
        for iBinY in range(1, histo.GetYaxis().GetNbins()+1):
            for iBinZ in range(1, histo.GetZaxis().GetNbins()+1):
                binCenterY = histo.GetYaxis().GetBinCenter(iBinY)
                weight = weightsZ[iBinZ-1]*weightsY(binCenterY) if weightsY(binCenterY) > 0 else weightsZ[iBinZ-1] 
                weighted_content = histo.GetBinContent(iBinX, iBinY, iBinZ) * weight
                weighted_error = histo.GetBinError(iBinX, iBinY, iBinZ) * weight if weight > 0 else 0
                histo.SetBinContent(iBinX, iBinY, iBinZ, weighted_content)
                histo.SetBinError(iBinX, iBinY, iBinZ, weighted_error)
    proj_hist = histo.ProjectionX(histo.GetName(), 0, histo.GetYaxis().GetNbins()+1,
                                  0, histo.GetZaxis().GetNbins()+1, 'e')
    return proj_hist