import ROOT
from ROOT import TH1, TH2, TH3, RooRealVar, RooDataHist, RooHistPdf, RooArgList, RooFit
import numpy as np

def make_dir_root_file(dir, file):
    if not file.GetDirectory(dir):
        file.mkdir(dir)

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
    hist_vn_vs_mass = ROOT.TH1D('hist_vn_vs_mass', 'hist_vn_vs_mass', len(inv_mass_bins)-1, np.array(inv_mass_bins))
    hist_vn_vs_mass.SetDirectory(0)

    if sampling != -1:
        print('Sampling vn versus mass...')
        for i in range(hist_vn_vs_mass.GetNbinsX()):
            bin_low_edge = hist_vn_vs_mass.GetXaxis().GetBinLowEdge(i)
            bin_high_edge = hist_vn_vs_mass.GetXaxis().GetBinUpEdge(i)
            print(f'Bin {i}: Low Edge = {bin_low_edge}, High Edge = {bin_high_edge}')
            vn_values = np.array([])
            sparse_integrals = []
            for _, sparse in thnSparses.items():
                # Selecting the mass range associated to the bin
                sparse.GetAxis(mass_axis).SetRangeUser(bin_low_edge, bin_high_edge)
                sparse_integrals.append(sparse.Projection(mass_axis).Integral())
            for iThn, ((key, sparse), (reso_key, reso)) in enumerate(zip(thnSparses.items(), resolutions.items())):
                hist_sp_temp = sparse.Projection(vn_axis)
                hist_sp_temp.SetName(f'hist_sp_{iThn}')
                hist_sp_temp.SetDirectory(0)

                x = ROOT.RooRealVar("x", "x", -0.8, 0.8)
                data_hist = ROOT.RooDataHist("data_hist", "Histogram data", ROOT.RooArgList(x), hist_sp_temp)

                # Step 3: Create a RooHistPdf directly from the histogram (no fitting)
                hist_pdf = ROOT.RooHistPdf("hist_pdf", "Histogram PDF", ROOT.RooArgList(x), data_hist)

                # Step 4: Sample 1000 points from the histogram PDF
                n_samples = 10000
                samples = hist_pdf.generate(ROOT.RooArgList(x), n_samples)

                # print(f"type(samples): {type(samples)}")
                # samples = samples.GetClonedTree()
                # print(f"type(samples): {type(samples)}")
                # print(f"samples.GetEntries(): {samples.GetEntries()}")

                # Step 5: Create a plot to visualize
                frame = x.frame(ROOT.RooFit.Title("Histogram PDF"))
                data_hist.plotOn(frame)
                hist_pdf.plotOn(frame)

                # Show the plot
                canvas = ROOT.TCanvas("canvas", "Histogram PDF", 800, 600)
                frame.Draw()
                canvas.SaveAs(f"/home/mdicosta/alice/hf-vn/test_output/images/hist_pdf_{i}.png")

                # Print samples to check their status
                print(f"samples: {samples}")

                # Step 6: Extract and save the sampled points
                sampled_points = []

                # Loop over the rows in the RooAbsData (samples)
                print(f"samples.numEntries(): {samples.numEntries()}")
                for i in range(samples.numEntries()):
                    sample = samples.get(i)  # Get the i-th sample (RooArgSet)
                    value = sample.getRealValue('x')  # Extract the 'x' value from the sample
                    sampled_points.append(value)  # Append the value to the list

                # Convert the list to a numpy array
                sampled_points = np.array(sampled_points)

                # Print the sampled points (optional)
                print(f"\nsampled_points: {sampled_points}\n")

                # Optionally save to a text file for later use
                np.savetxt(f"sampled_points_{i}.txt", sampled_points)

                # Optionally, print a few values
                print(f"First 10 sampled points: {sampled_points[:10]}")


            mean_sp = 0.
            mean_sp_err = 0.
            hist_vn_vs_mass.SetBinContent(i+1, mean_sp)
            hist_vn_vs_mass.SetBinError(i+1, mean_sp_err)
    else:
        print('No sampling, using the original histograms...')
        for iThn, (key, sparse) in enumerate(thnSparses.items()):
            print(f'Processing {key}...')
            print(f'Processing {iThn}...')
            print(f'Processing {sparse}...')
            hist_vn_proj_temp = sparse.Projection(vn_axis, mass_axis)
            hist_vn_proj_temp.SetName(f'hist_vn_proj_{iThn}')
            hist_vn_proj_temp.SetDirectory(0)
            
            if iThn == 0:
                hist_vn_proj = hist_vn_proj_temp.Clone('hist_vn_proj')
                hist_vn_proj.SetDirectory(0)
                hist_vn_proj.Reset()

            hist_vn_proj.Add(hist_vn_proj_temp)

        for i in range(hist_vn_vs_mass.GetNbinsX()):
            bin_low = hist_vn_proj.GetXaxis().FindBin(invmass_bins[i])
            bin_high = hist_vn_proj.GetXaxis().FindBin(invmass_bins[i+1])
            profile = hist_vn_proj.ProfileY(f'profile_{bin_low}_{bin_high}', bin_low, bin_high)
            mean_sp = profile.GetMean()
            mean_sp_err = profile.GetMeanError()
            hist_vn_vs_mass.SetBinContent(i+1, mean_sp)
            hist_vn_vs_mass.SetBinError(i+1, mean_sp_err)

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


def reweight_histo(histo, weights, histoname, specieweights=[]):
    if specieweights != []:
        if isinstance(histo, TH2) and not isinstance(histo, TH3):
            for iBinX in range(1, histo.GetXaxis().GetNbins()+1):
                for iBinY in range(1, histo.GetYaxis().GetNbins()+1):
                    origContent = histo.GetBinContent(iBinX, iBinY)
                    origError = histo.GetBinError(iBinX, iBinY)
                    weight = specieweights[iBinY-1]
                    content = origContent * weight
                    error = origError * weight if weight > 0 else 0
                    histo.SetBinContent(iBinX, iBinY, content)
                    histo.SetBinError(iBinX, iBinY, error)
            proj_hist = histo.ProjectionX(histoname, 0, histo.GetYaxis().GetNbins()+1, 'e')

        if isinstance(histo, TH3):
            for iBinX in range(1, histo.GetXaxis().GetNbins()+1):
                for iBinY in range(1, histo.GetYaxis().GetNbins()+1):
                    for iBinZ in range(1, histo.GetZaxis().GetNbins()+1):
                        binCentVal = histo.GetYaxis().GetBinCenter(iBinY)
                        origContent = histo.GetBinContent(iBinX, iBinY, iBinZ)
                        origError = histo.GetBinError(iBinX, iBinY, iBinZ)
                        weight = specieweights[iBinZ-1]*weights(binCentVal) if weights(binCentVal) > 0 else specieweights[iBinZ-1] 
                        content = origContent * weight
                        error = origError * weight if weight > 0 else 0
                        histo.SetBinContent(iBinX, iBinY, iBinZ, content)
                        histo.SetBinError(iBinX, iBinY, iBinZ, error)
            proj_hist = histo.ProjectionX(histoname, 0, histo.GetYaxis().GetNbins()+1,
                                          0, histo.GetZaxis().GetNbins()+1, 'e')

    else:
        if isinstance(histo, TH1) and not isinstance(histo, TH2):
            for iBin in range(1, histo.GetNbinsX()+1):
                if histo.GetBinContent(iBin) > 0.:
                    relStatUnc = histo.GetBinError(iBin) / histo.GetBinContent(iBin)
                    ptCent = histo.GetBinCenter(iBin)
                    histo.SetBinContent(iBin, histo.GetBinContent(iBin) * weights(ptCent))
                    histo.SetBinError(iBin, histo.GetBinContent(iBin) * relStatUnc)
            proj_hist = histo.Clone(histoname)
        if isinstance(histo, TH2) and not isinstance(histo, TH3):
            for iBinX in range(1, histo.GetXaxis().GetNbins()+1):
                for iBinY in range(1, histo.GetYaxis().GetNbins()+1):
                    binCentVal = histo.GetYaxis().GetBinCenter(iBinY)
                    origContent = histo.GetBinContent(iBinX, iBinY)
                    origError = histo.GetBinError(iBinX, iBinY)
                    weight = weights(binCentVal) if weights(binCentVal) > 0 else 0
                    content = origContent * weight
                    error = origError * weight if weight > 0 else 0
                    histo.SetBinContent(iBinX, iBinY, content)
                    histo.SetBinError(iBinX, iBinY, error)
            proj_hist = histo.ProjectionX(histoname, 0, histo.GetYaxis().GetNbins()+1, 'e')
        
    return proj_hist
