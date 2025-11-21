from ROOT import TFile
from itertools import product

input_file = TFile.Open("/home/mdicosta/DFlowOO/Correlations/020_ME_nodeltaetacut//AnalysisResults.root", "READ")
sparse_ME = input_file.Get("hf-correlator-flow-charm-hadrons-reduced/hSparseCorrelationsMECharmHad")

pool_bin_ax_num = 0
pt_trig_ax_num = 1
pt_assoc_ax_num = 2
delta_eta_ax_num = 3
delta_phi_ax_num = 4

pool_bins_axes = sparse_ME.GetAxis(pool_bin_ax_num)
pt_trig_axes = sparse_ME.GetAxis(pt_trig_ax_num)
pt_assoc_axes = sparse_ME.GetAxis(pt_assoc_ax_num)

bin_tuples = product(
    range(1, pool_bins_axes.GetNbins() + 1),
    range(1, pt_trig_axes.GetNbins() + 1),
    range(1, pt_assoc_axes.GetNbins() + 1),
)

delta_eta_histos, delta_phi_histos = {}, {}
for i_pool_bin, i_pt_trig_bin, i_pt_assoc_bin in bin_tuples:
    print(f"Processing Pool Bin {i_pool_bin}, Pt Trig Bin {i_pt_trig_bin}, Pt Assoc Bin {i_pt_assoc_bin}")
    sparse_ME.GetAxis(pool_bin_ax_num).SetRange(i_pool_bin, i_pool_bin)
    sparse_ME.GetAxis(pt_trig_ax_num).SetRange(i_pt_trig_bin, i_pt_trig_bin)
    sparse_ME.GetAxis(pt_assoc_ax_num).SetRange(i_pt_assoc_bin, i_pt_assoc_bin)
    delta_eta_histos[f"PoolBin_{i_pool_bin}_PtTrigBin_{i_pt_trig_bin}_PtAssocBin_{i_pt_assoc_bin}"] = sparse_ME.Projection(delta_eta_ax_num)
    delta_phi_histos[f"PoolBin_{i_pool_bin}_PtTrigBin_{i_pt_trig_bin}_PtAssocBin_{i_pt_assoc_bin}"] = sparse_ME.Projection(delta_phi_ax_num)

outfile = TFile.Open("/home/mdicosta/DFlowOO/Correlations/020_ME_nodeltaetacut/MixedEventQA.root", "RECREATE")
for key, histo in delta_eta_histos.items():
    print(f"Writing DeltaEta histogram for {key}")
    histo.Write(f"DeltaEta_{key}")
for key, histo in delta_phi_histos.items():
    print(f"Writing DeltaPhi histogram for {key}")
    histo.Write(f"DeltaPhi_{key}")
outfile.Close()
input_file.Close()
