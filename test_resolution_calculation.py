from utils import reweight_histo, get_vn_versus_mass
from ROOT import TFile

infile = TFile('/home/mdicosta/alice/hf-vn/test_output/preprocess/AnalysisResults_pt_10_15.root')
hResolution = infile.Get('Data_Flow_2023/hResolution')
reso = hResolution.GetBinContent(1)
sparseData = infile.Get('Data_Flow_2023/hSparseFlowCharm')

mass_axis = 0
sp_axis = 1

flowDataDict = {}
flowDataDict["Data_Flow_2023"] = sparseData
print(flowDataDict)
resoDict = {}
resoDict["Reso_Flow_2023"] = hResolution
print(resoDict)

inv_mass_bins = [1.70,1.72,1.73,1.74,1.75,1.78,1.81,1.83,1.85,1.87,1.89,1.92,1.95,1.96,1.97,1.98,1.99,2.00,2.01,2.02,2.03,2.04,2.05]

hVnVsMassSampled = get_vn_versus_mass(flowDataDict, resoDict, inv_mass_bins, mass_axis, sp_axis)
hVnVsMassStd = get_vn_versus_mass(flowDataDict, resoDict, inv_mass_bins, mass_axis, sp_axis, 10000)

outFile = TFile('test.root', 'RECREATE')
# hVnVsMassSampled.Write('hVnVsMassSampled')
hVnVsMassStd.Write('hVnVsMassStd')
hVnVsMassStd.Scale(1./reso)
hVnVsMassStd.Write('hVnVsMassStdResoScaled')
outFile.Close()