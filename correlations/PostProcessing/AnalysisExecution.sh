#!/bin/bash

# \brief Bash script to run the azimuthal correlation analysis
# \usage ./AnalysisExecution.sh

log_file="/home/mdicosta/DFlowOO/Correlations/Maps/stdoutFitCorrel_010_negDeta.log"

root -b -l <<'EOF' 2>&1 | tee "$log_file"

gSystem->SetBuildDir(".", kTRUE);

// Tell ROOT where yaml-cpp is installed
gSystem->AddIncludePath("-I/home/mdicosta/local/include");
gSystem->AddDynamicPath("/home/mdicosta/local/lib");
gSystem->Load("libyaml-cpp");

// Compile macros
.L DhCorrelationFitter.cxx++
.L FitCorrel.C++

// Run analysis
.x FitCorrel.C("/home/mdicosta/DFlowOO/Correlations/Maps/config_fit.json", "/home/mdicosta/DFlowOO/Correlations/Maps/config_020.yml")

.q
EOF