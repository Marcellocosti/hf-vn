#!/bin/bash

trial_fit() {
    config_file=$1
    suffix=$(basename "$config_file" .yml)
	echo "Evaluating $1"
	dir_path=$(dirname "$file_path")

    python3 ~/alice/DmesonAnalysis/run3/flow/BDT/run_cutvar.py $1 \
                                                               $config_flow \
                                                               $usepreprocessed \
                                                               $calc_weights \
                                                               $make_yaml \
                                                               $proj_data \
                                                               $proj_mc \
                                                               $efficiency \
                                                               $vn
}
export -f trial_fit

prompt_v2() {
    config_file=$1
    
    yaml_index=$(basename "$config_file" .yml | sed 's/config_var_//')
    echo "Processing file index $yaml_index and config_file $config_file"
    
    yaml_directory_path=$(dirname "$config_file")
    pt_directory_path=$(dirname "$yaml_directory_path")

    suffix=$(basename "$config_file" .yml)

    cmd=(python3 "/home/mdicosta/alice/DmesonAnalysis/run3/flow/BDT/ComputeV2vsFDFrac.py" "$config_file" \
        -i "$pt_directory_path/trails/cutvar_$yaml_index" \
        -o "$pt_directory_path/trails/cutvar_$yaml_index" \
        -s "$suffix" \
        -comb \
        -ic "$combPath" \
        -oc "$pt_directory_path/trails/cutvar_$yaml_index" \
        --systematics)

    echo "Executing: ${cmd[*]}"
    "${cmd[@]}"
}
export -f prompt_v2

##################################################################################################################################################

export config_modifies="/home/mdicosta/FlowDplus/FinalResults/3040final/multitrial_ptdiff_manytrials/fit_config_modifications.yml"
export config_default="/home/mdicosta/FlowDplus/FinalResults/3040final/config_3040_combined.yml" # bdt cut of corr and uncorr, skip_cut, sysematic
export combPath="/home/mdicosta/FlowDplus/FinalResults/3040final/cutvar_combined"

export n_parallel=15

export output_dir="/home/mdicosta/FlowDplus/FinalResults/3040final/multitrial_ptdiff_manytrials"
export syst_paths=$output_dir
export multitrial_config_path="/home/mdicosta/FlowDplus/FinalResults/3040final/multitrial_ptdiff_manytrials"

export usepreprocessed=True
export docw=True
export domy=True 
export doprojdata=True
export doprojmc=True
export doeff=True
export dovn=True
export dov2vf=True

export usepreprocessed=$([ "$usepreprocessed" = "False" ] && echo "" || echo "--use_preprocessed")
export calc_weights=$([ "$docw" = "False" ] && echo "" || echo "--do_calc_weights")
export make_yaml=$([ "$domy" = "False" ] && echo "" || echo "--do_make_yaml")
export proj_data=$([ "$doprojdata" = "False" ] && echo "" || echo "--do_proj_data")
export proj_mc=$([ "$doprojmc" = "False" ] && echo "" || echo "--do_proj_mc")
export efficiency=$([ "$doeff" = "False" ] && echo "" || echo "--do_efficiency")
export vn=$([ "$dovn" = "False" ] && echo "" || echo "--do_vn")
export v2_vs_frac=$([ "$dov2vf" = "False" ] && echo "" || echo "--do_v2_vs_frac")

# # Generate YAML file list with indices
# echo "Generating yaml files for systematics ..."
# python3 /home/mdicosta/alice/DmesonAnalysis/run3/flow/systematics/make_yaml_for_syst_ptdiff.py $config_default -m $config_modifies -o $output_dir -mb
# yaml_files=($(find "$multitrial_config_path" -type f -name "config_var_*.yml" -print0 | xargs -0 -n1))
# echo "Yaml files generated!"
# echo "${yaml_files[@]}"

pt_dirs=($(ls -d "$multitrial_config_path"/pt_*))
for dir in "${pt_dirs[@]}"; do
    # Extract just the folder name (without the full path)
    folder_name=$(basename "$dir")
    echo "Processing directory:"
    echo "-----------------------"
    echo "$dir"

    # Find YAML files in the current directory
    yaml_files=($(find "$dir" -maxdepth 2 -type f -name "config_var_*.yml"))
    echo "${#yaml_files[@]}"

    # # perform multitrial fits
    # echo "Performing multitrial fits ..."
    # echo "${yaml_files[@]}"
    # echo "${#yaml_files[@]}"
    # printf "%s\n" "${yaml_files[@]}" | parallel -j $n_parallel --progress --eta trial_fit {}
    # echo "Multitrial fits performed!"

    # repeat the V2VsFrac estimation
    find "$dir" -type d -name "V2VsFrac" -exec rm -rf {} +
    parallel -j $n_parallel --progress --eta prompt_v2 {1} ::: "${yaml_files[@]}"
    find "$dir" -type d -name "syst_summary" -exec rm -rf {} +
    python3 ~/alice/DmesonAnalysis/run3/flow/systematics/compute_syst_multitrial_bdt.py "$dir" "/home/mdicosta/FlowDplus/FinalResults/3040final/cutvar_combined/" -o $dir --prompt
    
done

python3 "/home/mdicosta/alice/DmesonAnalysis/run3/flow/systematics/merge_syst_pt_bins.py" $output_dir