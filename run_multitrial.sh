#!/bin/bash

export n_parallel=10
export config_modifies="/home/mdicosta/alice/hf-vn/configs/systMultitrial.yml"
export config_default="/home/mdicosta/alice/hf-vn/configs/test_config.yml"
export output_dir="/home/mdicosta/alice/hf-vn/test_output/cutvar_test"

# Generate YAML file list with indices
echo "Generating yaml files for systematics ..."
python3 /home/mdicosta/alice/hf-vn/make_cutsets_multitrial.py $config_default -m $config_modifies -o $output_dir
yaml_files=($(find "$output_dir" -type f -name "config_var_*.yml" -print0 | xargs -0 -n1))
echo "Yaml files generated!"
echo "${yaml_files[@]}"

pt_dirs=($(ls -d "$output_dir"_multitrial/pt_*))
for dir in "${pt_dirs[@]}"; do
    # Extract just the folder name (without the full path)
    folder_name=$(basename "$dir")
    echo "Processing directory:"
    echo "-----------------------"
    echo "$dir"

    # Find YAML files in the current directory
    yaml_files=($(find "$dir" -maxdepth 2 -type f -name "config_var_*.yml"))
    echo "Number of yaml files found: ${#yaml_files[@]}"

    # Perform projections (and fitting)
    for yaml_file in "${yaml_files[@]}"; do
        echo "Processing YAML file: $yaml_file"
        python3 /home/mdicosta/alice/hf-vn/run_analysis.py "$yaml_file" --combined
    done

done
