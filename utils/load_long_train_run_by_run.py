import os
import subprocess
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import yaml
from ROOT import TFile, THnSparse

def split_into_three(lst):
    # Calculate the base size of each sublist and the remainder
    n = len(lst)
    base_size = n // 3
    remainder = n % 3
    
    # Create the sublists
    sublists = []
    start = 0
    for i in range(3):
        end = start + base_size + (1 if i < remainder else 0)
        sublist = lst[start:end]
        sublists.append(sublist)
        start = end
    
    return sublists

def load_single_runs(config, output_dir, log_lines):
    # Iterate through runs
    threads = config["threads"]
    for run in config['single_runs']:
        run_number = run['number']  # Assuming the run dictionary has a "number" key
        folder = run['folder']
        num_merged = run['num_merged']

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Download files using alien_cp
        with ThreadPoolExecutor(max_workers=threads) as executor:
            downloads = [executor.submit(download_file, f"{folder}/{job:04d}", f"{job:04d}", f"{output_dir}/single_runs/{run_number}") for job in range(run['njobs'] + 1)]

        for job in range(run['njobs'] + 1):
            if not os.path.isfile(f"{output_dir}/single_runs/{run_number}/{job:04d}/AnalysisResults.root"):
                # log_lines.append(f"Checked {output_dir}/single_runs/{run_number}/{job:04d}/AnalysisResults.root\n")
                log_lines.append(f"[Run: {run_number}, Job: {job}] AnalysisResults.root not found for job: {job}\n")

        # Step 2: Find all downloaded files
        command = f'find {output_dir}/single_runs/{run_number} -wholename "*/AnalysisResults.root" | tr "\n" " "'
        print(f"command: {command}")
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        output_list = result.stdout.strip().split()
        print(f"Files found: {len(output_list)}")
        print(output_list)
        print('\n\n')

        if config["merge_all"]:
            # Step 3: Split files into sublists for merging
            total_files = len(output_list)
            files_per_merge = max(1, total_files // num_merged)
            sublists = [output_list[i:i + files_per_merge] for i in range(0, total_files, files_per_merge)]

            # Step 4: Merge the files in parallel
            for i, sublist in enumerate(sublists):
                merge_name = f"{output_dir}/MERGED_{run_number}_{i}.root"
                merge_command = f"hadd -f {merge_name} " + " ".join(sublist)
                print(f"\n[Merging] {merge_command}")
                os.system(merge_command)

    print("All jobs completed.")
    return log_lines

def download_file(file_path, run, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    command = f"alien_cp {file_path}/AnalysisResults.root file:{output_dir}/{run}/"
    print(f"[Downloading] {command}")
    os.system(command)
    input_file = Path(f"{output_dir}/runs/{run}/AnalysisResults.root")  # check copying of file
    return command, input_file # Optionally return something for logging

def merge_files(sublist, merge_name, sparse_path='', force=False):
    merge_command = f"hadd -f {merge_name} " + " ".join(sublist) if force else f"hadd {merge_name} " + " ".join(sublist)
    print(f"\n[Merging] {merge_command}")
    os.system(merge_command)
    merged_file = TFile.Open(merge_name, 'r')
    if sparse_path != '':
        status = True if isinstance(merged_file.Get(sparse_path), THnSparse) else False
    else:
        status = True
    return sublist, status
    
def run_downloader(config, output_dir, log_lines):

    suffix = config["suffix"]
    num_merged = config["num_merged"]
    sparse_path = config.get("sparse_path", '')
    threads = config["threads"]
    force = config["force"]

    print(f"Starting parallel downloads with {threads} threads...\n")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        downloads = [executor.submit(download_file, file_path, run, f"{output_dir}/runs/") for file_path, run in zip(config['grid_dirs'], config['grid_runs'])]

    for run in config['grid_runs']:
        if not os.path.isfile(f"{output_dir}/runs/{run}/AnalysisResults.root"):
            # log_lines.append(f"Checked {output_dir}/runs/{run}/AnalysisResults.root\n")
            log_lines.append(f"[Run: {run}] AnalysisResults.root not found!\n")

    # Step 2: Find downloaded files
    find_command = f'find {output_dir}/runs -wholename "*/AnalysisResults.root" | tr "\\n" " "'
    result = subprocess.run(find_command, shell=True, text=True, capture_output=True)
    output_list = result.stdout.strip().split()

    print("\nTotal files found:", len(output_list))

    # Step 3: Create sublists for merging
    total_files = len(output_list)
    files_per_merge = max(1, total_files // num_merged)
    sublists = [output_list[i:i + files_per_merge] for i in range(0, total_files, files_per_merge)]

    # Step 4: Parallel merging of files
    if config["merge_all"]:
        print(f"\nStarting parallel merging with {threads} threads...\n")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            mergers = []
            for i, sublist in enumerate(sublists):
                merge_name = f"{output_dir}/MERGED_{i}_{suffix}.root"
                mergers.append(executor.submit(merge_files, sublist, merge_name, sparse_path, force))
            
            for future in as_completed(mergers):
                future.result()  # catch exceptions and confirm merge
            
        print(f"mergers: {mergers}")
        for i, future in enumerate(mergers):
            sublist, status = future.result()
            if status:
                log_lines.append(f"\nMerged {sublist} into {output_dir}/MERGED_{i}_{suffix}.root with status: {status}.")
            else:
                log_lines.append(f"\nError merging files into {output_dir}/MERGED_{i}_{suffix}.root. Trying to split runs")
                subsublist = split_into_three(sublist)
                for list in subsublist:
                    merge_command = f"hadd -f {output_dir}/MERGED_{i+num_merged}_{suffix}.root {list}" if force else f"hadd {merge_name} {list}"
                    print(f"\n[Merging] {merge_command}")
                    os.system(merge_command)
                num_merged += len(subsublist)

        print(f"\n\n")
        for future in as_completed(downloads):
            _, input_file = future.result()  # just to catch any exceptions
            if not input_file.is_file():
                log_lines.append(f"    Error downloading file: {input_file} --> run merging needed!\n")

    return log_lines

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download files from ALIEN and merge them using hadd.")
    parser.add_argument("cfg", type=str, default="cfg.yml", help="Path to the input file list (default: runs.txt)")
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as yml_file:
        config = yaml.load(yml_file, yaml.FullLoader)

    output_dir = f"{config['output_dir']}/Train{config['train_number']}/"
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # Step 5: Copy the runs.txt file
    with open(f"{output_dir}/cfg.yml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    log_lines = []
    if config["download_merged_runs"]:
        print(f"Loading merged runs ... ")
        run_downloader(config, output_dir, log_lines)
    if config["download_single_runs"]:
        print(f"Loading single runs ... ")
        load_single_runs(config, output_dir, log_lines)


    with open(f"{output_dir}/log.txt", "w") as file:
        file.writelines(log_lines)

    print(f"Lines appended to {output_dir}/log.txt successfully!")