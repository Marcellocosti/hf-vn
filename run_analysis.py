import os
import sys
import argparse
import yaml
import shutil
import concurrent.futures
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor
work_dir = os.path.dirname(os.path.realpath(__file__))

paths = {
	"YamlCuts": os.path.join(work_dir, "./yaml_makers/make_cutsets_cfgs.py"),
}

def check_dir(dir):

	if not os.path.exists(dir):
		print(f"\033[32m{dir} does not exist, it will be created\033[0m")
		os.makedirs(dir)
	else:
		print(f"\033[33m{dir} already exists, it will be removed and recreated\033[0m")
		shutil.rmtree(dir)
		os.makedirs(dir)

	return

def run_correlated_cut_variation(config, operations, nworkers, out_dir):

#___________________________________________________________________________________________________________________________
	# make yaml file
	if operations['make_yaml']:
		print("\033[32mINFO: Make yaml will be performed\033[0m")
		check_dir(f"{out_dir}/config")
		print(f"\033[32mpython3 {paths['YamlCuts']} {config} -o {out_dir} --correlated\033[0m")
		os.system(f"python3 {paths['YamlCuts']} {config} -o {out_dir} --correlated")
	else:
		print("\033[33mWARNING: Make yaml will not be performed\033[0m")


def run_combined_cut_variation(config, operations, nworkers, out_dir):

#___________________________________________________________________________________________________________________________
	# make yaml file
	if operations['make_yaml']:
		print("\033[32mINFO: Make yaml will be performed\033[0m")
		check_dir(f"{out_dir}/config")
		print(f"\033[32mpython3 {paths['YamlCuts']} {config} -o {out_dir}\033[0m")
		os.system(f"python3 {paths['YamlCuts']} {config} -o {out_dir}")
	else:
		print("\033[33mWARNING: Make yaml will not be performed\033[0m")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('flow_config', metavar='text', default='config_flow_d0.yml', help='configuration file')
	parser.add_argument("--correlated", "-corr", action="store_true", help="perform correlated v2 estimation")
	parser.add_argument("--combined", "-comb", action="store_true", help="perform combined v2 estimation")
	args = parser.parse_args()

	start_time = time.time()

	# Load and copy the configuration file
	with open(args.flow_config, 'r') as cfgFlow:
		config = yaml.safe_load(cfgFlow)
 
	operations = config['operations']
	nworkers = config['nworkers']
	if args.correlated:
		output_dir = f"{config['out_dir']}/cutvar_{config['suffix']}" + "_correlated"
	else:
		output_dir = f"{config['out_dir']}/cutvar_{config['suffix']}" + "_combined"
	os.system(f"mkdir -p {output_dir}")
  
	# copy the configuration file
	nfile = 0
	os.makedirs(f'{output_dir}/config_flow', exist_ok=True)
	while os.path.exists(f'{output_dir}/config_flow/{os.path.splitext(os.path.basename(args.flow_config))[0]}_{config['suffix']}_{nfile}.yml'):
		nfile = nfile + 1
	os.system(f'cp {args.flow_config} {output_dir}/config_flow/{os.path.splitext(os.path.basename(args.flow_config))[0]}_{config['suffix']}_{nfile}.yml')

	if args.correlated:
		run_correlated_cut_variation(args.flow_config, operations, nworkers, output_dir)
	if args.combined:
		run_combined_cut_variation(args.flow_config, operations, nworkers, output_dir)
  
	end_time = time.time()
	execution_time = end_time - start_time
	print(f"\033[34mTotal execution time: {execution_time:.2f} seconds\033[0m")
