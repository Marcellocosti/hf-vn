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
	"Preprocess": os.path.join(work_dir, "./pre_process.py"),
	"YamlCuts": os.path.join(work_dir, "./make_cutsets_cfgs.py"),
	"Projections": os.path.join(work_dir, "./proj_thn.py"),
}

def check_dir(dir):

	if not os.path.exists(dir):
		print(f"\033[32m{dir} does not exist, it will be created\033[0m")
		os.makedirs(dir)
	else:
		print(f"\033[33m{dir} already exists, it will be overwritten\033[0m")
		shutil.rmtree(dir)
		os.makedirs(dir)

	return


def make_yaml(config, out_dir, correlated=False, combined=False):
	print("\033[32mINFO: Make yaml will be performed\033[0m")
	check_dir(f"{out_dir}/cutsets")
	print(f"\033[32mpython3 {paths['YamlCuts']} {config} -o {out_dir} --correlated\033[0m")
	if correlated:
		os.system(f"python3 {paths['YamlCuts']} {config} -o {out_dir} --correlated")
	if combined:
		os.system(f"python3 {paths['YamlCuts']} {config} -o {out_dir}")

def project(config, nworkers, mCutSets):
	print("\033[32mINFO: Projections will be performed\033[0m")
	check_dir(f"{out_dir}/proj")

	def run_projections(i):
		"""Run sparse projection for a given cutset index."""
		iCutSets = f"{i:02d}"
		print(f"\033[32mProcessing cutset {iCutSets}...\033[0m")

		config_cutset = f"{out_dir}/cutsets/cutset_{iCutSets}.yml"
		cmd = (
			f"python3 {paths['Projections']} {config} -cc {config_cutset} "
		)
		print(f"\033[32m{cmd}\033[0m")
		os.system(cmd)

	with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
		results_proj = list(executor.map(run_projections, range(mCutSets)))

def run_correlated_cut_variation(config, operations, nworkers, out_dir):

#___________________________________________________________________________________________________________________________
	# make yaml file
	if operations['make_yaml']:
		make_yaml(config, out_dir, correlated=True)
	else:
		print("\033[33mWARNING: Make yaml will not be performed\033[0m")
	
	mCutSets = len([f for f in os.listdir(f"{out_dir}/cutsets") if os.path.isfile(os.path.join(f"{out_dir}/cutsets", f))])
	print(f"mCutSets: {mCutSets}")
	# quit()
#___________________________________________________________________________________________________________________________
	# Projection for MC and apply the ptweights
	if operations["proj_mc"] or operations["proj_data"]:
		project(config, nworkers, mCutSets)
	else:
		print("\033[33mWARNING: Projections will not be performed\033[0m")


def run_combined_cut_variation(config, operations, nworkers, out_dir):

#___________________________________________________________________________________________________________________________
	# make yaml file
	if operations['make_yaml']:
		make_yaml(config, out_dir, combined=True)
	else:
		print("\033[33mWARNING: Make yaml will not be performed\033[0m")

	mCutSets = len([f for f in os.listdir(f"{out_dir}/cutsets") if os.path.isfile(os.path.join(f"{out_dir}/cutsets", f))])
	print(f"mCutSets: {mCutSets}")

#___________________________________________________________________________________________________________________________
	# Projection for MC and apply the ptweights
	if operations["proj_mc"] or operations["proj_data"]:
		project(config, nworkers, mCutSets)
	else:
		print("\033[33mWARNING: Projections will not be performed\033[0m")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('flow_config', metavar='text', default='config_flow_d0.yml', help='configuration file')
	parser.add_argument("--correlated", "-corr", action="store_true", help="perform correlated analysis")
	parser.add_argument("--combined", "-comb", action="store_true", help="perform combined analysis")
	args = parser.parse_args()

	start_time = time.time()

	# Load and copy the configuration file
	with open(args.flow_config, 'r') as cfgFlow:
		config = yaml.safe_load(cfgFlow)
 
	operations = config['operations']
	nworkers = config['nworkers']
	if args.correlated:
		out_dir = f"{config['out_dir']}/cutvar_{config['suffix']}" + "_correlated"
	else:
		out_dir = f"{config['out_dir']}/cutvar_{config['suffix']}" + "_combined"
	os.system(f"mkdir -p {out_dir}")
  
	# copy the configuration file
	nfile = 0
	os.makedirs(f'{out_dir}/config_flow', exist_ok=True)
	while os.path.exists(f'{out_dir}/config_flow/{os.path.splitext(os.path.basename(args.flow_config))[0]}_{config['suffix']}_{nfile}.yml'):
		nfile = nfile + 1
	os.system(f'cp {args.flow_config} {out_dir}/config_flow/{os.path.splitext(os.path.basename(args.flow_config))[0]}_{config['suffix']}_{nfile}.yml')

	if operations.get('preprocess_data') or operations.get('preprocess_mc'):
		print("\033[32mINFO: Preprocess will be performed\033[0m")
		os.system(f"python3 {paths['Preprocess']} {args.flow_config}")

	if args.correlated:
		run_correlated_cut_variation(args.flow_config, operations, nworkers, out_dir)
	if args.combined:
		run_combined_cut_variation(args.flow_config, operations, nworkers, out_dir)
  
	end_time = time.time()
	execution_time = end_time - start_time
	print(f"\033[34mTotal execution time: {execution_time:.2f} seconds\033[0m")
