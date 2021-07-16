import glob
import os
import subprocess
import sys

SCRATCH_DIR = sys.argv[1]
DATA_DIR = sys.argv[2]

TEMP_OUTPUT_DIR = f"{SCRATCH_DIR}/mt_outputs/val_temp/"
OUTPUT_DIR = f"{SCRATCH_DIR}/mt_outputs/val/"
NORM_OUTPUT_DIR = f"{SCRATCH_DIR}/mt_outputs/val_norm/"

files = sorted(glob.glob(f'{TEMP_OUTPUT_DIR}/*.pt'))

os.system(f'bash -c "cut {DATA_DIR}/mt_enghinglish/dev.txt -f 2 | python3 my_cm_tokenizer.py 0 > {OUTPUT_DIR}/reference.txt"')
os.system(f'bash -c "cat {SCRATCH_DIR}/preprocessed/valid.hi_IN | python3 my_cm_tokenizer.py 0 > {NORM_OUTPUT_DIR}/reference.txt"')

def calc_bleu(outfile):
	output = subprocess.check_output(f'sacrebleu -tok none {OUTPUT_DIR}/reference.txt < {outfile}', shell=True)
	output = output.decode("utf-8")
	return float(output.split('=')[1].strip().split(' ')[0])

def calc_bleu_norm(outfile):
	output = subprocess.check_output(f'sacrebleu -tok none {NORM_OUTPUT_DIR}/reference.txt < {outfile}', shell=True)
	output = output.decode("utf-8")
	return float(output.split('=')[1].strip().split(' ')[0])

print('checkpoint: BLEU, BLUE_normalized')
for file in files:
	checkpoint_name = file.split('/')[-1]
	print(checkpoint_name, end=': ', flush=True)
	os.system("bash -c \"cat " + file + f" | grep -P ^H | sort -V | cut -f 3- | sed 's/\\[hi_IN\\]//g' | python3 my_post_process.py {DATA_DIR} 0 2>/dev/null > {OUTPUT_DIR}/{checkpoint_name}.hyp\"")
	print(calc_bleu(f'{OUTPUT_DIR}/{checkpoint_name}.hyp'), end = ', ')
	os.system("bash -c \"cat " + file + f" | grep -P ^H | sort -V | cut -f 3- | sed 's/\\[hi_IN\\]//g' | python3 my_cm_tokenizer.py 0 2>/dev/null > {NORM_OUTPUT_DIR}/{checkpoint_name}.hyp\"")
	print(calc_bleu_norm(f'{NORM_OUTPUT_DIR}/{checkpoint_name}.hyp'))
