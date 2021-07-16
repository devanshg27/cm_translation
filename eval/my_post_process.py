import jsonlines
import sys
from collections import defaultdict

DATA_DIR=sys.argv[1]
IS_TEST_SET=bool(int(sys.argv[2]))

transliteration_to_rom_dict = defaultdict(lambda: defaultdict(int))

with jsonlines.open(f'{DATA_DIR}/processed_data/train.jsonl') as reader:
	for obj in reader:
		for x in obj['Devanagari_Hinglish']:
			if x[2] == 'hi':
				transliteration_to_rom_dict[x[1]][x[0]] += 1

transliteration_to_rom_best = {}

for key, value in transliteration_to_rom_dict.items():
	transliteration_to_rom_best[key] = max(value.items(), key=lambda x: x[1])[0]

import csv
import glob

dd_files = glob.glob(f"{DATA_DIR}/dakshina_dataset_v1.0/hi/lexicons/*.tsv")
dd_transliteration_to_rom_dict = defaultdict(lambda: defaultdict(int))

for dd_file in dd_files:
	with open(dd_file) as fdd:
		for row in csv.reader(fdd, delimiter="\t", quotechar=None, quoting=csv.QUOTE_NONE):
			dd_transliteration_to_rom_dict[row[0]][row[1]] += int(row[2])

dd_transliteration_to_rom_best = {}

for key, value in dd_transliteration_to_rom_dict.items():
	dd_transliteration_to_rom_best[key] = max(value.items(), key=lambda x: x[1])[0]

from indictrans import Transliterator
trn = Transliterator(source='hin', target='eng', build_lookup=True, rb=False)

from sys import stdin, stderr
from nltk.tokenize.casual import casual_tokenize

def is_ascii(s):
	return all(ord(c) < 128 for c in s)

def is_english_char(s):
	return s.isalpha() and is_ascii(s)

def is_hindi_char(s):
	return int(0x900) <= ord(s) <= int(0x97F)

def filter_mixed_script(l):
	newl = []
	for word in l:
		if all(map(is_hindi_char, word)):
			newl.append(word)
		elif any(map(is_hindi_char, word)):
			# filter words with both roman and devanagari scripts
			pass
		else:
			newl.append(word)
	return newl

for line in stdin:
	line = casual_tokenize(line.rstrip(), preserve_case=IS_TEST_SET, reduce_len=False, strip_handles=False)
	line = filter_mixed_script(line)
	for idx, word in enumerate(line):
		if all(map(is_hindi_char, word)):
			if word in transliteration_to_rom_best:
				line[idx] = transliteration_to_rom_best[word]
			elif word in dd_transliteration_to_rom_best:
				print('Using dakshina_dataset for the following word: ', word, trn.transform(word), file=stderr)
				line[idx] = dd_transliteration_to_rom_best[word]
			else:
				print('Using indictrans for the following word: ', word, trn.transform(word), file=stderr)
				line[idx] = trn.transform(word)
		elif any(map(is_hindi_char, word)):
			assert(False)
			print('Mixed English and Hindi characters: ', word, file=stderr)
	print(' '.join(line))
