import jsonlines
import csv
from three_step_decoding import *
from nltk.tokenize.casual import casual_tokenize

tsd = ThreeStepDecoding('lid_models/hinglish',
						htrans='nmt_models/rom2hin.pt',
						etrans='nmt_models/eng2eng.pt')

dataset = []
dataset_t = []

for file in ["dev", "train"]:
	idx = 0
	with open(f'/home/devanshg27/calcs_shared/mt_enghinglish/{file}.txt', 'r') as f, open(f'/home/devanshg27/calcs_shared/google_translate/{file}.txt', 'r') as f_translated:
		with jsonlines.open(f'/home/devanshg27/calcs_shared/processed_data/{file}.jsonl', mode='w', flush=True) as writer:
			cf = csv.reader(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONE, escapechar=None)
			hindi_sentences = f_translated.readlines()
			for row in cf:
				assert(len(row) == 2)
				if idx % 10 == 0:
					print(f"{idx}/{len(hindi_sentences)} completed")
				english_sentence = row[0].strip()
				hindi_sentence = hindi_sentences[idx].strip()
				roman_hinglish_sentence = row[1].strip()
				english_sentence = casual_tokenize(english_sentence, preserve_case=True, reduce_len=False, strip_handles=False)
				hindi_sentence = casual_tokenize(hindi_sentence, preserve_case=True, reduce_len=False, strip_handles=False)
				roman_hinglish_sentence = casual_tokenize(roman_hinglish_sentence, preserve_case=True, reduce_len=False, strip_handles=False)
				
				isSingleWord = False
				if len(roman_hinglish_sentence) == 1:
					isSingleWord = True
					roman_hinglish_sentence.append('.')
				devanagari_hinglish_sentence = list(tsd.tag_sent(' '.join(roman_hinglish_sentence)))
				if isSingleWord:
					devanagari_hinglish_sentence.pop()
					roman_hinglish_sentence.pop()
				writer.write({
						"English": english_sentence,
						"Hindi": hindi_sentence,
						"Roman_Hinglish": roman_hinglish_sentence,
						"Devanagari_Hinglish": devanagari_hinglish_sentence,
					}
				)

				idx += 1
			assert(len(hindi_sentences) == idx)
