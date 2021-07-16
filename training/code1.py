from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import os
import sys
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
import jsonlines

BASE_DIR = f'{sys.argv[1]}/preprocessed/'
DATA_DIR = f'{sys.argv[2]}/'
MODEL = sys.argv[3]

assert MODEL == 'mBARTen' or MODEL == 'mBARThien', "MODEL should be 'mBARTen' or 'mBARThien'"

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

def parse_iitb_file(file_en, file_hi, data_id):
    normalizer = DevanagariNormalizer()
    en_data = []
    hi_data = []
    with open(file_en) as f2, open(file_hi) as f1:
        for src, tgt in zip(f1, f2):
            hi_data.append(src.strip() + '\n')
            en_data.append(tgt.strip() + '\n')
    for i in range(len(hi_data)):
        hi_data[i] = normalizer.normalize(hi_data[i])
    assert len(en_data) == len(hi_data)
    print(f'total size of {data_id} data is {len(en_data)}')
    return en_data, hi_data

# Datasets

iitb_en_train, iitb_hi_train = parse_iitb_file(DATA_DIR+'iitb_corpus/parallel/IITB.en-hi.en', DATA_DIR+'iitb_corpus/parallel/IITB.en-hi.hi', 'IITB_TRAIN')
iitb_en_val, iitb_hi_val = parse_iitb_file(DATA_DIR+'iitb_corpus/dev_test/dev.en', DATA_DIR+'iitb_corpus/dev_test/dev.hi', 'IITB_VALIDATION')
iitb_en_test, iitb_hi_test = parse_iitb_file(DATA_DIR+'iitb_corpus/dev_test/test.en', DATA_DIR+'iitb_corpus/dev_test/test.hi', 'IITB_TEST')

def parse_shared(file, data_id):
    src_data = []
    tgt_data = []
    with jsonlines.open(f'{DATA_DIR}/processed_data/{file}.jsonl') as reader:
        for obj in reader:
            if MODEL == 'mBARTen':
                src_data.append(' '.join(obj['English']) + '\n')
            elif MODEL == 'mBARThien':
                src_data.append(' '.join(obj['Hindi']) + ' ## ' + ' '.join(obj['English']) + '\n')
            tgt_data.append(' '.join([(x[1] if x[2] == 'hi' else x[1]) for x in obj['Devanagari_Hinglish']]) + '\n')
    print(f'total size of {data_id} data is {len(src_data)}')
    return src_data, tgt_data

calcs_src_train, calcs_tgt_train = parse_shared("train", "CALCS_TRAIN")
calcs_src_val, calcs_tgt_val = parse_shared("dev", "CALCS_VALIDATION")

def parse_shared_test(data_id):
    src_data = []
    arr1 = []
    arr2 = []
    with open(DATA_DIR+'mt_enghinglish/test.txt', 'r') as f, open(DATA_DIR+'translated_data/test.txt', 'r') as f_translated:
        for row in f:
            english_sentence = row.strip()
            arr1.append(english_sentence)
        for row in f_translated:
            hindi_sentence = row.strip()
            arr2.append(hindi_sentence)
    assert len(arr1) == len(arr2)
    if MODEL == 'mBARTen':
        src_data = [arr1[i] + '\n' for i in range(len(arr1))]
    elif MODEL == 'mBARThien':
        src_data = [arr2[i] + ' ## ' + arr1[i] + '\n' for i in range(len(arr1))]
    print(f'total size of {data_id} data is {len(src_data)}')
    return src_data, src_data

calcs_src_test, calcs_tgt_test = parse_shared_test("CALCS_TEST")

file_mapping = {
    'train.en_XX': calcs_src_train,
    'train.hi_IN': calcs_tgt_train,
    'valid.en_XX': calcs_src_val,
    'valid.hi_IN': calcs_tgt_val,
    'test.en_XX': calcs_src_test,
    'test.hi_IN': calcs_tgt_test,
    'iitb.en_XX': iitb_en_train + iitb_en_val + iitb_en_test,
    'iitb.hi_IN': iitb_hi_train + iitb_hi_val + iitb_hi_test,
}

for k, v in file_mapping.items():
    with open(f'{BASE_DIR}{k}', 'w') as fp:
        fp.writelines(v)