import os
import sys

assert len(sys.argv) == 2

DATA_DIR = sys.argv[1]
filename = sys.argv[2] # path to TEST_OUTPUT_DIR_TEMP/<best val checkpoint>.hyp

os.system("bash -c \"cat " + filename + " | grep -P ^H | sort -V | cut -f 3- | sed 's/\\[hi_IN\\]//g' | python3 my_post_process.py " + DATA_DIR + " 1 2>/dev/null > ./mt_eng_hinglish.txt\"")
# OR
os.system("bash -c \"cat " + filename + " | grep -P ^H | sort -V | cut -f 3- | sed 's/\\[hi_IN\\]//g' | python3 my_cm_tokenizer.py 1 2>/dev/null > ./mt_eng_hinglish.txt\"")

os.system("zip submission.zip mt_eng_hinglish.txt")