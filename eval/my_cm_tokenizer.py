import sys
from sys import stdin
from nltk.tokenize.casual import casual_tokenize

IS_TEST_SET=bool(int(sys.argv[1]))

for line in stdin:
	line = casual_tokenize(line.rstrip(), preserve_case=IS_TEST_SET, reduce_len=False, strip_handles=False)
	print(' '.join(line))
