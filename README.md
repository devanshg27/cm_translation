# CoMeT

## Preprocessing

To transliterate the dataset, use `preprocessing/lang_tag_calcs_shared.py`. It uses the fork of csnli available [here](https://github.com/devanshg27/csnli).


For mBART-hien, the Hindi translations of the English sentences are required. To translate the dataset, use `preprocessing/CALCS_Shared_Translation.ipynb`.

The transliterated and translated versions of the [CALCS Shared Task English-Hinglish dataset](https://ritual.uh.edu/lince/datasets) have also been provided.

## Training

Install the dependencies using

```bash
conda env create --file environment.yml
conda activate cmtranslation2
```

Download mBART pre-trained checkpoint:

```bash
wget -c https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
```

Download and extract [IIT Bombay English-Hindi Corpus v3.0](https://www.cfilt.iitb.ac.in/~parallelcorp/iitb_en_hi_parallel/), [Dakshina dataset v1.0](https://github.com/google-research-datasets/dakshina) and [CALCS Shared Task English-Hinglish dataset](https://ritual.uh.edu/lince/datasets) to a directory. Also copy the `preprocessing/processed_data` and `preprocessing/translated_data` to the same directory. The final directory should look like this:

```
.
├── dakshina_dataset_v1.0
│   └── hi
│       └── lexicons
│           ├── hi.translit.sampled.dev.tsv
│           ├── hi.translit.sampled.test.tsv
│           └── hi.translit.sampled.train.tsv
├── iitb_corpus
│   ├── dev_test
│   │   ├── dev.en
│   │   ├── dev.hi
│   │   ├── test.en
│   │   └── test.hi
│   └── parallel
│       ├── IITB.en-hi.en
│       └── IITB.en-hi.hi
├── mt_enghinglish
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
├── processed_data
│   ├── dev.jsonl
│   └── train.jsonl
└── translated_data
    ├── dev.txt
    ├── test.txt
    └── train.txt
```

Finally, train the model:

```bash
bash train.sh <path to mbart.cc25.v2.tar.gz> <temporary directory which will be created> <path to dataset directory> <mBARTen or mBARThien>
```

The checkpoints are stored in the directory `<temporary directory>/checkpoint`.

## Evaluation

Install the dependencies for evaluation(the same as training) and run evaluation:

```bash
conda env create --file eval-environment.yml
conda activate cmtranslation2
bash eval.sh <temporary directory> <path to dataset directory>
```

Install the dependencies and perform post-processing and score calculation:

```bash
conda env create --file scoring-environment.yml
conda activate indictrans
python3 calc_scores.py <temporary directory> <path to dataset directory>
```

To create the file which can be submitted on the [LinCE Benchmark wesbite](https://ritual.uh.edu/lince/home), use `eval/create_submission.py`.