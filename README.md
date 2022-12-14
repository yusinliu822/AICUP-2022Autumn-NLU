# AICUP-2022Autumn-NLU

## Introduction
[AI CUP](https://www.aicup.tw/) 2022/ [自然語言理解的解釋性資訊標記競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/26)

We proposed three methods, `simpleQA`, `divideQA`, and `NER`.

Document: https://drive.google.com/file/d/13T01XyyMi-mwPhd6EhW8X6k_mPMBK9Ox/view?usp=share_link

### Best Score
|         | # | Score    |
| ------- | - | -------- |
| Public  | 6 | 0.835461 |
| Private | 4 | 0.887998 |

## Requirements ( SimpleQA )
- OS: Ubuntu 16.04.7 LTS
- Python version: 3.8.15
- packages: see `requirements.txt`

## Project Folder Description

### Data
- Data provided by [AICUP](https://www.aicup.tw/)

### SimpleQA
- `simpleQA/simpleQA.ipynb` and `simpleQA/simpleQA.ipynb` are the same, but `.ipynb` provide better readabilty.
- Get best score of our three methods.

### DivideQA
- `divideQA/divideQA.ipynb` and `divideQA/divideQA.ipynb` are the same, but `.ipynb` provide better readabilty.
- Using a more complex way than simpleQA to preprocess data.

### NER
- Preprocess data like a NER Task.

## Reference Data (Large File Store in Google Drive)
- [Best SimpleQA model](https://drive.google.com/drive/folders/1-khQXsHEY2KD8kNjzUeWRBBH0t0A1pWw?usp=sharing)
- [Preprocessed Data of DivideQA & NER](https://drive.google.com/drive/folders/1BxLNhE6KNnvOpevThMrtAZEHsiiryf-y?usp=sharing)
