# Goals

In this repository, we will see different methods to achieve similarity search across multiple languages. 
We aim to compare: 
* LASER (Baseline) 
* XLM MLM 100 languages
* XLM-R 
* sentence transformers with multilingual models 

We compare the below models on the publicly available newstest2012 from WMT 2012. 

For each sentence in the source language, we calculate the closest sentence in the joint embedding space in the target language. If this sentence has the same index in the file, it is considered as correct, and as an error else wise. Therefore, the N-way parallel corpus should not contain duplicates.

1. First, get the data from the section "Get data".  
2. Second, install the necessary tools such as tokenizers and BPE encoder.   
3. Third, preprocess the data (this step is not necessary for all models but interesting for the seek of experimenting).   
4. Then, get sentence embeddings with the method of your choice. In this section, we have not optimized the code so it can be very slow for XLM models. It is quite fast with sentence-transformers though. It could be useful to improve this part to get sentence embeddings faster ==> We could use larger batches (batch version is available but needs to be optimized). Feel free to raise an issue with suggestions. 
5. Finally, compute similarity search for each pair of languages.   

# Results 

## Baseline (LASER)
|     |   cs  |   de  |   en  |   es  |   fr   |  avg  |  
|-----|-------|-------|-------|--------|-------|-------|
| cs  | 0.00% | 0.70% | 0.90% | 0.67%  | 0.77% | 0.76% |
| de  | 0.83% | 0.00% | 1.17% | 0.90%  | 1.03% | 0.98% |
| en  | 0.93% | 1.27% | 0.00% | 0.83%  | 1.07% | 1.02% |
| es  | 0.53% | 0.77% | 0.97% | 0.00%  | 0.57% | 0.71% |
| fr  | 0.50% | 0.90% | 1.13% | 0.60%  | 0.00% | 0.78% |
| avg | 0.70% | 0.91% | 1.04% | 0.75%  | 0.86% | 1.06% |

## Sentence-transformers with 'distiluse-base-multilingual-cased'

Czech language is currently not covered by sentence-transformers. More experiments are coming to do a thorough comparison. 
Hence, the following is a better view of what performance can achieve sentence-transformers using as base model 'distiluse-base-multilingual-cased'

|     |   cs  |   de  |   en  |   es  |   fr   |  avg  |  
|-----|-------|-------|-------|--------|-------|-------|
| cs  | NA | NA | NA | NA  | NA | NA |
| de  | NA | 0.00% | 1.40%  | 1.60%  | 1.53% | TODO |
| en  |NA | 1.47% | 0.00% | 1.33%   | 0.97% | TODO |
| es  | NA |  1.80%  | 1.10% | 0.00%  | 1.03% | TODO |
| fr  | NA|  1.73%   |0.97%  | 1.07%  | 0.00% | TODO |
| avg | NA | TODO | TODO | TODO  | TODO | TODO |

<b> Sentence-transformers outperform LASER on the pair (French, English), (English, French).   
Otherwise, LASER shows higher performance. </b>

Russian is considered in our analysis. LASER has not published in their repo the results for Russian. We will reproduce their experiments soon and include Russian for rigurous analysis. 

|langs  | de |       en  |     es     |   fr    |    ru    |    avg     | 
|-----|-------|-------|-------|--------|-------|-------|
| de |    0.00%  |  1.40%   | 1.60%  |  1.53%  |  2.60%  |  1.78% |
| en    | 1.47% |  0.00%  |  1.33%  |  0.97%   | 0.50%   | 1.07% |
| es  |   1.80%  |  1.10%   | 0.00%  | 1.03%  |  2.13%  |  1.52% |
| fr  |   1.73%  |  0.97%  |  1.07%   | 0.00%  |  2.00%  |  1.44% |
| ru   |  2.76%   | 0.50%   | 2.23%  |  2.03%  |  0.00%  |   1.88% |
| avg   | 1.94%  |  0.99%  |  1.56%  |  1.39%  |  1.81%  |  1.54% | 

 
  
  
We can also do a Zero-Shot encoding and consider CS language.  
The following shows such results: 

|langs  | cs  |   de    |   en   |   es |   fr   |    ru   |  avg   | 
|-----|-------|-------|-------|--------|-------|-------|-------|
| cs  |   0.00%  | 39.46%  | 37.63%  | 38.20%  | 38.99%  | 41.69%  | 39.19% |
| de  |  36.10%  |  0.00%  |  1.40%  |  1.60%  |  1.53%  |  2.60%  |  8.64% |
| en  |  34.17%  |  1.47%  |  0.00%  |  1.33%  |  0.97%  |  0.50%  |  7.69% |
| es  | 35.36%  |  1.80%  |  1.10%  |  0.00%  |  1.03%  |  2.13%  |  8.29% |
| fr  |  35.96%  |  1.73%  |  0.97%  |  1.07%  |  0.00%  |  2.00%  |  8.34% |
| ru  |  37.96%  |  2.76%  |  0.50%  |  2.23%  |  2.03%   | 0.00%  |  9.10% |
| avg  | 35.91%   | 9.44%  |  8.32%  |  8.88%  |  8.91%   | 9.78% |  13.54% | 

## DistilBERT-base-multilingual-cased

With mean pooling, MAX_LEN = 100:

```
Confusion matrix:
langs   cs       de       en       es       fr       avg     
cs     0.00%   84.85%   87.71%   86.85%   88.71%   87.03%
de    75.06%    0.00%   64.20%   70.36%   69.13%   69.69%
en    74.16%   54.28%    0.00%   24.84%   29.14%   45.60%
es    75.99%   81.62%   40.86%    0.00%   50.95%   62.35%
fr    90.08%   78.16%   37.53%   56.88%    0.00%   65.66%
avg   78.82%   74.73%   57.58%   59.73%   59.48%   66.07%
```


With mean pooling, MAX_LEN = 50: 
```
Confusion matrix:
langs   cs       de       en       es       fr       avg     
cs     0.00%   84.18%   85.98%   82.65%   89.08%   85.47%
de    74.79%    0.00%   61.51%   65.23%   67.27%   67.20%
en    75.72%   55.04%    0.00%   26.71%   31.60%   47.27%
es    78.39%   81.75%   37.40%    0.00%   55.54%   63.27%
fr    75.36%   63.20%   35.16%   34.33%    0.00%   52.01%
avg   76.07%   71.05%   55.01%   52.23%   60.87%   63.05%
```


With cls pooling, MAX_LEN = 100: 
```
Confusion matrix:
langs   cs       de       en       es       fr       avg     
cs     0.00%   89.84%   98.07%   89.88%   91.94%   92.43%
de    72.86%    0.00%   88.68%   71.33%   73.63%   76.62%
en    81.55%   64.47%    0.00%   42.99%   51.85%   60.21%
es    77.46%   78.82%   85.55%    0.00%   60.11%   75.48%
fr    75.42%   76.76%   83.82%   50.18%    0.00%   71.55%
avg   76.82%   77.47%   89.03%   63.59%   69.38%   75.26%
```
With cls pooling, MAX_LEN = 50: 
```
Confusion matrix:
langs   cs       de       en       es       fr       avg     
cs     0.00%   91.51%   98.30%   91.04%   93.81%   93.66%
de    75.02%    0.00%   89.68%   73.16%   76.92%   78.70%
en    82.78%   67.87%    0.00%   49.32%   56.51%   64.12%
es    78.69%   81.68%   86.38%    0.00%   64.84%   77.90%
fr    76.72%   80.22%   85.18%   54.18%    0.00%   74.08%
avg   78.31%   80.32%   89.89%   66.92%   73.02%   77.69%
```


## XLM with MLM covering 100 languages

With max_len = 100, (the higher the better according to experiments) 

langs |  cs   |    de   |    en    |   es   |    fr   |    ru  |     avg |
|-----|-------|-------|-------|--------|-------|-------|-------|
|cs  |   0.00%  | 74.46% |  74.63% |  71.86% |  47.45% |  42.72% |  62.22%|
|de  |  48.42%  |  0.00%  | 32.83% |  35.23% |  35.30%  | 40.93%  | 38.54%|
|en  |  53.28% |  37.70%  |  0.00% | 22.91% |  25.57%  | 24.18% |  32.73%|
|es  |  40.26% |  48.68%  | 19.35%  |  0.00%  | 25.87%  | 36.56% |  34.15%|
|fr  |  65.73%  | 50.35%  | 20.85%  | 37.30%  |  0.00%  | 31.73% |  41.19%|
|ru  |  41.56%  | 42.09%  | 21.51%  | 34.23%  | 34.73%  |  0.00% |  34.83%|
|avg |  49.85%  | 50.66%  | 33.83%  | 40.31% |  33.79%  | 35.22%  | 40.61%|


Without russian, (for easier comparison with Baseline) 

|langs  | cs      | de    |   en    |   es   |    fr     |  avg |
|-----|-------|-------|-------|--------|-------|-------|
|cs   |  0.00%  | 74.46%  | 74.63%  | 71.86%  | 47.45%  | 67.10%|
|de   | 48.42%  |  0.00%  | 32.83%  | 35.23%  | 35.30%  | 37.95%|
|en  |  53.28%  | 37.70%  |  0.00%  | 22.91%  | 25.57%  | 34.87%|
|es  |  40.26%  | 48.68%  | 19.35%  |  0.00%  | 25.87%  | 33.54%|
|fr  |  65.73%  | 50.35%  | 20.85%  | 37.30%  |  0.00%  | 43.56%|
|avg  |  51.92%  | 52.80%  | 36.91%  | 41.82%  | 33.55%  | 43.40%|


## XLM-RoBERTa 

@TODO: CHECK WHY XLM-R shows such poor performance

Mean Pooling Strategy is the strategy with the best performance. Refer to [this issue](https://github.com/MastafaF/multilingual_similarity_compare/issues/8) for comparison when using CLS Pooling Strategy. 
Input: MAX_LEN = 40, Mean Pooling Strategy 
```bash
sh similarity_XLM-R_batch.sh 40 mean True 
```

Output: 
```
Confusion matrix:
langs   cs       de       en       es       fr       avg     
cs     0.00%   91.31%   97.64%   98.37%   94.34%   95.41%
de    93.84%    0.00%   88.05%   92.87%   95.40%   92.54%
en    91.21%   77.79%    0.00%   72.56%   94.77%   84.08%
es    95.47%   93.14%   61.84%    0.00%   90.28%   85.18%
fr    91.97%   81.25%   77.12%   71.96%    0.00%   80.58%
avg   93.12%   85.87%   81.16%   83.94%   93.70%   87.56%
```

Input: MAX_LEN = 100, Mean Pooling Strategy 
```bash
sh similarity_XLM-R_batch.sh 100 mean True 
```

Output: 
```
Confusion matrix:
langs   cs       de       en       es       fr       avg     
cs     0.00%   91.24%   97.77%   97.80%   92.11%   94.73%
de    94.31%    0.00%   86.55%   92.44%   94.67%   91.99%
en    91.08%   74.93%    0.00%   71.56%   77.69%   78.81%
es    95.37%   88.38%   58.54%    0.00%   90.38%   83.17%
fr    92.71%   91.58%   87.41%   89.14%    0.00%   90.21%
avg   93.36%   86.53%   82.57%   87.74%   88.71%   87.78%
```

In the following, we show how to replicate such results. 
# General 

## Get data

For linux users: 
Use script wmt.sh 
```
sh wmt.sh 
```


## Install tools
We assume that tools like torch, tqdm, etc are already installed. On Google Collab, it is the case. (cf. examples --coming soon)  
Use script install-tools.sh 

```
sh install-tools.sh 
```


## Preprocess data (not necessary) 

When preprocessing data prior to encoding, we experienced lower performance. Hence, we do not recommend to do it. If one still wants to preprocess data, please use prepare-data-wmt bash file. Then, some small changes need to be added, so that input file can be found in the proper directory. We may add an argument for --input_file_name in the future. 

Use script prepare-data-wmt.sh
```
sh prepare-data-wmt.sh
```

# Replicate Results: Sentence-transformers
At the moment,  sentence-transformers on "distiluse-base-multilingual-cased" only covers Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish.   
Hence, Czech is not covered. We still decide to print the errors for language Czech for the seek of experiment on pure Zero-Shot learning. However, for comparison with baseline, we focus on other pairs of languages.

## Run similarity search 
```
sh similarity_sentenceBERT.sh
```

# Replicate Results: XLM 
## Run similarity search 
In this section, we still need to update source code to make it faster. At the moment, we iterate over each line and build encoding on the fly which takes too much time.

MAX_LEN is an integer parameter that is describing the number of tokens to consider when encoding.   
For each sentence with less tokens than MAX_LEN, we use zero-padding.   
For each sentence with more token than MAX_LEN, we ignore all tokens after MAX_LEN index.  
Default value is 40. Based on prior statistics on preprocessed data for French, we have: 

* mean_value of length sentence: 26.6
* std_value of length sentence: 15.4
* max_value of length sentence: 145

We recommend to increase MAX_LEN for experiments. Larger MAX_LEN gives better performance but slower computation. 
```
sh similarity_XLM.sh MAX_LEN
```

# Replicate Results: XLM-R 
## Run similarity search 
In this section, we still need to update source code to make it faster. At the moment, we iterate over each line and build encoding on the fly which takes too much time.

Parameters: 
* --max_len: maximum length of the sentences 
* --pooling_strat: pooling strategy in the set {cls, mean} at the moment. Needs to be optimized. Default value is cls embedding.
* --gpu: Use GPU if --gpu set to True and GPU support in your machine 
  
```
sh similarity_XLM-R.sh MAX_LEN POOLING_STRAT
```

Batch version: 
```
sh similarity_XLM-R_batch.sh MAX_LEN POOLING_STRAT GPU
```
