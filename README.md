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

With max_len = 40, 

|langs |  cs    |   de      | en       |es      | fr    |   ru     |  avg|  
|-----|-------|-------|-------|--------|-------|-------|-------|
|cs   |  0.00%  | 95.97%  | 96.47%  | 95.37%  | 96.30%  | 96.60%  | 96.14%|
|de   | 95.64%  |  0.00%  | 96.50%  | 96.80%  | 96.77%  | 98.04%  | 96.75%|
|en   | 95.44%  | 95.64%  |  0.00%  | 89.34%  | 91.77%  | 94.37%  | 93.31%|
|es   | 94.57%  | 96.27%  | 89.34%  |  0.00%  | 88.05%  | 95.87%  | 92.82%|
|fr   | 95.84%  | 95.90%  | 92.67%  | 88.41%  |  0.00%  | 96.97%  | 93.96%|
|ru   | 95.77%  | 97.50%  | 94.61%  | 96.27%  | 97.44%  |  0.00%  | 96.32%|
|avg  | 95.45%  | 96.26%  | 93.92%  | 93.24%  | 94.07%  | 96.37%  | 94.88%|


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
 
```
sh similarity_XLM-R.sh MAX_LEN POOLING_STRAT
```

Batch version: 
```
sh similarity_XLM-R_batch.sh MAX_LEN POOLING_STRAT
```
