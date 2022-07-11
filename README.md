## Predicting gene expression using millions of random promoter sequences

competition website:
https://www.synapse.org/#!Synapse:syn28469146/wiki/

### Process
1. Generate dataset [EDA_prep](EDA_prep.ipynb)
2. A naive LSTM model is tested on a random 4:1 train/val split on full dataset
   1. model -> simple LSTM with simple embedding dimension (6, 100) for A,G,C,T,N(unknown) + PAD, where PAD is a PAD placeholder to pad all sequence to 150(check [EDA_prep](EDA_prep.ipynb) for detail)
   2. batch_size = 512
3. Performance is documented in [log_naive_lstm.txt](log_naive_lstm.txt)
    - Pearson's R = 0.73
    - Spearman's R = 0.75