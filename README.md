<h1 align="center">Defending Against Harmful Supervision Hidden in Benign Samples</h1>

This is the official implementation of our paper **Defending Against Harmful Supervision Hidden in Benign Samples**


## Environment
The required dependencies have been attached in the requirement file:
```bash
pip install -r requirements.txt
```

## Scripts
The scripts for training and evaluation has been attached in [scripts](./scrtips). Please notice, for safety evlaution, key word evluation is only for response generation, the key word ASR is not accurate. The reported results are based on GPT4 evaluated ASR, please run `gpt_4_judge.ipynb` for pure_bad and `gpt_4_judge_beavertails.ipynb` for BeaverTails. For more information, please refer to [Constrained SFT](https://github.com/Unispac/shallow-vs-deep-alignment). You may need a batch API for evaluation. 

## Datasets
Due to license permission, we can't release the modified dataset to public. For Embedded Attack, please refer to 'preprocess.ipynb' to construct the datasets. 

Our implementation is based on the code below. We appreciate their contribution.

- Constrained SFT: https://github.com/Unispac/shallow-vs-deep-alignment
- Safety at one shot: https://github.com/Kevin-Zh-CS/safety-at-one-shot/tree/main
