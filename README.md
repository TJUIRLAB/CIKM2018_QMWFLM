<!--
 * @Author: your name
 * @Date: 2020-01-20 14:10:46
 * @LastEditTime : 2020-01-21 15:20:24
 * @LastEditors  : Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /CIKM2018_QMWFLM/README.md
 -->
 # QMWF-LM
This is the code of the paper:

[A Quantum Many-body wave function Inspired Language Modeling Approach](https://arxiv.org/abs/1808.09891)

## DEPENDENCIES

- python 2.7+
- numpy
- tensorflow 1.2+
- scikit-learn (sklearn)
- pandas

This model is for TREC-QA dataset and WIKI-QA dataset. the pretrained embedding we used is [glove](https://nlp.stanford.edu/projects/glove/)


## RUN

You can run this model by:

```
python train.py --data trec --clean False 
```

```
python train.py --data wiki --clean True
```

## REFERENCES

If you find this code is useful, please consider citing our work.

```
@inproceedings{zhang2018quantum,
  title={A quantum many-body wave function inspired language modeling approach},
  author={Zhang, Peng and Su, Zhan and Zhang, Lipeng and Wang, Benyou and Song, Dawei},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={1303--1312},
  year={2018},
  organization={ACM}
}

```

## Contributor

-   [@ZhanSu](https://github.com/shuishen112)

