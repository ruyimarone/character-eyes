# Character Eyes

Code for our project analyzing character level taggers. This repository is a **work in progress** but contains some of our code and analysis. 
More will be added soon!

![example activations](images/act_ex.png)

## Contents
- `model.py` - A fully character level tagger model, implemented in [DyNet](http://dynet.io/). It has support for **asymmetric** bi-directional RNNs, which we found had performance effects depending on linguistic properties of the language. 
- Pretrained models for 5 of our 24 languages
- Ready-to-train datasets (from [Univseral Dependencies 2.3](http://universaldependencies.org/)) for all 24 languages
- [This notebook](https://github.com/ruyimarone/character-eyes/blob/master/PDI.ipynb) reproduces some of the figures and charts in our paper. 

## Coming Soon
- Interactive Notebooks - play with character level representations on the fly!
- better dependencies/`requirements.txt`
- Storage size permitting, more pretrained models including asymmetric configurations


Much of the code is modified from [Mimick](https://github.com/yuvalpinter/Mimick), a character level system that can replace OOVs or UNKs with learned representations approximating a closed vocabulary set of word embeddings. 

## Citation format

When using our work, please use the following `.bib` entry:

```
@article{charactereyes,
  title={Character Eyes: Seeing Language through Character-Level Taggers},
  author={Pinter, Yuval and Marone, Marc and Eisenstein, Jacob},
  journal={arXiv preprint arXiv:1903.05041},
  year={2019}
}
```
