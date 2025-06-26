This is the code associated with the submission "Diffusion Model with Selective Attention for Temporal Knowledge Graph Reasoning" under review at ECML-PKDD 2025.

## Datasets

All the processed datasets we used in the paper can be downloaded at [Baidu Yun](https://pan.baidu.com/s/1Yx3n1tUvQeviKY1OttYP8Q?pwd=6cha)(password:6cha). Put datasets in the folder 'data' to run experimments.


Requrienments


## Run scripts
```{bash}
python main.py -d ICEWS14 --history-len 4 --lambdax 3.0 --graph-layer 2 --use-valid False --max-epochs 20 --timestamps 365
```

```{bash}
python main.py -d ICEWS18 --history-len 4 --lambdax 2.0 --graph-layer 2 --use-valid True --max-epochs 20 --timestamps 304
```


## Acknowledge
Some of our code is also referenced from PLEASING, and the original dataset can be found here: [PLEASING](https://github.com/KcAcoZhang/PLEASING).
And RE-GCN: [DiffuTKG](https://github.com/AONE-NLP/DiffuTKG)
