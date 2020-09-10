# InterpretContactMap
Deep learning methods for interpreting protein contact maps

Contribute
---------------------
DeepGRN has been developed and used by the Bioinformatics, Data Mining and Machine Learning Laboratory (BDM)
. Help from every community member is very valuable to make the tool better for everyone.
Checkout the [Lab Page](http://calla.rnet.missouri.edu/cheng/).


## Required software:

```
[PSI-BLAST 2.2.26](https://www.ncbi.nlm.nih.gov/books/NBK131777/) (For generating PSSM sequence profile)
[CCMpred](https://github.com/soedinglab/CCMpred) (For generating pseudo-likelihood maximization)
[Python 3.6.0](https://www.python.org/)
```

## Required Python modules:

```
numpy 1.15.2
Keras 2.1.6
tensorflow-gpu 1.9.0
```

## Feature geraration
Two types of features are required: PSSM sequence profile which can be generated from PSI-BLAST, and PLM (pseudo-likelihood maximization) which can be generated from CCMpred from multiple sequence alignments (MSAs) produced by DeepMSA. The details of how to acquire both features are described in the [DeepDist paper](https://www.biorxiv.org/content/10.1101/2020.03.17.995910v1), which is also developed from the BDM lab. 

The sequence databases used in the DeepMSA homologous sequences search include Uniclust30 (2017-10), Uniref90 (2018-04) and Metaclust50 (2018-01), our in-house customized database which combines Uniref100 (2018-04) and metagenomics sequence databases (2018-04), and NR90 database (2016). Sample features can be found under the  example folder, and users can build both features from their own customized sequence databases.


## Training (train.py)
Train models for protein contact prediction:

```
python train.py [feature_location] [output_dir] [epoch_number] [batch_size] [model_type]
```

model_type can be one of "baselineModel", "regional_attention" or "sequence_attention"


## Predict (predict.py)

```
python predict.py [model_path] [model_type] [epoch_number] [new_data_path] [out_dir]
```






