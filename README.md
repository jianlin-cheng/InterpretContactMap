# InterpretContactMap
Deep learning methods for interpreting protein contact maps

Contribute
---------------------
DeepGRN has been developed and used by the Bioinformatics, Data Mining and Machine Learning Laboratory (BDM)
. Help from every community member is very valuable to make the tool better for everyone.
Checkout the [Lab Page](http://calla.rnet.missouri.edu/cheng/).


## Required software:

[PSI-BLAST 2.2.26](https://www.ncbi.nlm.nih.gov/books/NBK131777/) (For generating PSSM sequence profile)

[CCMpred](https://github.com/soedinglab/CCMpred) (For generating pseudo-likelihood maximization)

[Python 3.6.5](https://www.python.org/)

CUDA-9.0.176

cuDNN-7.1.4

## Required Python modules:

```
numpy 1.19.2
pandas 1.1.2
tensorflow-gpu 1.9.0
keras 2.1.6
h5py 2.8.0
```

## Feature geraration
Two types of features are required: PSSM sequence profile which can be generated from PSI-BLAST, and PLM (pseudo-likelihood maximization) which can be generated from CCMpred from multiple sequence alignments (MSAs) produced by DeepMSA. The details of how to acquire both features are described in the [DeepDist paper](https://www.biorxiv.org/content/10.1101/2020.03.17.995910v1), which is also developed by the BDM lab. 

The sequence databases used in the DeepMSA homologous sequences search include Uniclust30 (2017-10), Uniref90 (2018-04) and Metaclust50 (2018-01), our in-house customized database which combines Uniref100 (2018-04) and metagenomics sequence databases (2018-04), and NR90 database (2016). Sample features can be found under the  example folder, and users can build both features from their own customized sequence databases.


## Prediction

Predict from given PLM and PSSM data (predict.py):
  * `-h, --help`            show this help message and exit
  * `-m, --model_type`      Type of model, can be one of sequence_attention, regional_attention or combined
  * `-l, --plm_data`        Path to PLM data. Should be a numpy array flatten from (441,L,L), where L is the length of the input sequence. It should be saved as .npy format (https://numpy.org/doc/stable/reference/generated/numpy.save.html).
  * `-s, --pssm_data`       Path to PSSM data. Should be a text file start with " # PSSM" as the first line, and the following contents should be 20 lines each contains L values.
  * `-o, --out_file`        Path to output contact map. An L by L numeric matrix saved as TSV format.


Example:

```
python predict.py -m sequence_attention -l example/plm/T0970.plm -s example/other/X-T0970.txt -o outmap.tsv
```
                        
                        

## Training

Train models for protein contact prediction with given PLM, PSSM and labels for training and validation datasets (train.py):
  * `-h, --help`                 show this help message and exit.
  * `-plm, --plm_data_path`      Path to PLM data files.
  * `-pssm, --pssm_data_path`    Path to PSSM data files.
  * `-l, --label_path`           Path to label data files. Should be text files with 0/1 indicate the contacts.
  * `-s, --sample_list_file`     Config file indicating the sample names for training and validation.

  * `-m, --model_type`           Type of model, can be one of "sequence_attention", "regional_attention".
  * `-o, --output_dir`           Path where the trained models and history are saved.
  * `-pa, --patience`            Stop the training early for no improvements in validation after x epochs, default is 5.
  * `-e, --epochs`               Number of epochs, default is 60.

Example:

```
python train.py -m sequence_attention -plm example/plm/ -pssm example/other/ -l example/bin_class/ -s example/train_sample_list.txt -o ./
```