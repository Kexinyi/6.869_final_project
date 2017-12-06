## Combining Visual and Contextual Information for Question Answering

This is the github repo for 6.869 final project (Fall 2017) by Kexin Yi and Aditya Thomas.

### Overview
Visual and contextual question answering as separate task
<div align="center">
  <img src="https://github.com/Kexinyi/6.869_final_project/blob/master/img/Slide1.jpg" width="450px">
</div>

Combining visual and contextual information for question answering
<div align="center">
  <img src="https://github.com/Kexinyi/6.869_final_project/blob/master/img/Slide2.jpg" width="450px">
</div>

### Results on COCO-QA
The following table contains test results from the COCO-QA dataset under different models and contextual input signals.
```
-----------------------------------------------------------------------------------------------
 model                      0 caption                  1 caption                    5 captions
-----------------------------------------------------------------------------------------------
 LSTM                       0.3657                     0.6300                       0.8632
 CNN+LSTM                   0.5167                     0.5942                       0.5879
 CNN+LSTM+SA                0.5869                     0.6462                       0.8240 
-----------------------------------------------------------------------------------------------
```

### Raw data

COCO: 
```
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

COCO API
```
cd data
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
```

COCO-QA
```
cd data
wget http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip
unzip cocoqa-2015-05-17.zip -d cocoqa
```

### Preprocess raw data
Before preprocessing, create a folder for the preprocessed data.
```
mkdir -p data/preprocessed_h5
```
All data should be arranged according to the following structure
```
6.869_final_project
|-- data
.    |-- coco
.    |    |-- train2014
.    |    |-- val2014
     |    |-- annotations
     |-- cocoqa
     |-- cocoapi
     |-- preprocessed_h5
```

To preprocess images (warning: the output h5 file is ~100G in size)
```
python extract_features.py --split train --output_h5_files data/preprocessed_h5/train_features.h5
python extract_features.py --split val --output_h5_files data/preprocessed_h5/val_features.h5
```

To preprocess annotations
```
python preprocess_anns.py --split train
python preprocess_anns.py --split test --input_vocab_json data/preprocessed_h5/vocab_0caps.json
```
One can preprocess caption-concatenated annotations by changing the `--num_captions` argument, such as
```
python preprocess_anns.py --split train --num_captions 5
python preprocess_anns.py --split test --input_vocab_json data/preprocessed_h5/vocab_5caps.json --num_captions 5
```

### Pretrained model
```
mkdir pretrained
cd pretrained
wget https://www.dropbox.com/s/7bmirgfwi785d5l/CNN_LSTM_SA_5caps.zip
unzip CNN_LSTM_SA_5caps.zip
```
To test, go back to root folder and run
```
python test.py \ 
    --load_path pretrained/CNN_LSTM_SA_5caps \
    --test_question_h5 data/preprocessed_h5/test_anns_5caps.h5 \
    --vocab_json data/preprocessed_h5/vocab_5caps.json
```
Similarly for data with other numbers of captions.

### Train your own model
```
mkdir results
python train.py \
   --checkpoint_path results/CNN+LSTM+SA_5caps \
   --model_type CNN+LSTM+SA \
   --train_question_h5 data/preprocessed_h5/train_anns_5caps.h5 \
   --val_question_h5 data/preprocessed_h5/val_anns_5caps.h5 \
   --vocab_json data/preprocessed_h5/vocab_5caps.json
```
