python preprocess_annotations.py --split train
python preprocess_annotations.py --split val --input_vocab_json ../data/vqa_h5/vocab.json
python preprocess_annotations.py --split test --input_vocab_json ../data/vqa_h5/vocab.json