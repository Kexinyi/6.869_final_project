import sys
import os
import argparse
import json
import os
import h5py
import numpy as np

from preprocess import build_vocab, tokenize, encode
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True)
parser.add_argument('--vqa_path', default='../data/VQA/')
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--output_vocab_json', default='../data/vqa_h5/vocab.json')
parser.add_argument('--output_h5_path', default='../data/vqa_h5/')

def main(args):
  assert args.split == 'train' or args.split == 'val' or args.split == 'test'

  print('preprocessing VQA annotations for %s data' % args.split)

  # read the VQA questions
  with open(args.vqa_path + 'v2_OpenEnded_mscoco_%s2014_questions.json' % args.split) as f: 
    vqa_q = json.load(f)
    vqa_questions = vqa_q['questions']

  # read the VQA answers (present in the annotations)  
  if args.split != 'test':  
    with open(args.vqa_path + 'v2_mscoco_%s2014_annotations.json' % args.split) as f:
      vqa_ann = json.load(f)  
      vqa_annotations = vqa_ann['annotations']

  # build/expand vocabulary
  print('building vocabulary')
  if args.split != 'test':
    answer_token_to_idx = build_vocab((ann['multiple_choice_answer'] for ann in vqa_annotations))
  else:
    answer_token_to_idx = {}
  question_token_to_idx = build_vocab((q['question'] for q in vqa_questions), punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
  vocab = {'question_token_to_idx': question_token_to_idx, 'answer_token_to_idx': answer_token_to_idx}

  if args.input_vocab_json != '':
    print('expanding vocabulary')
    new_vocab = vocab

    with open(args.input_vocab_json, 'r') as f:
      vocab = json.load(f)

    num_new_words_q = 0
    for word in new_vocab['question_token_to_idx']:
      if word not in vocab['question_token_to_idx']:
        # print('found new word %s in question' % word)
        idx = len(vocab['question_token_to_idx'])
        vocab['question_token_to_idx'][word] = idx
        num_new_words_q += 1
    print('found %d new words in questions' % num_new_words_q)

    if args.split != 'test':
      num_new_words_a = 0
      for word in new_vocab['answer_token_to_idx']:
        if word not in vocab['answer_token_to_idx']:
          # print('found new word %s in answers' % word)
          idx = len(vocab['answer_token_to_idx'])
          vocab['answer_token_to_idx'][word] = idx
          num_new_words_a += 1
      print('found %d new words in answers' % num_new_words_a)

  print('%d question tokens in new vocab' % len(vocab['question_token_to_idx']))
  print('%d answer tokens in new vocab' % len(vocab['answer_token_to_idx']))

  if args.output_vocab_json != '':
    print('saving vocabulary to %s' % args.output_vocab_json)
    with open(args.output_vocab_json, 'w') as f:
      json.dump(vocab, f)

  # encode questions and answers
  print('encoding annotations')
  vqa_questions_encoded = []
  vqa_orig_idxs = []
  vqa_image_idxs = []
  vqa_answers_encoded = []
  vqa_captions = []

  for orig_idxs, q in enumerate(vqa_questions):
    vqa_orig_idxs.append(orig_idxs)
    vqa_image_idxs.append(q['image_id'])    

    question = q['question']
    question_tokens = tokenize(question, punct_to_keep = [';', ','], punct_to_remove = ['?', '.'])
    question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk = 1)    
    vqa_questions_encoded.append(question_encoded)
      
    if args.split != 'test':
      answer = vqa_annotations[orig_idxs]['multiple_choice_answer']
      answer_tokens = tokenize(answer)
      answer_encoded = encode(answer_tokens, vocab['answer_token_to_idx'], allow_unk = 1)
      vqa_answers_encoded.append(answer_encoded)

  # pad encoded questions
  max_question_length = max(len(x) for x in vqa_questions_encoded)
  for qe in vqa_questions_encoded:
    while len(qe) < max_question_length:
      qe.append(vocab['question_token_to_idx']['<NULL>'])
    
  # pad encoded answers
  if args.split != 'test':
    max_answer_length = max(len(x) for x in vqa_answers_encoded)
    for ae in vqa_answers_encoded:
      while len(ae) < max_answer_length:
        ae.append(vocab['answer_token_to_idx']['<NULL>'])

  # write to output h5 file
  output_h5_file = args.output_h5_path + '%s_annotations.h5' % args.split
  print('saving annotations to %s' % output_h5_file)
  vqa_questions_encoded = np.asarray(vqa_questions_encoded, dtype=np.int32)
  vqa_answers_encoded = np.asarray(vqa_answers_encoded, dtype=np.int32)
  with h5py.File(output_h5_file, 'w') as f:
    f.create_dataset('questions', data=vqa_questions_encoded)
    f.create_dataset('answers', data=vqa_answers_encoded)
    f.create_dataset('image_idxs', data=np.asarray(vqa_image_idxs))
    f.create_dataset('orig_idxs', data=np.asarray(vqa_orig_idxs))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
