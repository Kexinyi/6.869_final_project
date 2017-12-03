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
parser.add_argument('--cocoqa_path', default='../data/cocoqa/')
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--output_vocab_json', default='../data/preprocessed_h5/vocab.json')
parser.add_argument('--output_h5_path', default='../data/preprocessed_h5/')
parser.add_argument('--test_set_size', default=20000)

def main(args):
    assert args.split == 'train' or args.split == 'test'
    print('preprocessing COCO-QA annotations for %s data' % args.split)

    # read in the image indices
    image_idxs = []
    with open('{}/{}/img_ids.txt'.format(args.cocoqa_path, args.split)) as f:
        for line in f:
            image_idxs.append(int(line.strip()))

    # read in the questions
    questions = []
    with open('{}/{}/questions.txt'.format(args.cocoqa_path, args.split)) as f:
        for line in f:
            questions.append(line.strip())

    # read in the asswers
    answers = []
    with open('{}/{}/answers.txt'.format(args.cocoqa_path, args.split)) as f:
        for line in f:
            answers.append(line.strip())

    # build/expand vocabulary
    print('building vocabulary')
    answer_token_to_idx = build_vocab(ans for ans in answers)
    question_token_to_idx = build_vocab(q for q in questions)
    vocab = {
        'question_token_to_idx': question_token_to_idx,
        'answer_token_to_idx': answer_token_to_idx
    }

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
    questions_encoded = []
    answers_encoded = []

    for i, q in enumerate(questions):
        question_tokens = tokenize(q)
        qe = encode(question_tokens, vocab['question_token_to_idx'], allow_unk = 1)    
        questions_encoded.append(qe)
        answers_encoded.append(vocab['answer_token_to_idx'][answers[i]])

    # pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])
    
    # write to output h5 file
    if args.split == 'train':
        output_h5_train = args.output_h5_path + 'train_annotations.h5'
        print('saving %d training questions to %s' % (len(questions_encoded), output_h5_train))
        with h5py.File(output_h5_train, 'w') as f:
            f.create_dataset('questions', data=np.asarray(questions_encoded))
            f.create_dataset('answers', data=np.asarray(answers_encoded))
            f.create_dataset('image_idxs', data=np.asarray(image_idxs))
    else:
        output_h5_test = args.output_h5_path + 'test_annotations.h5'
        print('saving %d test questions to %s' % (args.test_set_size, output_h5_test))
        with h5py.File(output_h5_test, 'w') as f:
            f.create_dataset('questions', data=np.asarray(questions_encoded[:args.test_set_size]))
            f.create_dataset('answers', data=np.asarray(answers_encoded[:args.test_set_size]))
            f.create_dataset('image_idxs', data=np.asarray(image_idxs[:args.test_set_size]))
        output_h5_val = args.output_h5_path + 'val_annotations.h5'
        print('saving %d validation questions to %s' % (len(questions_encoded)-args.test_set_size, output_h5_val))
        with h5py.File(output_h5_val, 'w') as f:
            f.create_dataset('questions', data=np.asarray(questions_encoded[args.test_set_size:]))
            f.create_dataset('answers', data=np.asarray(answers_encoded[args.test_set_size:]))
            f.create_dataset('image_idxs', data=np.asarray(image_idxs[args.test_set_size:]))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
