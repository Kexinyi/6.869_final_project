import torch
from torch.autograd import Variable

import argparse
import json

import utils
from data import ClevrDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', default='exps/CNN+LSTM+SA')
parser.add_argument('--test_question_h5', default='data/preprocessed_h5/test_anns_0caps.h5')
parser.add_argument('--test_features_h5', default='data/preprocessed_h5/val_features.h5')
parser.add_argument('--vocab_json', default='data/preprocessed_h5/vocab_0caps.json')
parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--batch_size', default=64, type=int)

def main(args):
  print('loading trained model from %s' % (args.load_path + '/checkpoint.pt'))
  model, kwargs = utils.load_model(args.load_path + '/checkpoint.pt')
  model.cuda()
  model.eval()

  vocab = utils.load_vocab(args.vocab_json)
  test_loader_kwargs = {
    'question_h5': args.test_question_h5,
    'feature_h5': args.test_features_h5,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'max_samples': None,
    'num_workers': args.loader_num_workers,
  }

  print('loading test data')
  with ClevrDataLoader(**test_loader_kwargs) as test_loader:
    print('%d samples in the test set' % len(test_loader.dataset))

    print('checking test accuracy...')
    acc = check_accuracy(args, model, test_loader)
  print('test accuracy = %.4f' % acc)

  with open(args.load_path + '/checkpoint.pt.json') as f:
    info = json.load(f)

  with open(args.load_path + '/result.txt', 'w') as res:
    res.write('test accuracy: %4f\n' % acc)
    res.write('best val accuracy: %4f\n' % info['best_val_acc'])
    res.write('arguments: \n')
    for k, v in vars(args).items():
      res.write(str(k) + ': ' + str(v) + '\n')


def check_accuracy(args, model, loader):
  num_correct, num_samples = 0, 0
  for batch in loader:
    questions, image_idxs, feats, answers = batch

    questions_var = Variable(questions.cuda(), volatile=True)
    feats_var = Variable(feats.cuda(), volatile=True)
    answers_var = Variable(feats.cuda(), volatile=True)

    scores = model(questions_var, feats_var)

    if scores is not None:
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == answers).sum()
      num_samples += preds.size(0)

  acc = float(num_correct) / num_samples
  return acc

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)