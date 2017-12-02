'''
Preprocess coco images
Model: resnet101 (final fc layer removed)
'''

import sys
sys.path.append('/data/vision/billf/jwu-phys/prog/kexin/repos/cocoapi/PythonAPI')
from pycocotools.coco import COCO

import argparse, os, json
import h5py
import numpy as np
from scipy.misc import imread, imresize

import torch
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)


def build_model():
  cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
  layers = [
    cnn.conv1,
    cnn.bn1,
    cnn.relu,
    cnn.maxpool,
  ]
  for i in range(args.model_stage):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(cnn, name))
  layers.append(getattr(cnn, 'avgpool'))
  model = torch.nn.Sequential(*layers)
  model.cuda()
  model.eval()
  return model


def run_batch(cur_batch, model):
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

  image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
  image_batch = (image_batch / 255.0 - mean) / std
  image_batch = torch.FloatTensor(image_batch).cuda()
  image_batch = torch.autograd.Variable(image_batch, volatile=True)
  feats = model(image_batch)
  feats = feats.data.cpu().clone().numpy()

  return feats


def main(args):
  img_path = '/data/vision/billf/object-properties/dataset/torralba-3/COCO'
  coco_path = '../data/coco'
  data_type = args.split + '2014'
  if args.split != 'test':
    ins_file = '{}/annotations/instances_{}.json'.format(coco_path,data_type)
  else:
    ins_file = '%s/annotations/image_info_test2014.json' % coco_path
  coco = COCO(ins_file)

  model = build_model()

  img_size = (args.image_height, args.image_width)
  with h5py.File(args.output_h5_file, 'w') as f:
    feat_dset = None
    i0 = 0
    cur_batch = []
    image_idxs = []

    for img_id in coco.imgs:
      image_idxs.append(int(img_id))
      img_info = coco.imgs[img_id]
      img = imread(img_path+'/'+data_type+'/'+img_info['file_name'], mode='RGB')
      img = imresize(img, img_size, interp='bicubic')
      img = img.transpose(2, 0, 1)[None]
      cur_batch.append(img)
      if len(cur_batch) == args.batch_size:
        feats = run_batch(cur_batch, model)
        if feat_dset is None:
          N = len(coco.imgs)
          _, C, H, W = feats.shape
          feat_dset = f.create_dataset('features', (N, C, H, W),
                                       dtype=np.float32)
        i1 = i0 + len(cur_batch)
        feat_dset[i0:i1] = feats
        i0 = i1
        print('Processed %d / %d images' % (i1, len(coco.imgs)))
        cur_batch = []
      if args.max_images != None and len(image_idxs) >= args.max_images:
        break

    if len(cur_batch) > 0:
      feats = run_batch(cur_batch, model)
      i1 = i0 + len(cur_batch)
      feat_dset[i0:i1] = feats
      print('Processed %d / %d images' % (i1, len(coco.imgs)))
    f.create_dataset('image_idxs', data=np.asarray(image_idxs))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
