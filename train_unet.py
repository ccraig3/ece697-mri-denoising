# by: Cameron Craig
from lightning_unet import LitUNet
from mri_sup_dataset import MriSupDataset

import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import pickle as pkl

MIN_NOISE = 0.
MAX_NOISE = 0.07

# Criterion
def my_criterion(prediction, y):
  return F.mse_loss(prediction, y) + F.l1_loss(prediction, y)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--imsize', type=int, required=False, default=320)
  parser.add_argument('--train_imgs', type=str, required=True)
  parser.add_argument('--val_imgs', type=str, required=True)
  parser.add_argument('--test_imgs', type=str, required=True)
  parser.add_argument('--train_bias', type=str, required=True)
  parser.add_argument('--val_bias', type=str, required=True)
  parser.add_argument('--test_bias', type=str, required=True)
  parser.add_argument('--ckpt_save_path', type=str, required=False, default='./model_ckpts/')
  parser.add_argument('--load_ckpt', type=str, required=False)
  parser.add_argument('--proj_name', type=str, required=True)
  parser.add_argument('--run_name', type=str, required=True)
  parser.add_argument('--precision', type=int, required=False, default=16)
  parser.add_argument('--num_gpus', type=int, required=False, default=1)
  parser.add_argument('--max_epochs', type=int, required=True)
  parser.add_argument('--batch_size', type=int, required=True)
  parser.add_argument('--run_test_afterwards', type=bool, required=False, default=False)
  parser.add_argument('--wf', type=int, required=False, default=5)
  parser.add_argument('--num_workers', type=int, required=False, default=8)

  args = parser.parse_args()

  '''
  SIZE = args.imsize
  TRAIN_IMGS = args.train
  VAL_IMGS = args.val
  if args.test:
    TEST_IMGS = args.test
  TRAIN_BIAS = args.train_bias
  VAL_BIAS = args.val_bias
  if args.test_bias:
    TEST_BIAS = args.test_bias
  '''

  # Open bias fields (lists of numpy arrays)
  print('Loading train bias fields...', end='', flush=True)
  with open(args.train_bias, 'rb') as bias_file:
    train_bias_fields = pkl.load(bias_file)
  print('Done')
  
  print('Loading val bias fields...', end='', flush=True)
  with open(args.val_bias, 'rb') as bias_file:
    val_bias_fields = pkl.load(bias_file)
  print('Done')
  
  if args.test_bias:
    print('Loading test bias fields...', end='', flush=True)
    with open(args.test_bias, 'rb') as bias_file:
      test_bias_fields = pkl.load(bias_file)
    print('Done')
  
  # initialize the datasets
  train_set = MriSupDataset(root_dir = args.train_imgs, bias_fields=train_bias_fields, noise_bounds=(MIN_NOISE, MAX_NOISE), size=args.imsize, preload = False)
  val_set = MriSupDataset(root_dir = args.val_imgs, bias_fields=val_bias_fields, noise_bounds=(MIN_NOISE, MAX_NOISE), size=args.imsize, preload = False)
  if args.test_imgs:
    test_set = MriSupDataset(root_dir = args.test_imgs, bias_fields=test_bias_fields, noise_bounds=(MIN_NOISE, MAX_NOISE), size=args.imsize, preload = False)
  
  # Create Dataloaders
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

  # Send logs to weights and biases
  wandb_logger = WandbLogger(project=args.proj_name, name=args.run_name)

  # Setup Pytorch Lightning trainer
  print('Setup trainer...', flush=True)
  trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator='gpu', devices=args.num_gpus, default_root_dir=args.ckpt_save_path, logger=wandb_logger, precision=args.precision)
  print('Done')

  # Build model
  unet_model = LitUNet(my_criterion, args.batch_size, wandb_logger, wf=args.wf)
  if args.load_ckpt:
    unet_model.load_from_checkpoint(args.load_ckpt)
  
  # Train model
  print('Fit trainer...', flush=True)
  trainer.fit(unet_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

  if args.run_test_afterwards:
    # Test Model
    trainer.test(unet_model, dataloaders=test_loader)
