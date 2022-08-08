import torch
import torch.nn.functional as F
from unet import UNet
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import piq

class LitUNet(pl.LightningModule):
  def __init__(self, criterion, batch_size, logger, in_channels=1, out_channels=1, wf=6, depth=6, padding=True, up_mode='upconv', batch_norm=True, img_log_period=100):
    super().__init__()
    self.save_hyperparameters()
    self.criterion = criterion
    self.unet = UNet(in_channels=in_channels, n_classes=out_channels, wf=wf, depth=depth, padding=padding, up_mode=up_mode, batch_norm=batch_norm)
    self.img_log_period=img_log_period
    self.example_input_array = torch.randn(1, 1, 320, 320)
    self.logger_func = logger

  #def train_dataloader(self):
  #  return torch.utils.data.DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=2)

  #def val_dataloader(self):
  #  return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=2)
  
  #def test_dataloader(self):
  #  return torch.utils.data.DataLoader(self.test_set, batch_size=self.hparams.batch_size, num_workers=2)

  def forward(self, x):
    return torch.clamp(self.unet(x), min=0., max=1.)

  def training_step(self, batch, batch_idx):
    X, y = batch
    prediction = torch.clamp(self.unet(X), min=0., max=1.)
    loss = self.criterion(prediction, y)
    return {'loss': loss, 'X': X, 'target': y, 'preds': prediction, 'batch_idx': batch_idx}
  
  def training_step_end(self, outputs):
    loss = outputs['loss']
    X = outputs['X']
    y = outputs['target']
    prediction = outputs['preds']
    batch_idx = outputs['batch_idx']
    self.log('train/loss', loss, on_step=True, on_epoch=True)
    if (batch_idx % self.img_log_period == 0) and (X.size(0) >= 5):
      self.logger_func.log_image(key='train/X', images=[x_i for x_i in X[:5]])
      self.logger_func.log_image(key='train/y', images=[y_i for y_i in y[:5]])
      self.logger_func.log_image(key='train/pred', images=[p_i for p_i in prediction[:5]])
      #self.log('train/X', [x_i for x_i in X[:5]])
      #self.log('train/y', [y_i for y_i in y[:5]])
      #self.log('train/pred', [p_i for p_i in prediction[:5]])

  def validation_step(self, batch, batch_idx):
    X, y = batch
    prediction = torch.clamp(self.unet(X), min=0., max=1)
    val_loss = self.criterion(prediction, y)
    return {'loss': val_loss, 'X': X, 'preds': prediction, 'target': y, 'batch_idx': batch_idx}

  def validation_step_end(self, outputs):
    val_loss = outputs['loss']
    X = outputs['X']
    prediction = outputs['preds']
    y = outputs['target']
    batch_idx = outputs['batch_idx']
    self.log('val/loss', val_loss)
    #self.val_ssim(prediction, y)
    self.log('val/ssim', piq.ssim(X, y, data_range=1.).item())
    #self.val_psnr(prediction, y)
    self.log('val/psnr', piq.psnr(X, y, data_range=1.).item())
    if (batch_idx % self.img_log_period == 0) and (X.size(0) >= 5):
      self.logger_func.log_image(key='val/X', images=[x_i for x_i in X[:5]])
      self.logger_func.log_image(key='val/y', images=[y_i for y_i in y[:5]])
      self.logger_func.log_image(key='val/pred', images=[p_i for p_i in prediction[:5]])
      #self.log('val/X', [x_i for x_i in X[:5]])
      #self.log('val/y', [y_i for y_i in y[:5]])
      #self.log('val/pred', [p_i for p_i in prediction[:5]])
  
  def test_step(self, batch, batch_idx):
    X, y = batch
    prediction = torch.clamp(self.unet(X), min=0., max=1)
    test_loss = self.criterion(prediction, y)
    return {'loss': test_loss, 'X': X, 'preds': prediction, 'target': y, 'batch_idx': batch_idx}
  
  def test_step_end(self, outputs):
    test_loss = outputs['loss']
    prediction = outputs['preds']
    y = outputs['target']
    X = outputs['X']
    batch_idx = outputs['batch_idx']
    self.log('test/loss', test_loss)
    #self.test_ssim(prediction, y)
    self.log('test/ssim', piq.ssim(X, y, data_range=1.).item())
    #self.test_psnr(prediction, y)
    self.log('test/psnr', piq.psnr(X, y, data_range=1.).item())
    if (batch_idx % self.img_log_period == 0) and (X.size(0) >= 5):
      self.logger_func.log_image(key='test/X', images=[x_i for x_i in X[:5]])
      self.logger_func.log_image(key='test/y', images=[y_i for y_i in y[:5]])
      self.logger_func.log_image(key='test/pred', images=[p_i for p_i in prediction[:5]])
      #self.log('test/X', [x_i for x_i in X[:5]])
      #self.log('test/y', [y_i for y_i in y[:5]])
      #self.log('test/pred', [p_i for p_i in prediction[:5]])

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    return optimizer