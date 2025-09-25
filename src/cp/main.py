import lightning.pytorch as pl
import torch

from src.cp.dataset.data_module import TrainValDataModule
from src.utils.main_utils import (
    arg_parser,
    make_checkpoint_callback,
    make_config,
    make_early_stopping_callback,
    make_logger,
    make_output_dir,
    make_tensorboard_logger,
)
from src.utils.model_utils import load_model

torch.set_float32_matmul_precision("medium")

# Enable anomaly detection for debugging gradient computation issues
torch.autograd.set_detect_anomaly(True)

parser = arg_parser()
args = parser.parse_args()

# load config
config = make_config(args.hparams_csi_pred)

# set random seed
pl.seed_everything(config.seed, workers=True)

# make output directory
output_dir = make_output_dir(config)

# make logger
logger = make_logger(output_dir)
logger.info("load config from - {}".format(args.hparams_csi_pred))
logger.info("output directory - {}".format(output_dir))
logger.info("seed - {}".format(config.seed))


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
logger.info("device - {} | {}".format(device, device_name if torch.cuda.is_available() else "CPU"))

# model
model = load_model(config, device=device)
logger.info("model - {}".format(str(model)))

# data module
datamodule = TrainValDataModule(config.data)
logger.info("setup data module")

# callbacks
ckpt_cb = make_checkpoint_callback(config=config, output_dir=output_dir)
earlystop_cb = make_early_stopping_callback(config=config)
tb_logger = make_tensorboard_logger(config=config, output_dir=output_dir)

# trainer
training_cfg = config.training
trainer = pl.Trainer(  # strategy TODO: add strategy
    max_epochs=training_cfg.num_epochs,
    accelerator=config.accelerator,
    devices=config.devices,
    precision=config.precision,
    callbacks=[ckpt_cb, earlystop_cb],
    logger=tb_logger,
    log_every_n_steps=training_cfg.log_every_n_steps,
    gradient_clip_val=training_cfg.gradient_clip_val,
    accumulate_grad_batches=training_cfg.accumulate_grad_batches,
    check_val_every_n_epoch=training_cfg.check_val_every_n_epoch,
    enable_progress_bar=training_cfg.enable_progress_bar,
    enable_model_summary=training_cfg.enable_model_summary,
)

# start training
if config.model.checkpoint_path is not None:
    logger.info("Resuming training from checkpoint: {}".format(config.model.checkpoint_path))
    trainer.fit(model, datamodule, ckpt_path=config.model.checkpoint_path)
else:
    logger.info("Starting training from scratch...")
    trainer.fit(model, datamodule)

# log complete
logger.info("Training complete. Checkpoint saved at: {}".format(ckpt_cb.best_model_path))
logger.info("TensorBoard logs saved at: {}".format(tb_logger.log_dir))
logger.info("Output directory: {}".format(output_dir))
logger.info("Run 'tensorboard --logdir {}' to view TensorBoard logs.".format(tb_logger.log_dir))
