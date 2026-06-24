import os, sys
import time
import yaml
import platform
import torch
from argparse import ArgumentParser
import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.utils_training import CustomProgressBar, SaveBestStateDictCallback
###############################################################################
from utils_model.CVAE import CVAE
###############################################################################
from utils_data.DataModule import DataModule
from utils.constants import ROOT_PATH


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--train_name', default='test', type=str)
    parser.add_argument('--robot_name', default='shadowhand', type=str, help='Name of the robot')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training from last checkpoint')
    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    # torch.autograd.set_detect_anomaly(True)

    args = get_parser()

    if args.resume:
        train_name = args.robot_name + '_' + args.train_name
        log_dir = os.path.join('logs', train_name)
        chkpt_dir = os.path.join(log_dir, 'ckpts_dir')

        # Load config
        config_path = os.path.join(log_dir, 'config.yaml')
        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)
        
    else:
        # Load config
        config_path = os.path.join(ROOT_PATH, f'configs/train.yaml')
        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)

        # Prepare logs basedir
        cfg['robot_name'] = args.robot_name
        # cfg['fixed_rt'] = args.fixed_rt
        cfg['fixed_rt'] = True

        if args.robot_name == 'shadowhand':
            cfg['num_zones'] = 23
            cfg['num_types'] = 21
        elif args.robot_name == 'allegro_right':
            cfg['num_zones'] = 21
            cfg['num_types'] = 20

        train_name = cfg['robot_name'] + '_' + args.train_name
        
        log_dir = os.path.join('logs', train_name)
        chkpt_dir = os.path.join(log_dir, 'ckpts_dir')

        if os.path.exists(log_dir):
            if not 'node' in platform.node():
                print(f"Log directory {log_dir} already exists. Removing it ? (y/n)", end=' ')
                response = input().strip().lower()
                if response == 'y':
                    os.system(f'rm -rf "{log_dir}"')
                else:
                    sys.exit()
            else:
                os.system(f'rm -rf "{log_dir}"')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(chkpt_dir, exist_ok=True)

        with open(os.path.join(log_dir, 'config.yaml'), "w") as file:
            yaml.dump(cfg, file)

    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])

    # Tensorboard logger
    tb_logger = TensorBoardLogger(save_dir=log_dir, name='tb_dir', log_graph=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=chkpt_dir,
        filename=f'{train_name}' + '-{epoch:02d}',
        every_n_epochs=10,
        save_top_k=-1,
        save_last=True
    )

    # DataModule
    data_module = DataModule(cfg, shuffle=True)

    # Init Model
    model = CVAE(cfg)
    model.train()

    # Train
    trainer = L.Trainer(
        accelerator='gpu',
        devices='auto',
        strategy='ddp',
        max_epochs=cfg['n_epochs'],
        logger=tb_logger,
        callbacks=[CustomProgressBar(), checkpoint_callback, SaveBestStateDictCallback(chkpt_dir, train_name=train_name, save_best_epoch=False)],
        default_root_dir=log_dir,
        log_every_n_steps=1,
        precision='bf16-mixed',
        accumulate_grad_batches=cfg['batch_accumulation'],
        sync_batchnorm=True,
        # gradient_clip_val=1.0,            # Compatible with DDP, not FSDP
        # gradient_clip_algorithm='norm',   # Compatible with DDP, not FSDP
    )

    if os.environ.get("GLOBAL_RANK", "0") == "0":
        print(f'Training {train_name.upper()} on: {platform.node()}')

    start_time = time.time()

    if args.resume:
        # Load the last checkpoint
        last_checkpoint = os.path.join(chkpt_dir, 'last.ckpt')
        trainer.fit(model, datamodule=data_module, ckpt_path=last_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(os.path.join(chkpt_dir, f'{train_name}.ckpt'))

    print(f"Done with training {train_name} on {cfg['n_epochs']} epochs with a batchsize of {cfg['batch_size']}.")
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")