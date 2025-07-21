#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import warnings

import pytorch_lightning as pl
import torch
from easydict import EasyDict
from loguru import logger
from prefigure.prefigure import push_wandb_config
from pytorch_lightning.callbacks import ModelCheckpoint

from data.fairplay_dataloader import CreateDataLoader
from options.train_options import TrainOptions
from src.training.ccstereo_trainer import (
    AutoencoderDemoCallback,
    AutoencoderTrainingWrapper,
)
from util.tensorboard import config_wandb_logger
from visual.plot_spec import SpectrogramPlotter

warnings.simplefilter(action='ignore', category=FutureWarning)

def create_training_wrapper(model_config, model, test_config, train_config):
    ema_copy = None
    training_config = model_config['training']
    use_diff_audio = model_config["use_diff_audio"]
    use_ema = training_config.get("use_ema", False)
    logger.info(f"Use diff audio: {use_diff_audio}")
    latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)
    teacher_model = training_config.get("teacher_model", None)
    return AutoencoderTrainingWrapper(
        model,
        lr=training_config["learning_rate"],
        warmup_steps=training_config.get("warmup_steps", 0),
        encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
        sample_rate=model_config["sample_rate"],
        loss_config=training_config["loss_configs"],
        optimizer_configs=training_config["optimizer_configs"],
        use_ema=use_ema,
        ema_copy=ema_copy if use_ema else None,
        force_input_mono=training_config.get("force_input_mono", False),
        latent_mask_ratio=latent_mask_ratio,
        teacher_model=teacher_model,
        logging_config=training_config.get('logging', {}),
        test_hop_size=test_config["test_hop_size"],
        train_config=train_config
    )

def load_training_wrapper(ckpt_path, model_config, model, test_config, train_config):
    ema_copy = None
    training_config = model_config['training']
    use_diff_audio = model_config["use_diff_audio"]
    logger.info(f"Use diff audio: {use_diff_audio}")
    use_ema = training_config.get("use_ema", False)

    latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)
    teacher_model = training_config.get("teacher_model", None)
    return AutoencoderTrainingWrapper.load_from_checkpoint(
        ckpt_path,
        autoencoder=model,
        strict=True,
        sample_rate=model_config["sample_rate"],
        loss_config=training_config["loss_configs"],
        optimizer_configs=training_config["optimizer_configs"],
        use_ema=use_ema,
        ema_copy=ema_copy if use_ema else None,
        force_input_mono=training_config.get("force_input_mono", False),
        latent_mask_ratio=latent_mask_ratio,
        teacher_model=teacher_model,
        logging_config=training_config.get('logging', {}),
        #
        test_hop_size=test_config["test_hop_size"],
        #
        train_config=train_config
    )

def create_demon_callback(model_config, **kwargs):  # noqa: F811
    training_config = model_config['training']
    demo_config = training_config.get("demo", {})
    return AutoencoderDemoCallback(
        demo_every=demo_config.get("demo_every", 2000),
        max_num_sample=demo_config.get("max_num_sample", 4),
        sample_size=model_config["sample_size"],
        sample_rate=model_config["sample_rate"],
        **kwargs
    )
    
class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def model_testing(
        ckpt_path,
        model=None, 
        epochs=200, 
        global_step=0,
        wandb_logger=None, 
        model_config=None, 
        train_config=None, 
        test_config=None, 
        test_loader=None,
        save_visual=True,
        save_root="./Test/spectrogram",
    ):
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    training_wrapper = create_training_wrapper(model_config, model, test_config, train_config)
    wandb_logger.watch(training_wrapper)
    
    if save_visual:
        fn_ = ckpt_path.split("/")[-3]
        save_visual_root = os.path.join(save_root, fn_)
        spec_save_path = os.path.join(save_visual_root, "spec")

        training_wrapper.save_visual_root = save_visual_root
        training_wrapper.spec_plotter = SpectrogramPlotter(spec_save_path)

        os.makedirs(save_visual_root, exist_ok=True)
        os.makedirs(os.path.join(save_visual_root, "gt_diff"), exist_ok=True)
        os.makedirs(os.path.join(save_visual_root, "pred_diff"), exist_ok=True)
        os.makedirs(os.path.join(save_visual_root, "gt_binaural"), exist_ok=True)
        os.makedirs(os.path.join(save_visual_root, "pred_binaural"), exist_ok=True)

        # Read eval_list.txt into a list
        eval_list_path = "./util/eval_list.txt"
        with open(eval_list_path, "r") as f:
            eval_list = [line.strip() for line in f]
        training_wrapper.eval_list = eval_list

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        num_nodes=1,
        # strategy=args.strategy,
        # precision=args.precision,
        # accumulate_grad_batches=args.accum_batches, 
        # callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=epochs,
        # gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        #
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
    )
    trainer.global_step = global_step
    trainer.test(training_wrapper, dataloaders=test_loader)
    return

def main():

    # args, model_config = load_stable_audio_args()
    args = TrainOptions().parse()
    # Get JSON config from args.model_config
    
    logger.critical(f"Loading model config from << {args.model_config} >>")
    args.model_config = os.path.join("./configs/fairplay/", args.model_config)
    with open(args.model_config) as f:
        model_config: dict = EasyDict(json.load(f))
    args.update(**vars(model_config))
    model_config.update({"use_visual": args.use_visual})
    
    from models.ccstereo_model.audioVisual_model import AudioVisualModel
    from models.ccstereo_model.models import ModelBuilder

    builder = ModelBuilder()
    
    net_visual = builder.build_visual(
        backbone=args.backbone,
        img_dim=128*args.dim_scale,
    )
    net_audio = builder.build_audio(
        ngf=16*args.dim_scale,
        input_nc=2,
        output_nc=2,
        img_dim=128*args.dim_scale,
    )
    nets = (net_visual, net_audio)

    # construct our audio-visual model
    model = AudioVisualModel(nets, args)

    model.sample_rate = model_config['sample_rate']
    args.device = torch.device("cuda")

    #construct data loader
    train_loader, val_loader, test_loader = CreateDataLoader(args)

    print('#training clips = %d' % len(train_loader.dataset))
    print('#validation clips = %d' % len(val_loader.dataset))
    print('#testing clips = %d' % len(test_loader.dataset))

    wandb_logger, log_root, project_name, run_name = config_wandb_logger(args)

    checkpoint_dir = os.path.join(log_root, project_name, run_name, "checkpoints") 
    train_log_dir = os.path.join(log_root, project_name, run_name, "train_logs") 
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(train_log_dir, exist_ok=True)
    model_config.update({"train_log_dir": train_log_dir})
    push_wandb_config(wandb_logger, args)

    ckpt_config = args.training.checkpoint
    logger.info(f"Checkpoint config: {ckpt_config}")
    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, 
        save_top_k=1, 
        monitor="test/stft_l2_dist_epoch",
        mode="min",
    )

    test_config = {
        "test_steps": args.test_steps,
        "test_cfg_scale": args.test_cfg_scale,
        "test_hop_size": args.test_hop_size
    }

    train_config = {
        "epochs": args.epochs,
        "epoch_inters": len(train_loader),
        "use_flow": args.use_flow,
        "multi_frames": args.multi_frames,
        "use_mask": args.use_mask,
        "lr_img": args.lr_img,
        "lr_aud": args.lr_aud,
        "weight_decay": args.weight_decay,
    }
    
    training_wrapper = create_training_wrapper(model_config, model, test_config, train_config)
    training_wrapper.checkpoint_dir = checkpoint_dir

    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    demo_callback = create_demon_callback(model_config, demo_dl=train_loader, train_config=train_config)
    exc_callback = ExceptionCallback()
    
    # from stable_audio_tools.training.mono2binaural_trainer import LPIPS_MD
    # lpips_md = LPIPS_MD(model_config).to(args.device)
    # training_wrapper.lpips_md = lpips_md
    
    wandb_logger.watch(training_wrapper)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        num_nodes=1,
        # strategy=args.strategy,
        # precision=args.precision,
        # accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=args.epochs,
        default_root_dir=train_log_dir,
        reload_dataloaders_every_n_epochs=0,
        #
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
        # deterministic=True,
    )
    logger.warning("------- Training Setting -------")
    logger.warning(f"Use Optical Flow: {args.use_flow}")
    logger.warning(f"Use Multiple Frames: {args.multi_frames}")
    trainer.fit(training_wrapper, train_loader, test_loader, ckpt_path=None)

    ckpt_callback.best_model_path = "./logs_ct/FAIRPLAY-5S/run-20241129_075245-fymu0ey3/best_model.pth"
    best_train_wrapper = load_training_wrapper(ckpt_callback.best_model_path, model_config, model, test_config, train_config)

    save_visual = False
    if save_visual:
        save_root="./Test/spectrogram"
        fn_ = checkpoint_dir.split("/")[-2]
        save_visual_root = os.path.join(save_root, fn_)
        spec_save_path = os.path.join(save_visual_root, "spec")

        best_train_wrapper.save_visual_root = save_visual_root
        best_train_wrapper.spec_plotter = SpectrogramPlotter(spec_save_path)

        os.makedirs(save_visual_root, exist_ok=True)
        os.makedirs(os.path.join(save_visual_root, "attention"), exist_ok=True)
        # os.makedirs(os.path.join(save_visual_root, "gt_diff"), exist_ok=True)
        # os.makedirs(os.path.join(save_visual_root, "pred_diff"), exist_ok=True)
        # os.makedirs(os.path.join(save_visual_root, "gt_binaural"), exist_ok=True)
        # os.makedirs(os.path.join(save_visual_root, "pred_binaural"), exist_ok=True)

        # Read eval_list.txt into a list
        eval_list_path = "./util/eval_list.txt"
        with open(eval_list_path, "r") as f:
            eval_list = [line.strip() for line in f]
        best_train_wrapper.eval_list = eval_list

    trainer.test(best_train_wrapper, dataloaders=test_loader)
    
if __name__ == '__main__':
    main()