import os
import wandb
from pytorch_lightning.loggers import WandbLogger

def config_wandb_logger(args):
    os.environ["WANDB_START_METHOD"] = "fork"
    os.environ["WANDB_SERVICE_WAIT"] = "30000"
    os.environ["WANDB__SERVICE_WAIT"] = "30000"
    os.environ["WANDB_API_KEY"] = "ce5d4dbf4560979e87ff5a0a2d33bb48d9ff4c81"
    os.environ["WANDB_MODE"] = args.wandb_mode
    # os.environ["WANDB_MODE"] = "offline"
    # os.environ["WANDB_MODE"] = "disabled"
    log_root = args.wandb_dir
    wandb_logger = WandbLogger(
            project=args.wandb_proj,
            name=args.wandb_name,
            save_dir=args.wandb_dir,
            dir=args.wandb_dir,
            settings=wandb.Settings(
                code_dir=".", 
                _service_wait=3000,
                init_timeout=3000
            )
    )
    if args.wandb_mode != "disabled":
        run_name = wandb_logger.experiment.dir.split("/")[-2]
        project_name = wandb_logger.experiment.project
    else:
        run_name = "debug"
        project_name = "debug"
    return wandb_logger, log_root, project_name, run_name