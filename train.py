import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# new imports
from yacs.config import CfgNode as CN
import copy

# datasets
import datasets.ssdg_pacs
import datasets.ssdg_officehome
import datasets.ssdg_digits_dg
import datasets.ssdg_vlcs

# trainers
import trainers.FBCSA
import trainers.MI


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.alpha:
        cfg.TRAINER.FBASA.ALPHA = args.alpha
    else:
        cfg.TRAINER.FBASA.ALPHA = 1

    if args.lamb:
        cfg.TRAINER.FBASA.LAMBDA = args.lamb
    else:
        cfg.TRAINER.FBASA.LAMBDA = 0

    if args.one_source_l:
        cfg.DATASET.ONE_SOURCE_L = args.one_source_l
    else:
        cfg.DATASET.ONE_SOURCE_L = None

    if args.batch_size:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size*len(cfg.DATASET.SOURCE_DOMAINS)
        cfg.DATALOADER.TRAIN_U.BATCH_SIZE = args.batch_size*len(cfg.DATASET.SOURCE_DOMAINS)
        print(f"Batch size: {cfg.DATALOADER.TRAIN_X.BATCH_SIZE}")


def extend_cfg(cfg, args):
    cfg.TRAINER.FBASA = CN()
    cfg.TRAINER.FBASA.CONF_THRE = 0.95  # confidence threshold
    cfg.TRAINER.FBASA.STRONG_TRANSFORMS = ()  # strong augmentations
    cfg.TRAINER.FBASA.C_OPTIM = copy.deepcopy(cfg.OPTIM)  # classifier's optim setting
    cfg.TRAINER.FBASA.CLASSIFIER = "normal"  # stochastic or normal
    cfg.TRAINER.FBASA.IMBALANCE = args.imbalance  # class imbalance type
    cfg.TRAINER.FBASA.GAMMA = args.gamma  # class imbalance ratio



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--imbalance",
        type=str,
        default="uniform",
        help="randomize the number of labels per class"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="class imbalance ratio"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--alpha",
        default=None,
        type=float,
        help="alpha value for alpha-TIM"
    )
    parser.add_argument(
        "--lamb",
        default=None,
        type=float,
        help="lambda value for lambda-TIM"
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="number of samples per domain in a batch (total batch size = batch_size * num_domains)"
    )
    parser.add_argument(
        "--one-source-l",
        default=None,
        type=str,
        help="name of the labelled source domain if only one source domain is used"
    )
    args = parser.parse_args()
    main(args)
