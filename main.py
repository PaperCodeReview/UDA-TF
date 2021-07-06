import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from common import set_seed
from common import set_default
from common import get_logger
from common import get_session
from common import search_same
from common import create_stamp
from dataloader import set_dataset

import tensorflow as tf

def main(args=None):
    set_seed()
    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
        args.stamp = create_stamp()

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    ##########################
    # Strategy
    ##########################
    strategy = tf.distribute.MirroredStrategy()
    num_workers = strategy.num_replicas_in_sync
    assert args.batch_size % strategy.num_replicas_in_sync == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, strategy.num_replicas_in_sync))
    logger.info("GLOBAL BATCH SIZE : {}".format(args.batch_size))


    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args.data_path, args.dataset)
    if args.steps is not None:
        steps_per_epoch = args.steps
    elif args.dataset == 'cifar10':
        steps_per_epoch = 50000 // args.batch_size
        validation_steps = 10000 // args.batch_size
    elif args.dataset == 'svhn':
        steps_per_epoch = 73257 // args.batch_size
        validation_steps = 26032 // args.batch_size
    elif args.dataset == 'imagenet':
        steps_per_epoch = len(trainset) // args.batch_size
        validation_steps = len(valset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))


    ##########################
    # Model & Metric & Generator
    ##########################
    metrics = {
        'acc'       :   tf.keras.metrics.CategoricalAccuracy('acc', dtype=tf.float32),
        'val_acc'   :   tf.keras.metrics.CategoricalAccuracy('val_acc', dtype=tf.float32),
        'loss'      :   tf.keras.metrics.Mean('loss', dtype=tf.float32),
        'val_loss'  :   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
        'total_loss':   tf.keras.metrics.Mean('total_loss', dtype=tf.float32),
        'unsup_loss':   tf.keras.metrics.Mean('unsup_loss', dtype=tf.float32)}
    
    with strategy.scope():
        model = 



if __name__ == "__main__":
    def check_arguments(args):
        assert args.src_path is not None, 'src_path must be entered.'
        assert args.data_path is not None, 'data_path must be entered.'
        assert args.result_path is not None, 'result_path must be entered.'
        return args

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",       type=str,       default='resnet50')
    parser.add_argument("--batch_size",     type=int,       default=256)
    parser.add_argument("--dataset",        type=str,       default='cifar10')
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=100)

    parser.add_argument("--optimizer",      type=str,       default='sgd')
    parser.add_argument("--lr",             type=float,     default=.03)
    parser.add_argument("--loss",           type=str,       default='crossentropy', choices=['crossentropy'])
    parser.add_argument("--temperature",    type=float,     default=0.07)

    parser.add_argument("--brightness",     type=float,     default=0.,             help='0.4')
    parser.add_argument("--contrast",       type=float,     default=0.,             help='0.4')
    parser.add_argument("--saturation",     type=float,     default=0.,             help='0.4')
    parser.add_argument("--hue",            type=float,     default=0.,             help='0.4')

    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--tb_interval",    type=int,       default=0)
    parser.add_argument("--lr_mode",        type=str,       default='constant',     choices=['constant', 'exponential', 'cosine'])
    parser.add_argument("--lr_value",       type=float,     default=.1)
    parser.add_argument("--lr_interval",    type=str,       default='20,50,80')
    parser.add_argument("--lr_warmup",      type=int,       default=0)

    parser.add_argument('--src_path',       type=str,       default='.')
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default='-1')
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--ignore-search",  type=str,       default='')

    args = set_default(check_arguments(parser.parse_args()))
    main(args)