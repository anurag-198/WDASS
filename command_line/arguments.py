# Argument Parser
import argparse

parser = argparse.ArgumentParser(description='Semantic Segmentation')

parser.add_argument('--lr', type=float, default=0.0025)
parser.add_argument('--arch', type=str, default='deepv2.DeepV2R101',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')

parser.add_argument('--num_workers', type=int, default=4,
                    help='cpu worker threads per dataloader instance')
parser.add_argument('--do_flip', action='store_true', default=False)
parser.add_argument('--edgeLoss', action='store_true', default=True,
                    help='edge weights')
parser.add_argument('--cv', type=int, default=0,
                    help=('Cross-validation split id to use. Default # of splits set'
                          ' to 3 in config'))
parser.add_argument('--full_crop_training', action='store_true', default=False,
                    help='Full Crop Training')
parser.add_argument('--pre_size', type=int, default=None,
                    help=('resize long edge of images to this before'
                          ' augmentation'))
parser.add_argument('--log_msinf_to_tb', action='store_true', default=False,
                    help='Log multi-scale Inference to Tensorboard')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')

parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new lr ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--global_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--amsgrad', action='store_true', help='amsgrad for adam')

parser.add_argument('--test_mode', action='store_true', default=False,
                    help=('Minimum testing to verify nothing failed, '
                          'Runs code for 1 epoch of train and val'))

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=150,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--brt_aug', action='store_true', default=False,
                    help='Use brightness augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--poly_step', type=int, default=110,
                    help='polynomial epoch step')
parser.add_argument('--bs_trn', type=int, default=4,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_val', type=int, default=8,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=str, default='512,512',
                    help=('training crop size: either scalar or h,w'))
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--resume', type=str, default=None,
                    help=('continue training from a checkpoint. weights, '
                          'optimizer, schedule are restored'))
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--restore_net', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--result_dir', type=str, default='./logs',
                    help='where to write log output')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--width', type=int, default=2200,
                    help='same size for all datasets')

parser.add_argument('--multiprocessing_distributed', action='store_true', default=False)
parser.add_argument('--dist_url', type=str, default="tcp://127.0.0.1:6789")
parser.add_argument('--dist_backend', type=str, default="nccl")
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init_decoder', default=False, action='store_true',
                    help='initialize decoder with kaiming normal')

# Multi Scale Inference
parser.add_argument('--multi_scale_inference', action='store_true',
                    help='Run multi scale inference')

parser.add_argument('--default_scale', type=float, default=1.0,
                    help='default scale to run validation')


parser.add_argument('--eval', type=str, default=None,
                    help=('just run evaluation, can be set to val or trn or '
                          'folder'))
parser.add_argument('--eval_folder', type=str, default=None,
                    help='path to frames to evaluate')
parser.add_argument('--n_scales', type=str, default=None)

parser.add_argument('--sample_size', type=int, default=None,
                    help='sample size for fine-tuning')
parser.add_argument('--coarse_sample', type=int, default=2975,
                    help='sample size for coarse')
parser.add_argument('--fine_sample', type=int, default=25000,
                    help='sample size for fine')
parser.add_argument('--psl', type=int, default=None,
                    help='pseudo label iteration')
parser.add_argument('--backbone', type=str, default=None,
                    help='backbone')
parser.add_argument('--decoder', type=str, default=None,
                    help='decoder')
parser.add_argument('--max_iter', type=int, default=150000,
                    help='max_itern')
parser.add_argument('--alpha', type=float, default=0.999,
                    help='max_itern')
parser.add_argument('--noEdge', action='store_true', default=False,
                    help='no edge loss')
parser.add_argument('--edge_wt', type=float, default=10,
                    help='max_itern')
parser.add_argument('--contrast_wt', type=float, default=1.0,
                    help='max_itern')
parser.add_argument('--c_inter', type=float, default=0.5,
                    help='inter domain')
parser.add_argument('--c_real', type=float, default=0.25,
                    help='real')
parser.add_argument('--c_syn', type=float, default=0.25,
                    help='synthetic')
parser.add_argument('--use_wl', action='store_true', default=False,
                    help='max_itern')
parser.add_argument('--not_ema', action='store_true', default=False,
                    help='max_itern')
parser.add_argument('--test', action='store_true', default=False,
                    help='max_itern')
parser.add_argument('--bn_buffer', action='store_true', default=False,
                    help='update bn buffers')
parser.add_argument('--use_contrast', action='store_true', default=False,
                    help='update bn buffers')
parser.add_argument('--pretrain', action='store_true', default=False,
                    help='update bn buffers')
parser.add_argument('--weak_label', type=str, default='coarse',
                    help='decoder')
parser.add_argument('--imloss', action='store_true', default=False,
                    help='use image loss')
parser.add_argument('--improto', action='store_true', default=False,
                    help='use image loss')
parser.add_argument('--synthia', action='store_true', default=False,
                    help='use image loss')

parser.add_argument('--dump_assets', action='store_true',
                    help='Dump interesting assets')
parser.add_argument('--dump_all_images', action='store_true',
                    help='Dump all images, not just a subset')
parser.add_argument('--no_metrics', action='store_true', default=False,
                    help='prevent calculation of metrics')