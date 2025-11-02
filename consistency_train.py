# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import json
import os
import re
import warnings

import click
import torch

import dnnlib
from torch_utils import distributed as dist
from training import consistency_training_loop

warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--num-samples',      help='Samples in datasets', metavar='N',                           type=int, default=10000, show_default=True)
@click.option('--data',             help='Dataset path', metavar='STR',                                type=str, default=None, show_default=True)
@click.option('--dataset-name',     help='Dataset name', metavar="STR",                                type=click.Choice(['Board', 'Protein', 'RNA', 'Rotation', 'Cone', 'Fisher', 'Line', 'Peak', 'Volcano', 'Earthquake', 'Fire', 'Flood']), default='Board', show_default=True)
@click.option('--manifold',         help='Which manifold the data is on', metavar='STR',               type=click.Choice(['Euclidean', 'Torus', 'Sphere', 'SO3']), default='Euclidean', show_default=True)
@click.option('--outdir',           help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--precond',          help='Preconditioning & loss function', metavar='flow',            type=click.Choice(['flow']), default='flow', show_default=True)
@click.option('--loss-type',        help='Loss type', metavar='STR',                                   type=click.Choice(['Continuous', 'Discrete']), default='Continuous', show_default=True)
@click.option('--simplified-loss',  help='Whether to use simplified loss',                             is_flag=True)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',            help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--lr',               help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',              help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)

# Performance-related.
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--workers',          help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',             help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',         help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',             help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',             help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',             help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',             help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',         help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--distillation',     help='Consistency distillation', metavar='BOOL',                   type=bool, default=False, show_default=True)
@click.option('--teacher',          help='Teacher model from network pickle', metavar='PKL|URL',       type=str)
@click.option('--resume',           help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run',    help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    c = dnnlib.EasyDict()

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Initialize config dict.
    if opts.dataset_name == 'Board':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.torus_dataset.BoardTorusDataset', N=opts.num_samples, seed=c.seed)
    elif opts.dataset_name == 'Protein':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.torus_dataset.ProteinAngles', root=opts.data)
    elif opts.dataset_name == 'RNA':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.torus_dataset.RNAAngles', root=opts.data)
    elif opts.dataset_name == 'Rotation':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.so3_dataset.RotationDataset', N=opts.num_samples, seed=c.seed, noise=0.1, proj_y=10.)
    elif opts.dataset_name == 'Cone':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.so3_dataset.RawDataset', root=opts.data, category='cone')
    elif opts.dataset_name == 'Fisher':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.so3_dataset.RawDataset', root=opts.data, category='fisher24')
    elif opts.dataset_name == 'Line':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.so3_dataset.RawDataset', root=opts.data, category='line')
    elif opts.dataset_name == 'Peak':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.so3_dataset.RawDataset', root=opts.data, category='peak')
    elif opts.dataset_name == 'Volcano':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.sphere_dataset.Volcano', root=opts.data)
    elif opts.dataset_name == 'Earthquake':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.sphere_dataset.Earthquake', root=opts.data)
    elif opts.dataset_name == 'Fire':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.sphere_dataset.Fire', root=opts.data)
    elif opts.dataset_name == 'Flood':
        c.dataset_kwargs = dnnlib.EasyDict(class_name='datasets.sphere_dataset.Flood', root=opts.data)

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Validate dataset options.
    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
    c.dataset_kwargs.data_dimension = dataset_obj.dimension
    c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
    del dataset_obj # conserve memory

    # Network architecture.
    c.network_kwargs.update(in_channels=c.dataset_kwargs.data_dimension, base_channels=128, x_channel_mult=[2, 4, 4, 2], emb_channel_mult=2)

    # Preconditioning & loss function.
    if opts.precond == 'flow':
        c.network_kwargs.class_name = 'training.networks.FlowPrecond'
        if opts.loss_type == 'Continuous':
            c.loss_kwargs.class_name = 'training.loss.ConsistencyLoss'
            c.loss_kwargs.simplified = opts.simplified_loss
        else:
            c.loss_kwargs.class_name = 'training.loss.DiscreteConsistencyLoss'
        c.loss_kwargs.N = c.dataset_kwargs.data_dimension
        c.loss_kwargs.manifold = opts.manifold
    else:
        raise NotImplementedError()

    if opts.distillation and opts.teacher is None:
        raise click.ClickException('--distillation requires --teacher')
    if not opts.distillation:
        c.loss_kwargs.update(distillation=False, teacher_model=opts.teacher)
    else:
        c.loss_kwargs.update(distillation=True, teacher_model=opts.teacher)

    # Network options.
    c.network_kwargs.update(dropout=opts.dropout)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    if opts.loss_type == 'Continuous':
        c.loss_kwargs.update(
            tangent_warmup_steps=(c.total_kimg * 1000 // opts.batch),
            # tangent_warmup_steps=10000,
            simplified=opts.simplified_loss,
        )
    else:
        c.loss_kwargs.update(
            dt=0.01,
        )
    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    loss_name = opts.loss_type
    if opts.simplified_loss:
        loss_name += 'Simplified'
    desc = f'{opts.dataset_name}-{loss_name}-consistency-gpus{dist.get_world_size():d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Preconditioning & loss:  {"Consistency"}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    consistency_training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
