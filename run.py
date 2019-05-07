import os
import subprocess
import itertools
import time

# folder to save
base_path = 'results_FC_scales'

if not os.path.exists(base_path):
    os.makedirs(base_path)

# server setup
launcher = "srun --nodes=1 --gres=gpu:1 --time=40:00:00 --mem=60G" # THIS IS AN EXAMPLE!!!

# experimental setup
width = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
depth = [2, 3, 4, 5, 6, 7, 8, 9, 10]
seeds = list(range(3))
dataset = ['mnist', 'cifar10']
loss = ['NLL','linear_hinge']
model = ['fc']

grid = itertools.product(width, depth, seeds, dataset, loss, model)

processes = []
for w, dep, s, d, l, m in grid:

    save_dir = base_path + '/{}_{:04d}_{:02d}_{}_{}_{}'.format(dep, w, s, d, l, m)
    if os.path.exists(save_dir):
        # folder created only at the end when all is done!
        print('folder already exists, quitting')
        continue

    cmd = launcher + ' '
    cmd += 'python main.py '
    cmd += '--save_dir {} '.format(save_dir)
    cmd += '--width {} '.format(w)
    cmd += '--depth {} '.format(dep)
    cmd += '--seed {} '.format(s)
    cmd += '--dataset {} '.format(d)
    cmd += '--model {} '.format(m)
    cmd += '--lr {} '.format('0.1')
    cmd += '--lr_schedule '
    cmd += '--iterations {} '.format(65000)
    # cmd += '--print_freq {} '.format(1), # dbg
    # cmd += '--verbose '

    # print(cmd)

    f = open(save_dir + '.log', 'w')
    subprocess.Popen(cmd.split(), stdout=f, stderr=f)#.wait()
    
    