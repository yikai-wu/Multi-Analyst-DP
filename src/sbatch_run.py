import os
import sys
import argparse

# t-time c-cpu m-RAM g-gpu p-partition
pre_set_config = {
    "train-s":[
        "90:00:00",
        8,
        32,
        1,
        "compsci-gpu"
    ],
    "train-m":[
        "90:00:00",
        8,
        64,
        1,
        "compsci-gpu"
    ],
    "train-l":[
        "90:00:00",
        8,
        64,
        1,
        "compsci-gpu"
    ],
    "cpu-s":[
        "90:00:00",
        8,
        32,
        0,
        "compsci"
    ],
    "cpu-l":[
        "90:00:00",
        8,
        64,
        0,
        "compsci"
    ],
}

parser = argparse.ArgumentParser()

parser.add_argument('-c', type=int, help='number of cpu nodes', default=1)
parser.add_argument('-g', type=int, help='number of gpus', default=0)
parser.add_argument('-m', type=int, help='RAM requested', default=2)
parser.add_argument('-r', type=str, help='The command to run')
parser.add_argument('-t', type=str, help='time requested', default="2:00:00")
parser.add_argument('-p', type=str, help='computation partition', default="compsci-gpu")
parser.add_argument('-n', type=str, help='computation node')
parser.add_argument('-e', type=str, help='email notification')
parser.add_argument('-x', type=str, help='Pre-set configurations')
parser.add_argument('--nooutput', action='store_true', help='Output the command and settings')
parser.add_argument('--exclude', type=str, help='exclude some nodes')
parser.add_argument('--specialinfo', type=str, help='special info for the log name', default='slurm')


args = parser.parse_args()

assert args.r is not None, 'please specify the job you want to run after argument -r'

sh_tmp = "./.sbatchsrun.sh"
if os.path.isfile(sh_tmp):
    os.remove(sh_tmp)
writer = open(sh_tmp, 'w')

if args.x is not None:
    if args.x in pre_set_config:
        t, c, m, g, p = pre_set_config[args.x]
    else:
        print("unrecognizable pre-set config, select one among:")
        for k in pre_set_config:
            print(k)
        exit()
else:
    t, c, m, g, p = args.t, args.c, args.m, args.g, args.p

sh_base = ["#!/bin/bash\n",
           "#SBATCH -t {}\n".format(t),
           "#SBATCH -o ./sbatchlog/slurm-{}-%j.out\n".format(args.specialinfo),
           "#SBATCH -c {}\n".format(c),
           "#SBATCH --mem={}G\n".format(m),
           "#SBATCH --gres=gpu:{}\n".format(g),
           ]

if args.n is not None:
    sh_base.append("#SBATCH --nodelist={}\n".format(args.n))
else:
    sh_base.append("#SBATCH --partition={}\n".format(p))

if args.e is not None:
    sh_base.append("#SBATCH --mail-type=END,FAIL\n")
    sh_base.append("#SBATCH --mail-user={}\n".format(args.e))

for x in sh_base:
    if not args.nooutput:
        print(x[:-1])
    writer.write(x)

if not args.nooutput:
    print(args.r)
writer.write(args.r)
writer.close()

log_dir = 'sbatchlog'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

if args.exclude is None:
    os.system("sbatch .sbatchsrun.sh")
else:
    if not args.nooutput:
        print("excluding {}".format(args.exclude))
    os.system("sbatch --exclude={} .sbatchsrun.sh".format(args.exclude))
if not args.nooutput:
    print("\nTo check info, use sacct -j JobID\nTo cancel job, use scancel JobID\nThe log is stored in ./sbatchlog/slurm-JobID.out")
os.system("rm .sbatchsrun.sh")
