import argparse

workflow = ['download_map','generate_demand','cost_precompute','solve','post_process']


# ============= parsing input ========================
parser = argparse.ArgumentParser()


parser.add_argument('-c','--casename',
                    dest = 'casename',
                    action = 'store',
                    required = True,
                    help = 'The casename to run. Make sure a folder under ./cases is already created and a config.py file is created'
                    )

parser.add_argument('-t','--task',
                    dest = 'task',
                    action = 'store',
                    required = True,
                    choices=workflow + ['full'],
                    help = 'specify the task'
                    )

parser.add_argument('-n','--new',
                    dest = 'new',
                    action = 'store',
                    default = False,
                    help = 'whether to run the task as new. If false, the task will skip if any resultant savefile exists. Otherwise, rerun and overwrite'
                    )

parser.add_argument('-g','--grid',
                    dest = 'grid',
                    action = 'store',
                    default = False,
                    help = 'whether to use a grid network. Make sure compatible config file is provided'
                    )

parser.add_argument('--heuristics',
                    dest = 'heuristics',
                    action = 'store',
                    default = True,
                    help = 'whether to use a hueristics in problem solvings'
                    )

args = parser.parse_args()
casename = args.casename
task = args.task
new = True if args.new == 'True' else False
grid = True if args.grid == 'True' else False
heuristics = True if args.heuristics == 'True' else False
# ======================== dynamic task run =============================

import sys
import os
casepath = './cases/' + casename
assert os.path.exists(os.path.join(casepath,'config.py')), 'no config file exists in the directiory. create one first'
# for intermediate data save
if not os.path.exists(os.path.join(casepath,'data')): os.makedirs(os.path.join(casepath,'data'))
# for result outputs, images, etc
if not os.path.exists(os.path.join(casepath,'output')): os.makedirs(os.path.join(casepath,'output'))

from importlib import import_module

def run_task(casename,task,**kargs):

    task_mod = import_module('.' + task, 'scripts')
    task_mod.run(casename,**kargs)

    print(task, ' completed for case ', casename)

if task == 'full':
    for task in workflow:
        run_task(casename,task,
        new = new,
        grid = grid,
        heuristics = heuristics)
else:
    run_task(casename,task,
        new = new,
        grid = grid,
        heuristics = heuristics)

