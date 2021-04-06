import os
import shutil
import re

root_path = "/home/marco/data/timeseries_entero/7"
for target in ['C', 'RD']:
    for hz in [2, 20]:
        path = os.path.join(root_path, target, str(hz))
        pattern = re.compile(r'{}b.dat$'.format(hz))
        substi = '{}b_moved.dat'.format(2 if hz == 20 else 20)
        for filename in sorted(os.listdir(path)):
            if pattern.search(filename):
                new_name = pattern.sub(substi, filename)
                src_path = os.path.join(path, filename)
                dest_path = os.path.join(root_path, target, '2' if hz == 20 else '20', new_name)
                shutil.move(src_path, dest_path)
                

