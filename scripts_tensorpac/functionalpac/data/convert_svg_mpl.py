"""
Converts svgs to matplotlib 

https://github.com/nvictus/svgpath2mpl
"""
import sys
from svgpath2mpl import parse_path
import pickle
from xml.dom import minidom
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np


def save_patch(obj, name):
    with open(name + '.patch', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_patch(name):
    with open(name + '.patch', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    svg_file = sys.argv[1]
    doc = minidom.parse(svg_file)  # parseString also exists
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    path_properties = [path.getAttribute('style') for path
                       in doc.getElementsByTagName('path')]
    doc.unlink()
    patches = []
    for i, path in enumerate(path_strings):
        path_obj = parse_path(path)
        style = dict((a.strip(), b.strip()) for a, b in (member.split(':')
                                                         for member in path_properties[i].split(';')))
        fill = style['fill']
        patch = PathPatch(path_obj, fc=fill, edgecolor=None, linewidth=None)
        patches.append(patch)

    pc = PatchCollection(patches, alpha=0.5, match_original=True)
    save_patch(pc, svg_file.split('.')[0])
