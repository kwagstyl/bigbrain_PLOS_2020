import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert profiles to just intensity matrices')
parser.add_argument('profiles',type=str, help='.txt intensity profiles')
parser.add_argument('output',type=str, help='output profiles')
args=parser.parse_args()
profiles=args.profiles

num_lines=sum(1 for line in open(profiles))
num_profiles=num_lines//201
with open(profiles,'r') as P, open (args.output, 'w') as W:
  for r in range(num_profiles):
    IntensitySingle=[]
    for l in range(200):
        Line=P.readline().rstrip()
        LineSplit=Line.split(' ')
        IntensitySingle.append(float(LineSplit[4]))
    empty=P.readline()
    IntensityLine=' '.join(map(str,IntensitySingle))+'\n'
    W.write(IntensityLine)
