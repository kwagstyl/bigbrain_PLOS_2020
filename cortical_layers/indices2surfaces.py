import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Create surfaces from indices identified by NN')
parser.add_argument('profile',type=str, help='.txt profile file for xyzs')
parser.add_argument('indices', type=str, help='.txt index file')
parser.add_argument('demo',type=str, help='.obj demo file')
parser.add_argument('Rootname',type=str, help='Prefix name for output files')
args=parser.parse_args()
indices=args.indices
Indices=np.loadtxt(indices,dtype=int)
n_layers=np.shape(Indices)[1]

n_vert=len(Indices)
profile=args.profile
Profiles=open(profile,'r')
surfs={}
for n in range(n_layers):
  surfs["corticalsurface{0}".format(n)]=[]


for R in range(n_vert):
  xyz=[]
#Read in each profile as 200 rows.
  for l in range(200):
    Line=Profiles.readline().rstrip()
    LineSplit=Line.split(' ')
    xyz.append(LineSplit[0:3])
  empty=Profiles.readline()
  for n in range(n_layers):
      surfs["corticalsurface{0}".format(n)].append(' ' + ' '.join(xyz[Indices[R][n]]))


Rootname=args.Rootname
for n in range(n_layers):
    SurfaceName=Rootname+'_layer'+str(n)+'.obj'
    demo=args.demo
    with open(demo,'r') as input, open(SurfaceName, 'w') as output:
        line=input.readline()
        n_vert=int(line.split()[6])
        output.write(line)
        k=-1
        for line in input:
            k+=1
            if k<n_vert and surfs["corticalsurface{0}".format(n)][k]!=" 0 0 0":
                output.write('%s\n' % surfs["corticalsurface{0}".format(n)][k])
            else:
                output.write(line)

