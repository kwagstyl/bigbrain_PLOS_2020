import numpy as np
import argparse
#Read in surfaces and extend
parser = argparse.ArgumentParser(description='Expand surface inwards and outwards by set distance along t-link')
parser.add_argument('graysurface',type=str, help='.obj graysurface to be expanded')
parser.add_argument('whitesurface',type=str, help='.obj white surface to be shrunk')
parser.add_argument('distance',type=float, help='distance in mm to be expanded in either direction')
parser.add_argument('--inward_distance', type=float, help='if specified inward distance is different from outward')
args=parser.parse_args()


Sg=args.graysurface
fp=open(Sg,'r')
n_vert=[]
p=0
for i, line in enumerate(fp):
  if i==0:
    #Number of vertices
    line1=line.split()
    n_vert=int(line.split()[6])
    coords_gray=np.zeros([n_vert,3])
    norms_gray=np.zeros([n_vert,3])
  elif 0< i < n_vert+1 :
#import coordinates
    coords_gray[i-1]=line.split()
  elif n_vert+1 < i < 2*n_vert+2:
#import normals (we just write them back out, not recomputing)
    norms_gray[p]=line.split()
    p+=1

#import white surface
Sw=args.whitesurface
fp=open(Sw,'r')
n_vert=[]
p=0
for i, line in enumerate(fp):
  if i==0:
    #Number of vertices
    line1=line.split()
    n_vert=int(line.split()[6])
    coords_white=np.zeros([n_vert,3])
    norms_white=np.zeros([n_vert,3])
  elif 0< i < n_vert+1 :
#import coordinates
    coords_white[i-1]=line.split()
  elif n_vert+1 < i < 2*n_vert+2:
#import normals (we just write them back out, not recomputing)
    norms_white[p]=line.split()
    p+=1



#if inner and outer distances are specified
if args.inward_distance:
  InDist=np.abs(args.inward_distance)
else:
  InDist=args.distance

OutDist=args.distance

Go='outer_'+Sg
Gi='inner_'+Sw
with open(Go,'w') as go, open(Gi,'w') as gi, open(Sw,'r') as s:
  line=s.readline()
  n_vert=int(line.split()[6])
  go.write(line)
  gi.write(line)
  k=-1
  for sline in s:
    k+=1
    if k< n_vert:
    #read in vertex coordinates
 # calculate vector Dist mm long
      D=(coords_gray[k]-coords_white[k])/np.linalg.norm(coords_gray[k]-coords_white[k])
      OutM=D*OutDist
      InM=D*InDist
 # calculate expanded surface point
      goline=np.add(coords_gray[k],OutM).tolist()
      giline=np.subtract(coords_white[k],InM).tolist()
 # write new surface
      go.write('%s\n' % (' ' +' '.join(str(v) for v in goline)))
      gi.write('%s\n' % (' ' +' '.join(str(v) for v in giline)))
    else:
      go.write(sline)
      gi.write(sline)
