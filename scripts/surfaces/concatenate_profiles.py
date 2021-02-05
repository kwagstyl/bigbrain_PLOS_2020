import glob, os, argparse

parser = argparse.ArgumentParser(description='concetenate intensity profiles')
parser.add_argument('profile_suffix',type=str, help='common suffix for profiles to be concetenated')
parser.add_argument('outputprofiles',type=str, help='.txt output profile file')
args=parser.parse_args()

dirname='*'+args.profile_suffix
dir= os.listdir('./')
dir=glob.glob(dirname)

num_lines=sum(1 for line in open(dir[1]))
num_profiles=num_lines/101
with open(dir[1]) as f:
  Complete=f.readlines()

for b in dir:
  with open(b,'r') as f:
    for R in range(num_lines):
      line=f.readline()      
      if Complete[R]=='0 0 0 0 0\n':
        Complete[R]=line

with open(args.outputprofiles, 'w') as f:
  for line in Complete:
    f.write(line)
