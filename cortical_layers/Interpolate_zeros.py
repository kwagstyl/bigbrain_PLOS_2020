import numpy as np
def chunks(l,n):
  """Yield n-sized chunks from l"""
  for i in xrange(0, len(l), n):
    yield l[i:i+n]


def get_neighbours(surfname):
  """Get neighbours from obj file"""
  Polys=[]
  k=0
  with open(surfname,'r') as fp:
    for i, line in enumerate(fp):
      if i==0:
    #Number of vertices
        n_vert=int(line.split()[6])
      elif i==2*n_vert+3:
        n_poly=int(line)
      elif i>2*n_vert+5:
        if not line.strip():
          k=1
        elif k==1:
          Polys.extend(line.split())
  Polys=map(int, Polys)
  tris=list(chunks(Polys,3))
  neighbours=[[] for i in range(n_vert)]
  for tri in tris:
      neighbours[tri[0]].extend([tri[1],tri[2]])
      neighbours[tri[2]].extend([tri[1],tri[0]])
      neighbours[tri[1]].extend([tri[0],tri[2]])
#Get unique neighbours
  for k in range(len(neighbours)):
      neighbours[k]=(list(set(neighbours[k])))
  return neighbours;


hemis=['left','right']
for hemi in hemis:
    Indices=np.loadtxt('/data1/users/kwagstyl/bigbrain/cortical_layers/'+hemi+'_indices.txt',dtype=int)
    neighbours=get_neighbours('/data1/users/kwagstyl/bigbrain/TestData/white_'+hemi+'_up.obj')
    ZeroRows=np.where(~Indices.any(axis=1))[0]
    NewIndices=np.zeros((len(ZeroRows),7))
    k=-1
    for z in ZeroRows:
        k+=1
        N1=neighbours[z]
        N2=[]
        for N in N1:
            N2.extend(neighbours[N])
        N2=np.unique(N2)
        N3=[]
        N4=[]
        for N in N2:
            N3.extend(neighbours[N])
        N3=np.unique(N3)
        for Na in N3:
            N4.extend(neighbours[Na])
        N2=np.unique(N4)
        Nonzeros=N2[np.where(Indices[N2].any(axis=1))[0]]
        if Nonzeros.any():
            I=np.round(np.mean(Indices[Nonzeros],axis=0))
            NewIndices[k]=I
    Indices[ZeroRows]=NewIndices
    np.savetxt('/data1/users/kwagstyl/bigbrain/TestData/'+hemi+'_indices_nonzeros.txt',Indices,fmt='%i')



