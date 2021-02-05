import numpy as np

def indices2surfaces(profile_file, indices, demo, rootname):
    """write indices to surfaces based on coordinates in txt file"""
    indices=np.round(indices).astype(int)
    n_layers=np.shape(indices)[1]
    n_vert=len(indices)
    with open(profile_file,'r') as profiles:
        surfs={}
        for n in range(n_layers):
            surfs["corticalsurface{0}".format(n)]=[]
        for R in range(n_vert):
            xyz=[]
        #Read in each profile as 200 rows.
            for l in range(200):
                Line=profiles.readline().rstrip()
                LineSplit=Line.split(' ')
                xyz.append(LineSplit[0:3])
            empty=profiles.readline()
            for n in range(n_layers):
                surfs["corticalsurface{0}".format(n)].append(' ' + ' '.join(xyz[indices[R][n]]))
    for n in range(n_layers):
        SurfaceName=rootname+'_layer'+str(n)+'.obj'
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


def expand_to_fill(values, neighbours,mask=None):
    """expand overlay values by nearest neighbours
    including a mask/label will only fill to the label
    values can be label values, scalars or whole profiles to be filled outwards"""
    #minval= min(values)-0.01
    #overlay=np.full(len(neighbours),minval) 
    indices=np.where(values !=0)[0]
    overlay = values
   # not_filled = [n for n in range(len(neighbours)) if n not in indices]
    if mask is not None:
        #test if mask or label
        if np.max(mask)==1:
            label_vertices=np.where(mask==1)[0]
        else:
            label_vertices=mask
        not_filled = np.setdiff1d(label_vertices,indices)
    else:
        not_filled = np.setdiff1d( range(len(neighbours)),indices)
    outer_ring = indices.copy()
    not_filled_old=[]
    while len(not_filled)>1 :
        not_filled_old=not_filled
#        print(len(not_filled))
  #      print(len(outer_ring))
        new_outer_ring=[]
        for v in outer_ring:
            nbrs = neighbours[v]
#            t1=time.time()
            new_neighbours=[n for n in nbrs if n in not_filled]
#            t2=time.time()
            overlay[new_neighbours]=overlay[v]
            new_outer_ring.extend(new_neighbours)
#            t3=time.time()
#            print(t2-t1,t3-t2)
        new_outer_ring=np.unique(new_outer_ring)
        not_filled = np.setdiff1d(not_filled, new_outer_ring)
        if np.array_equal(not_filled, not_filled_old):
            break
    return overlay    

def chunks(l,n):
    """Yield n-sized chunks from l"""
    for i in range(0, len(l), n):
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
    Polys=list(map(int, Polys))
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

def surface_smoothing(values, surf_filename, fwhm=2):
    """smooth surface values using depth_potential. Will loop over multiple values if necessary
    smooths across surf_filename with fwhm set"""
    #check dimensions
    shrink=False
    flipped=False
    if np.ndim(values)==1:
        values=np.array([values])
        shrink=True
    elif values.shape[0]>values.shape[1]:
        values=values.T
        flipped=True
    new_values=np.zeros_like(values)
    for k,overlay in enumerate(values):
        np.savetxt('/tmp/tmp.txt', overlay, fmt='%i')
        print('smoothing surface '+str(k))
        subprocess.call('depth_potential -smooth '+ str(fwhm)+' /tmp/tmp.txt '+ surf_filename + ' /tmp/smtmp.txt',shell=True)
        new_overlay = np.round(np.loadtxt('/tmp/smtmp.txt')).astype(int)
        new_values[k] = new_overlay
    if shrink:
        return new_values[0]
    else:
        if flipped==True:
            return new_values.T
        return new_values
