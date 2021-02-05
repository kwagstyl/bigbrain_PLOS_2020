def profile2indices(profile):
    if profile[0]==7 or len(set(profile))<4:
        #If nonsense profile, return all indices as zeros.
        Indices=[0,0,0,0,0]
    else:
        try :
        #Get index of first layer 1
            Indices=[profile.index(1)]
        except ValueError:
        #If no layer 1s, get first nonzero,
        #  sometimes layer 1 is ripped off but we still want locations of other layers
            Indices=[next((i for i, x in enumerate(profile) if x), None)]
        try :
            Indices.append(profile.index(2))
        except ValueError:
        #If no layer 2
            Indices.append(profile.index(3))
            return Indices;
        try :
            Indices.append(profile.index(3))
        except ValueError:
        #If no layer 3
            Indices.append(profile.index(4))
        try :
            Indices.append(profile.index(4))
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(4))
        #get last occurance of index 4.
        Indices.append(len(profile)-profile[::-1].index(4))
    return Indices;



def profile2indices_post_process(profile):
    if 8 in profile or len(set(profile))<5:
        #If nonsense profile, return all indices as zeros.
        Indices=[0,0,0,0,0,0,0]
        return Indices
    else:
        try :
            if profile.index(0) < 100:
                profile[0:profile.index(0)]=[0]*profile.index(0)
                if 8 in profile or len(set(profile))<5:
                    Indices=[0,0,0,0,0,0,0]
                    return Indices
        except ValueError:
            pass
        try :
        #Get index of first layer 1
            Indices=[profile.index(1)]
            #print profile.index(1)
        except ValueError:
        #If no layer 1s, get first nonzero,
        #  sometimes layer 1 is ripped off but we still want locations of other layers
            Indices=[next((i for i, x in enumerate(profile) if x), None)]
        #then set all before that to 1, to get rid of some nonsense
        profile[0:Indices[0]]=[1]*Indices[0]
        try :
            Indices.append(profile.index(2))
        except ValueError:
            try :
                Indices.append(len(profile)-profile[::-1].index(1)-1)
            except ValueError:
                return [0,0,0,0,0,0,0]
        #If no layer 2
        #then set all before that to 2,
        profile[0:Indices[1]]=[2]*Indices[1]
        try :
            Indices.append(profile.index(3))
        except ValueError:
        #If no layer 3
            try :
                Indices.append(profile.index(4))
            except ValueError:
                #if no layer 3 or 4, nonsense
                try :
                    Indices.append(profile.index(5))
                except ValueError:
#                    print("error b")
                    return [0,0,0,0,0,0,0]
        profile[0:Indices[2]]=[3]*Indices[2]
        try :
            Indices.append(profile.index(4))
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(3)-1)
        profile[0:Indices[3]]=[4]*Indices[3]
        try :
            Indices.append(profile.index(5))
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(4)-1)
        profile[0:Indices[4]]=[5]*Indices[4]
        try :
            Indices.append(profile.index(6))
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(5)-1)
        #get last occurance of index 4.
        try:
            Indices.append(len(profile)-profile[::-1].index(6)-1)
        except ValueError:
            Indices.append(len(profile)-profile[::-1].index(5)-1)
    return Indices;

def layer_error(prediction,labels):
    """percentage error depth per layer"""
    p_indices=profile2indices_post_process(list(prediction))
    l_indices=profile2indices_post_process(list(labels))
    diff=[a_i - b_i for a_i,b_i in zip(p_indices,l_indices)]
    return diff
    
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


def indices2surfaces(profile_file, indices, demo, rootname):
    """write indices to surfaces based on coordinates in txt file"""
    indices=indices.astype(int)
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



def confidence(predictions,indices):
    """ calculate network confidence per profile 
    sum over the profiles the difference between the highest and second highest prediction score
    return per class"""
    p_sorted=np.sort(predictions,axis=1)
    confidences=np.zeros(len(indices)+1)
    pi=0
    for k,i in enumerate(indices):
        confidences[k]=np.mean(p_sorted[pi:i,-1]-p_sorted[pi:i,-2],axis=0)
        pi=i
    confidences[-1]=np.mean(p_sorted[pi:200,-1]-p_sorted[pi:200,-2],axis=0)
    #confidences = np.sum(p_sorted[:,:,-1]-p_sorted[:,:,-2],axis=0)
    return confidences

def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2), np.min(dist_2)



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
                
                
                
def import_surface(surfname):
    """import surface to yield line1, coords, normals, triangles"""
    Polys=[]
    p=0
    k=0
    with open(surfname, 'r') as fp:
        for i, line in enumerate(fp):
            if i==0:
              #Number of vertices
                line1 = line.split()
                n_vert = int(line1[6])
                coords = np.zeros([ n_vert, 3])
                norms = np.zeros([ n_vert, 3])
            elif 0 < i < n_vert+1:
            #import coordinates
                coords[ i-1 ] = list(map( float, line.split() ))
            elif n_vert+1 < i < 2*n_vert + 2:
            #import normals (we just write them back out, not recomputing)
                norms[p]=line.split()
                p+=1
            elif i==2*n_vert+3:
            #Not needed any more but imports number of polygons. Is replace with smaller number
                n_poly=int(line)
            elif i==2*n_vert+4:
            #Import colour info
                Col=line
            elif i>2*n_vert+5 and k==0:
                if not line.strip():
                    k=1
            elif k==1:
            #import polygons
                Polys.extend(line.split())
    Polys = list(map(int,Polys))
    tris = np.array(list(chunks(Polys,3)))
    return n_vert, coords, norms, tris;

def region_to_y_coordinate(regions):
    Slices=np.array(['1066','2807', '3300', '3863', '4366', '4892', '5431', '1582','1600','4080','6316','3380','0301','1066','5431'])
    index=(regions-1)//6
    Slice=(Slices[index]).astype(int)
    y_coords=-70 + (Slice-1) * 0.02
    return y_coords
