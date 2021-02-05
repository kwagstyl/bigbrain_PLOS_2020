import numpy as np
import os
import io_mesh
from scipy import spatial
from multiprocessing import Pool
from functools import partial
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


def find_nearest_point_on_triangle(p,tri):
    #solve claude's equation to find point on plane. eta and chi tell you whether it is in triangle 0<e<1
    #if not on triangle, check edges. if not on edge, take vertex
    #triangles need to go across
    a = (np.square(tri[0,0]-tri[2,0]) + np.square(tri[0,1]-tri[2,1]) + np.square(tri[0,2]-tri[2,2]))
    b = (tri[1,0]-tri[2,0])*(tri[0,0]-tri[2,0]) + (tri[1,1]-tri[2,1])*(tri[0,1]-tri[2,1]) + (tri[1,2]-tri[2,2])*(tri[0,2]-tri[2,2])
    c = b
    d = (np.square(tri[1,0]-tri[2,0]) + np.square(tri[1,1]-tri[2,1]) + np.square(tri[1,2]-tri[2,2]))
    f = (p[0] - tri[2,0])*(tri[0,0]-tri[2,0]) + (p[1]-tri[2,1])*(tri[0,1]-tri[2,1]) + (p[2]-tri[2,2])*(tri[0,2]-tri[2,2])
    g = (p[0] - tri[2,0])*(tri[1,0]-tri[2,0]) + (p[1]-tri[2,1])*(tri[1,1]-tri[2,1]) + (p[2]-tri[2,2])*(tri[1,2]-tri[2,2])
    chi = (d*f - b*g)/(a*d - b*c)
    eta = (-c*f + a*g)/(a*d - b*c)
    N1 = chi
    N2 = eta
    N3 = 1 - chi - eta
    X = N1*tri[0,0] + N2*tri[1,0] + N3*tri[2,0]
    Y = N1*tri[0,1] + N2*tri[1,1] + N3*tri[2,1]
    Z = N1*tri[0,2] + N2*tri[1,2] + N3*tri[2,2]
    return eta, chi, X, Y, Z;


def find_nearest_point_on_line(p,l):
    #if chi not between 0 and 1, then not on line
    #lines here go across x1, y1, z1
    chi = -(p[0]*(l[0,0]-l[1,0]) + p[1]*(l[0,1]-l[1,1]) + p[2]*(l[0,2]-l[1,2]) + 
            l[0,0]*(l[1,0]-l[0,0] + l[0,1]*(l[1,1]-l[0,1]) + l[0,2]*(l[1,2]-l[0,2]))) /(
        np.square(l[1,0]-l[0,0]) + np.square(l[1,1]-l[1,0]))
    N1 = 1 - chi
    N2 = chi
    X = N1*l[0,0] + N2*l[1,0]
    Y = N1*l[0,1] + N2*l[1,1]
    Z = N1*l[0,2] + N2*l[1,2]
    return chi, X, Y, Z;

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
        neighbours[tri[2]].extend([tri[0],tri[1]])
        neighbours[tri[1]].extend([tri[2],tri[0]])
#Get unique neighbours
    for k in range(len(neighbours)):      
        neighbours[k]=f7(neighbours[k])
    return np.array(neighbours);

def f7(seq):
    #returns uniques but in order to retain neighbour triangle relationship
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))];


import concurrent.futures                
def get_nearest_coords_parallel(surf1, surf2,neighbours):
    """ find nearest vertices in parallel"""
    dist = np.zeros(len(surf1))
    nearest_coords = np.zeros((len(surf1),3))
    fixed_args=[surf1,surf2,neighbours,dist,nearest_coords]
    func = partial(get_nearest_single,surf1,surf2,neighbours,dist,nearest_coords)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pool.map(func, range(len(surf1)))
    return dist,nearest_coords

def get_nearest_single(surf1,surf2,neighbours,dist,nearest_coords,k):
    coord=surf1[k]
    ListIndices=np.arange(len(surf2))
    se=0.0
    indices=[]
    nearest_coord=[]
    while len(indices)<1:
            se+=1
            indices=ListIndices[np.logical_and(surf2[:,0]>coord[0]-se,
                       np.logical_and(surf2[:,0]<coord[0]+se,
                       np.logical_and(surf2[:,1]>coord[1]-se,
                       np.logical_and(surf2[:,1]<coord[1]+se,
                       np.logical_and(surf2[:,2]>coord[2]-se,
                                      surf2[:,2]<coord[2]+se)))))]

    tmpsurf2 = surf2[indices]
    Index, dist[k]=closest_node(coord,tmpsurf2)
    dist[k],Index=spatial.KDTree(tmpsurf2).query(coord)
    RealIndex=indices[Index]
    Hexagon = neighbours[RealIndex]
       #consider keeping in python
    for n in range(len(Hexagon)-1):
        triangle=np.vstack((surf2[RealIndex],surf2[Hexagon[n]],surf2[Hexagon[n+1]]))
        chi, eta,x,y,z=find_nearest_point_on_triangle(coord,triangle)
        if 0 <= chi <= 1 and 0 <= eta <= 1:
            nearest_coord=np.array([x,y,z])
              #  print k
            break
        #check if chi and eta are bad, 
    if (chi < 0 or chi >1) and (eta < 0 or eta > 1):
        for n in range(len(Hexagon)):
            line= np.vstack((surf2[RealIndex],surf2[Hexagon[n]]))
            chi, x, y, z = find_nearest_point_on_line(coord,line)
            if 0 <= chi <= 1:
                nearest_coord = np.array([x,y,z])
               #     print k
                break
    if type(nearest_coord) is not np.ndarray:
        nearest_coord=tmpsurf2[Index]
    nearest_coords[k]=nearest_coord
    if k % 50000 ==0:
            print(str(k / (len(surf1)/100)) + "% complete")
   # return dist,nearest_coords


def get_nearest_coords(surf1,surf2, neighbours):
    """find nearest vertices in surf2 for each in surf1"""
    dist = np.zeros(len(surf1))
    nearest_coords = np.zeros([len(surf1),3])
    ListIndices=np.arange(len(surf2))
    k=-1
    for coord in surf1:
        k+=1
        #reduce search to only nearby coordinates
        se=-0.7
        indices=[]
        nearest_coord=[]
        while len(indices)<1:
            se+=1
            indices=ListIndices[np.logical_and(surf2[:,0]>coord[0]-se,
                       np.logical_and(surf2[:,0]<coord[0]+se,
                       np.logical_and(surf2[:,1]>coord[1]-se,
                       np.logical_and(surf2[:,1]<coord[1]+se,
                       np.logical_and(surf2[:,2]>coord[2]-se,
                                      surf2[:,2]<coord[2]+se)))))]
        tmpsurf2 = surf2[indices]
        #run nearest triangle, nearest edge and nearest vertex checking whether eta and chi are 0<x<1.
#        dist[k],Index=spatial.KDTree(tmpsurf2).query(coord)
        Index, dist[k]=closest_node(coord,tmpsurf2)
        RealIndex=indices[Index]
        Hexagon = neighbours[RealIndex]
        #consider keeping in python
        for n in range(len(Hexagon)-1):
            triangle=np.vstack((surf2[RealIndex],surf2[Hexagon[n]],surf2[Hexagon[n+1]]))
            chi, eta,x,y,z=find_nearest_point_on_triangle(coord,triangle)
            if 0 <= chi <= 1 and 0 <= eta <= 1:
                nearest_coord=np.array([x,y,z])
              #  print k
                break
        #check if chi and eta are bad, 
        if (chi < 0 or chi >1) and (eta < 0 or eta > 1):
            for n in range(len(Hexagon)):
                line= np.vstack((surf2[RealIndex],surf2[Hexagon[n]]))
                chi, x, y, z = find_nearest_point_on_line(coord,line)
                if 0 <= chi <= 1:
                    nearest_coord = np.array([x,y,z])
               #     print k
                    break
        if type(nearest_coord) is not np.ndarray:
            nearest_coord=tmpsurf2[Index]
        nearest_coords[k]=nearest_coord
        if k % 50000 ==0:
            print(str(k / (len(surf1)/100)) + "% complete")
    return dist, nearest_coords;

def get_nearest_coords_closest(surf1,surf2, neighbours):
    """find nearest vertices in surf2 for each in surf1"""
    dist = np.zeros(len(surf1))
    nearest_coords = np.zeros([len(surf1),3])
    ListIndices=np.arange(len(surf2))
    k=-1
    nearest_coord=None
    for coord in surf1:
        k+=1
        #reduce search to only nearby coordinates
        se=-0.7
        indices=None
        while indices is None or len(indices)<2:
            se+=1
            
            indices=ListIndices[np.logical_and(surf2[:,0]>coord[0]-se,
                       np.logical_and(surf2[:,0]<coord[0]+se,
                       np.logical_and(surf2[:,1]>coord[1]-se,
                       np.logical_and(surf2[:,1]<coord[1]+se,
                       np.logical_and(surf2[:,2]>coord[2]-se,
                                      surf2[:,2]<coord[2]+se)))))]
        tmpsurf2 = surf2[indices]
        #run nearest triangle, nearest edge and nearest vertex checking whether eta and chi are 0<x<1.
        Index=closest_node(coord, tmpsurf2)
        RealIndex=indices[Index]
        Hexagon = neighbours[RealIndex]
        #consider keeping in python
        for n in range(len(Hexagon)-1):
            triangle=np.vstack((surf2[RealIndex],surf2[Hexagon[n]],surf2[Hexagon[n+1]]))
            chi, eta,x,y,z=find_nearest_point_on_triangle(coord,triangle)
            if 0 <= chi <= 1 and 0 <= eta <= 1:
                nearest_coord=np.array([x,y,z])
                break
        
        #check if chi and eta are bad, 
        if (chi < 0 or chi >1) and (eta < 0 or eta > 1):
            for n in range(len(Hexagon)):
                line= np.vstack((surf2[RealIndex],surf2[Hexagon[n]]))
                chi, x, y, z = find_nearest_point_on_line(coord,line)
                if 0 <= chi <= 1:
                    nearest_coord = np.array([x,y,z])
               #     print k
                    break
        if (chi < 0 or chi >1) and (eta < 0 or eta > 1):
            nearest_coord=tmpsurf2[Index]
        if k==0:
            print(surf2[RealIndex])
        nearest_coords[k]=nearest_coord
        if k % 50000 ==0:
            print(str(k / (len(surf1)/100)) + "% complete")
    return nearest_coords;




def vector_triangle_plane_intersect(point,vector,triangle):
    # Plane: ax + by + cz = d
    # line xyz = vt + p
    #solve for t
    v1 = triangle[2]-triangle[0]
    v2 = triangle[1]-triangle[0]
    a,b,c = np.cross(v1,v2)
    d = triangle[0,0]*a+triangle[0,1]*b+triangle[0,2]*c
    t = (d - a*point[0] - b*point[1] - c*point[2])/(a*vector[0]+b*vector[1]+c*vector[2])
    point_on_plane=point+t*vector
    if np.isnan(point_on_plane).any() or np.isinf(point_on_plane).any():
        return None
    else:
        return point_on_plane
    
def triangle_point_intersection(inter,t):
    A_tt = ( t[1,0] - t[0,0] ) * ( t[1,0] - t[0,0] ) + ( t[1,1] - t[0,1] ) * ( t[1,1] - t[0,1] ) + ( t[1,2] - t[0,2] ) * ( t[1,2] - t[0,2] )
    A_st = ( t[1,0] - t[0,0] ) * ( t[2,0] - t[0,0] ) + ( t[1,1] - t[0,1] ) * ( t[2,1] - t[0,1] ) + ( t[1,2] - t[0,2] ) * ( t[2,2] - t[0,2] )
    A_ss = ( t[2,0] - t[0,0] ) * ( t[2,0] - t[0,0] ) + ( t[2,1] - t[0,1] ) * ( t[2,1] - t[0,1] ) + ( t[2,2] - t[0,2] ) * ( t[2,2] - t[0,2] )
    rhs_t = ( t[1,0] - t[0,0] ) * ( inter[0] - t[0,0] ) + ( t[1,1] - t[0,1] ) * ( inter[1] - t[0,1] ) + ( t[1,2] - t[0,2] ) * ( inter[2] - t[0,2] )
    rhs_s = ( t[2,0] - t[0,0] ) * ( inter[0] - t[0,0] ) + ( t[2,1] - t[0,1] ) * ( inter[1] - t[0,1] ) + ( t[2,2] - t[0,2] ) * ( inter[2] - t[0,2] )
    det = A_tt * A_ss - A_st * A_st
    if det == 0.0:
        print("fatal error in intersection")
        return None
    else:
        xi = ( A_ss * rhs_t - A_st * rhs_s ) / det
        eta = ( A_tt * rhs_s - A_st * rhs_t ) / det
        if xi >= 0.0 and eta >= 0.0 and xi + eta <= 1.0:
            return inter
        else :
            return None
        

        
def check_hexagon_for_intersection(point,vector,hexagon_indices,surface, index):
    """check hexagon surrounding a point for intersection"""
    for n in range(len(hexagon_indices)-1):
        triangle=np.vstack((surface[index],surface[hexagon_indices[n]],surface[hexagon_indices[n+1]]))
        point_in_plane=vector_triangle_plane_intersect(point,vector,triangle)
        if point_in_plane is not None:
            intersecting_point=triangle_point_intersection(point_in_plane,triangle)
            if intersecting_point is not None:
                return intersecting_point



def get_nearest_index(coord,surf2):
    """find nearest indices in surf2 a given coordinate"""
    ListIndices=np.arange(len(surf2))
        #reduce search to only nearby coordinates
    checkdist=0.4
    indices = None
    while indices is None or len(indices) <2:
        indices=ListIndices[np.logical_and(surf2[:,0]>coord[0]-checkdist,
                       np.logical_and(surf2[:,0]<coord[0]+checkdist,
                       np.logical_and(surf2[:,1]>coord[1]-checkdist,
                       np.logical_and(surf2[:,1]<coord[1]+checkdist,
                       np.logical_and(surf2[:,2]>coord[2]-checkdist,
                                      surf2[:,2]<coord[2]+checkdist)))))]
        checkdist+=0.5
#if can't find a vertex, expand the search radius by 0.5
    tmpsurf2 = surf2[indices]
        #run nearest triangle, nearest edge and nearest vertex checking whether eta and chi are 0<x<1.
    Index = closest_node(coord, tmpsurf2)
    RealIndex=indices[Index]
    return RealIndex;

def intersect_surface(surf1, vectors,surf2, neighbours, medial_wall=None):
    """find intersection with triangles on the next surface"""
    dist = np.zeros(len(surf1))
    intersecting_coords = np.zeros([len(surf1),3])
    b=0.
    for k, coord in enumerate(surf1):
        if medial_wall is not None:
            if medial_wall[k]==1:
                intersecting_point = coord
            else:
                start_index = k
                searchlight=neighbours[start_index]
                intersecting_point=check_hexagon_for_intersection(coord, vectors[k], searchlight,surf2,start_index)
                if intersecting_point is None:
                    for neighbour in searchlight:
                        searchlight2=neighbours[neighbour]
                        intersecting_point = check_hexagon_for_intersection(coord, vectors[k], searchlight2,surf2,start_index)
                        if intersecting_point is not None:
                            break
                            print(time.time()-t3)
                searchlight.append(start_index)
                while intersecting_point is None:
                    ne=[]
                    for n in neighbours[searchlight]:
                        ne.extend(n)
                    wider_ring=np.setdiff1d(ne,searchlight)
                    for v in wider_ring:
                        searchlight3= neighbours[v]
                        intersecting_point = check_hexagon_for_intersection(coord, vectors[k], searchlight3,surf2,start_index)
                    searchlight.extend(wider_ring)
                    if len(searchlight) >100:
                        b+=1.
                        intersecting_point = coord
                        continue
        if k % 5000 ==0:
            print(str(k / (len(surf1)/100)) + "% complete")
        intersecting_coords[k] = intersecting_point
    print(str(100.*b/k)+"% of vertices not moved")
    return intersecting_coords;

def clean_up_nans(newsurf, oldsurf, neighbours):
    Nans=np.unique(np.where(np.isnan(newsurf))[0])
    print(len(Nans))
    for n in Nans:
        neighbournan=np.array(neighbours[n])
        if np.isnan(newsurf[neighbournan]).any():
            newsurf[n]=oldsurf[n]
        else :
            newsurf[n]=np.mean(newsurf[neighbournan],axis=0)
    return newsurf
