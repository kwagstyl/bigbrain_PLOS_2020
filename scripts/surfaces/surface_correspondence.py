
# coding: utf-8

# In[101]:


import numpy as np
import os
import io_mesh as io
from scipy import spatial
import surface_tools as st
import matplotlib.pyplot as plt
import subprocess
from scipy.stats import skew


# In[102]:




surfdir='/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/'
hemis=['right','left']


# In[150]:


def calculate_area(surfname,fwhm):
    """calculate surface area using minctools"""
    try:
        subprocess.call("depth_potential -area_voronoi " + surfname + " /tmp/tmp_area.txt",shell=True)
        subprocess.call("depth_potential -smooth " + str(fwhm) + " /tmp/tmp_area.txt " + surfname + " /tmp/sm_area.txt",shell=True)
        area=np.loadtxt("/tmp/sm_area.txt")
        subprocess.call("rm /tmp/sm_area.txt /tmp/tmp_area.txt",shell=True)
    except OSError:
        print("depth_potential not found, please install CIVET tools or replace with alternative area calculation/data smoothing")
        return 0;
    return area;


def beta(alpha, aw, ap):
    """Compute euclidean distance fraction, beta, that will yield the desired
    volume fraction, alpha, given vertex areas in the white matter surface, aw,
    and on the pial surface, ap.

    A surface with `alpha` fraction of the cortical volume below it and 
    `1 - alpha` fraction above it can then be constructed from pial, px, and 
    white matter, pw, surface coordinates as `beta * px + (1 - beta) * pw`.
    """
    if alpha == 0:
        return np.zeros_like(aw)
    elif alpha == 1:
        return np.ones_like(aw)
    else:
        return 1-(1 / (ap - aw) * (-aw + np.sqrt((1-alpha)*ap**2 + alpha*aw**2)))

def generate_equivolumetric_coords(gray_surf, white_surf, n_surfs,fwhm=2):
    """ generate equivolumetric surface coordinates for matched gray white surfaces
    if"""
    
    wm = io.load_mesh_geometry(white_surf)
    gm = io.load_mesh_geometry(gray_surf)
    
    wm_vertexareas = calculate_area(white_surf, fwhm)
    pia_vertexareas = calculate_area(gray_surf, fwhm)
    vectors= wm['coords'] - gm['coords']
    surfs_coords = np.zeros((n_surfs,len(vectors),3))
    for depth in range(n_surfs):
        betas = beta(float(depth)/(n_surfs-1), wm_vertexareas, pia_vertexareas)
        surfs_coords[depth,:,:] = gm['coords'] + vectors* np.array([betas]).T
    return surfs_coords


def angle_vecs(vec1,vec2):
    """calculate angle between two vectors"""
    angle=np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return angle

def norm_differences(norms1, norms2, medial_wall):
    """calculate differences between normals of two surfaces"""
    angles=[]
    indices=np.where(medial_wall==0)[0]
    for index in indices:
        angles.append(angle_vecs(norms1[index],norms2[index]))
    return np.array(angles)


fwhms=[0,0.5,1.0,1.5, 2.0,5.0,10.0]
fwhms = [2.0]
gray_intersect=np.zeros((2,len(fwhms)))
white_intersect=np.zeros((2,len(fwhms)))
anglesmean=np.zeros((2,len(fwhms)))
grayanglesmean=np.zeros((2,len(fwhms)))
whiteanglesmean=np.zeros((2,len(fwhms)))
whiteanglesstd=np.zeros((2,len(fwhms)))
grayanglesstd=np.zeros((2,len(fwhms)))
Force=False
g=[]
for h,hemi in enumerate(hemis):
    medial_wall=np.loadtxt('/data1/users/kwagstyl/profile_location/surfaces/medial_wall_'+hemi+'.txt').astype(int)
    _, mid_xyz, mid_norms, _ = st.import_surface(os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj'))

#    check if middle equi_iso exists, else generate it
    if Force or not os.path.isfile(os.path.join(surfdir, "Normalized_vector_" + hemi + "_x.txt")) or not os.path.isfile(os.path.join(surfdir, "white_dist" + hemi + ".txt")):
        neighbours = st.get_neighbours(surfdir+'gray_' + hemi + '_327680.obj')
        n, gray_xyz, nm, t = st.import_surface(surfdir+'gray_' + hemi + '_327680.obj')
        n, white_xyz, nm, t = st.import_surface(surfdir+'white_' + hemi + '_327680.obj') 
        if not os.path.isfile(os.path.join(surfdir, 'equi_iso_0.5_'+hemi+'.obj')):
            print('generating equivolumetric coords')
            depths = 3
            mid_ind = (depths-1)//2
            equis = generate_equivolumetric_coords(surfdir+'gray_' + hemi + '_327680.obj', surfdir+'white_' + hemi + '_327680.obj',depths)

            g=io.load_mesh_geometry(os.path.join(surfdir, 'gray_' + hemi + '_327680.obj'))
            g['coords']= equis[mid_ind]
            io.save_mesh_geometry(os.path.join(surfdir, 'equi_0.5_'+hemi+'.obj'),g)
            subprocess.call('sphere_surf_interpolate.pl '+os.path.join(surfdir, 'equi_0.5_'+hemi+'.obj ') 
                            + os.path.join(surfdir, 'equi_iso_0.5_'+hemi+'.obj'),shell=True)
        if not os.path.isfile(os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj')):
            if hemi =='right':
                subprocess.call('param2xfm -clobber -scales -1 1 1 flip.xfm',shell=True)
                subprocess.call('transform_objects '+os.path.join(surfdir, 'equi_iso_0.5_'+hemi+'.obj ')
                               + 'flip.xfm ' + os.path.join(surfdir, 'equi_iso_0.5_right_like_left.obj '),shell=True)
                subprocess.call('subdivide_polygons ' + os.path.join(surfdir, 'equi_iso_0.5_right_like_left.obj ')
                                 + os.path.join(surfdir, 'equi_iso_0.5_right_like_left_up.obj '),shell=True)
                subprocess.call('transform_objects ' + os.path.join(surfdir, 'equi_iso_0.5_right_like_left_up.obj ')
                                 + 'flip.xfm ' + os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj'),shell=True)
            else :
                subprocess.call('subdivide_polygons '+os.path.join(surfdir, 'equi_iso_0.5_'+hemi+'.obj ') 
                            + os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj'),shell=True)
        n, mid_xyz, nm, t = st.import_surface(os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj'))
        new_gray = mid_xyz
        print('finding nearest gray')
        gray_dist, near_gray = st.get_nearest_coords_parallel(mid_xyz,gray_xyz,neighbours)
        print('finding nearest white')
        white_dist, near_white = st.get_nearest_coords_parallel(mid_xyz,white_xyz,neighbours)
        Vector=near_gray-near_white
        NormVector=Vector/np.linalg.norm(Vector,axis=1).reshape((len(Vector),1))
        print("saving out files")
        np.savetxt("" + surfdir + "Normalized_vector_" + hemi + "_x.txt",NormVector[:,0],fmt='%.5f')
        np.savetxt("" + surfdir + "Normalized_vector_" + hemi + "_y.txt",NormVector[:,1],fmt='%.5f')
        np.savetxt("" + surfdir + "Normalized_vector_" + hemi + "_z.txt",NormVector[:,2],fmt='%.5f')
        
        np.savetxt("" + surfdir + "white_dist" + hemi + ".txt",white_dist,fmt='%.5f')
        np.savetxt("" + surfdir + "gray_dist" + hemi + ".txt",gray_dist,fmt='%.5f')
    else :
        print("Found vectors, not recalculating")
    for f,fwhm in enumerate(fwhms):
        if not os.path.isfile(os.path.join(surfdir, "smoothed_z_"+str(fwhm) + hemi + ".txt")):
            print("smoothing vectors")
            os.system("depth_potential -smooth " + str(fwhm) + " " + surfdir + "Normalized_vector_" + hemi + "_x.txt " + os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj ')+ surfdir + "smoothed_x_"+str(fwhm) + hemi + ".txt")
            os.system("depth_potential -smooth " + str(fwhm) + " " + surfdir + "Normalized_vector_" + hemi + "_y.txt " + os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj ') + surfdir + "smoothed_y_"+str(fwhm) + hemi + ".txt")
            os.system("depth_potential -smooth " + str(fwhm) + " " + surfdir + "Normalized_vector_" + hemi + "_z.txt " + os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj ') + surfdir + "smoothed_z_"+str(fwhm) + hemi + ".txt")
        if not os.path.isfile(os.path.join(surfdir, "smoothed_gray_dist"+str(fwhm) + hemi + ".txt")):
            os.system("depth_potential -smooth " + str(fwhm) + " " + surfdir + "white_dist" + hemi + ".txt " 
                  + os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj ')+ surfdir + "smoothed_white_dist"+str(fwhm) + hemi + ".txt")
            os.system("depth_potential -smooth " + str(fwhm) + " " + surfdir + "gray_dist" + hemi + ".txt "
                  + os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj ')+ surfdir + "smoothed_gray_dist"+str(fwhm) + hemi + ".txt")

        print( 'reloading vectors')
        Nx=np.loadtxt("" + surfdir + "smoothed_x_"+str(fwhm) + hemi + ".txt")
        Ny=np.loadtxt("" + surfdir + "smoothed_y_"+str(fwhm) + hemi + ".txt")
        Nz=np.loadtxt("" + surfdir + "smoothed_z_"+str(fwhm) + hemi + ".txt")
        
        white_dist = np.loadtxt(os.path.join(surfdir,"smoothed_white_dist"+str(fwhm) + hemi + ".txt"))
        gray_dist = np.loadtxt(os.path.join(surfdir,"smoothed_gray_dist"+str(fwhm) + hemi + ".txt"))
        
        SmoothVector = np.transpose(np.vstack((Nx,Ny,Nz)))
        SmoothNormVector=SmoothVector/np.linalg.norm(SmoothVector,axis=1).reshape((len(SmoothVector),1))
        
        #import perfect normals for comparison
        print("calculating angles")
        angles=norm_differences(SmoothNormVector,mid_norms,medial_wall)
        anglesmean[h,f]=np.mean(angles)
        #anglesskew[h,f]=skew(angles)

        if not isinstance(g,dict):
            g=io.load_mesh_geometry(os.path.join(surfdir, 'equi_iso_0.5_up_'+hemi+'.obj'))
        g['coords'] = mid_xyz + SmoothNormVector* np.array([gray_dist]).T
        io.save_mesh_geometry(os.path.join(surfdir, 'tmp_'+str(fwhm)+'_gray_'+hemi+'_up_iso.obj'),g)
        # normals close to pial surface
        os.system('average_objects '+ os.path.join(surfdir, 'tmp_'+str(fwhm)+'_gray_'+hemi+'_up_iso.obj ')+ os.path.join(surfdir, 'tmp_'+str(fwhm)+'_gray_'+hemi+'_up_iso.obj'))
        _, gray_xyz, gray_norms, _ = st.import_surface(os.path.join(surfdir, 'tmp_'+str(fwhm)+'_gray_'+hemi+'_up_iso.obj'))
        gray_angles = norm_differences(SmoothNormVector,gray_norms,medial_wall)
        grayanglesmean[h,f]=np.mean(gray_angles)
        grayanglesstd[h,f]=np.std(gray_angles)

        g['coords'] = mid_xyz - SmoothNormVector* np.array([white_dist]).T
        io.save_mesh_geometry(os.path.join(surfdir, 'tmp_'+str(fwhm)+'_white_'+hemi+'_up_iso.obj'),g)
        os.system('average_objects '+ os.path.join(surfdir, 'tmp_'+str(fwhm)+'_white_'+hemi+'_up_iso.obj ')+ os.path.join(surfdir, 'tmp_'+str(fwhm)+'_white_'+hemi+'_up_iso.obj'))
        _, white_xyz, white_norms, _ = st.import_surface(os.path.join(surfdir, 'tmp_'+str(fwhm)+'_white_'+hemi+'_up_iso.obj'))
        white_angles = norm_differences(SmoothNormVector,white_norms,medial_wall)
        whiteanglesmean[h,f]=np.mean(white_angles)
        whiteanglesstd[h,f]=np.std(white_angles)
        os.system("make_prism_mesh "+os.path.join(surfdir, 'tmp_'+str(fwhm)+'_gray_'+hemi+'_up_iso.obj ')
                 + os.path.join(surfdir, 'tmp_'+str(fwhm)+'_white_'+hemi+'_up_iso.obj ')+
                  os.path.join(surfdir, 'tmp_'+str(fwhm)+'_prism_'+hemi+'_up_iso.obj'))
        gray_intersect[h,f]=int(subprocess.check_output('check_self_intersect '+os.path.join(surfdir, 'tmp_'+str(fwhm)+'_gray_'+hemi+'_up_iso.obj'),shell=True).split()[-1])
        white_intersect[h,f]=int(subprocess.check_output('check_self_intersect '+os.path.join(surfdir, 'tmp_'+str(fwhm)+'_white_'+hemi+'_up_iso.obj'),shell=True).split()[-1])
        
np.savez('intersection_stats.npz',gray_intersect=gray_intersect,
         white_intersect=white_intersect,anglesmean=anglesmean,
         grayanglesmean=grayanglesmean,whiteanglesmean=whiteanglesmean,
         grayanglesstd=grayanglesstd, whiteanglesstd=whiteanglesstd,
         fwhms=fwhms)

