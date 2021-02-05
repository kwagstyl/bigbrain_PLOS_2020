


import os, subprocess
import numpy as np
from profile_surface import get_neighbours, expand_to_fill, indices2surfaces
import io_mesh as io
import pandas as pd




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
        if surf_filename[-4:]=='.obj':
            surf_filename_sm=surf_filename
        else:
            surf_filename_sm=surf_filename + str(k)+'.obj'
        np.savetxt('/tmp/tmp.txt', overlay, fmt='%f')
        print('smoothing surface '+str(k))
        subprocess.call('depth_potential -smooth '+ str(fwhm)+' /tmp/tmp.txt '+ surf_filename_sm +' /tmp/smtmp.txt',shell=True)
        new_overlay = np.loadtxt('/tmp/smtmp.txt')
        new_values[k] = new_overlay
    if shrink:
        return new_values[0]
    else:
        if flipped==True:
            return new_values.T
        return new_values
    

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

def calculate_thickness(inner_surf,outer_surf,tfile):
    subprocess.call('cortical_thickness -tfs '+ inner_surf + ' '+outer_surf+' '+tfile_name,shell=True)
    return


# In[5]:


#running this box should be enough
resolution=20

surfdir='/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/surfs_'+str(resolution)+'microns/'
datadir='/data1/users/kwagstyl/bigbrain/NeuralNetworks/BigBrainData/'
tdir='/data1/users/kwagstyl/bigbrain/Hierarchy/'
hemis=["right","left"]
for hemi in hemis:
    layer=0
    indices=np.loadtxt(os.path.join(surfdir,'indices'+hemi+'_'+str(resolution)+'_nonzeros.txt'))
    smoothed_indices=surface_smoothing(indices,os.path.join(surfdir,'august2018_'+hemi+'_'+str(resolution)+'_layer'),fwhm=1)
    smoothed_indices=np.round(smoothed_indices).astype(int)
    smoothed_indices=np.sort(smoothed_indices)

    clipped=np.clip(smoothed_indices,0,199)
    indices2surfaces(os.path.join(datadir,'full8_1000um.mnc_'+hemi+'_profiles_raw.txt'),
                     clipped,os.path.join(surfdir,'august2018_'+hemi+'_'+str(resolution)+'_layer4.obj'),
                    os.path.join(surfdir, 'surfsmoothed_august2018_'+hemi+'_'+str(resolution)))

    #taubin smoothing
    for n in range(7):   
        surface=os.path.join(surfdir, 'surfsmoothed_august2018_'+hemi+'_'+str(resolution))+'_layer'+str(n)+'.obj'
        surface_new=os.path.join(surfdir, 'sm_20_surfsmoothed_august2018_'+hemi+'_'+str(resolution)+'_layer'+str(n)+'.obj')
        subprocess.call('/data1/users/kwagstyl/quarantines/Linux-x86_64/bin/adapt_object_mesh_taubin '+surface+' '+surface_new+' 9999999999 20',shell=True)
    #calculate thicknesses
    outer_surf=os.path.join(surfdir, 'sm_20_surfsmoothed_august2018_'+hemi+'_'+str(resolution)+'_layer0.obj')
    inner_surf=os.path.join(surfdir, 'sm_20_surfsmoothed_august2018_'+hemi+'_'+str(resolution)+'_layer6.obj')
    tfile_name=os.path.join(tdir,'thickness_'+hemi+'_total.txt')
    calculate_thickness(inner_surf,outer_surf,tfile_name)
    for n in np.arange(6):
        outer_surf=os.path.join(surfdir, 'sm_20_surfsmoothed_august2018_'+hemi+'_'+str(resolution)+'_layer'+str(n)+'.obj')
        inner_surf=os.path.join(surfdir, 'sm_20_surfsmoothed_august2018_'+hemi+'_'+str(resolution)+'_layer'+str(n+1)+'.obj')
        tfile_name=os.path.join(tdir,'thickness_'+hemi+'_layer'+str(n+1)+'.txt')
        calculate_thickness(inner_surf,outer_surf,tfile_name)


