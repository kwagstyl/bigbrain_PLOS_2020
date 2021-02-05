#generate a mid surface that is close to pia in sulci and white in gyri to avoid self intersections

import numpy as np
import subprocess
import io_mesh

surfdir='/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/'

hemis=['right', 'left']
for hemi in hemis:
#subprocess.call('depth_potential -depth_potential -alpha 0.05 mid_'+hemi+'_327680.obj depth_'+hemi+'.txt', shell=True)
    subprocess.call('depth_potential -mean_curvature -alpha 0.05 '+surfdir+'mid_'+hemi+'_327680.obj '+surfdir+'curvature_'+hemi+'.txt', shell=True)
    subprocess.call('depth_potential -smooth 1 '+surfdir+'curvature_'+hemi+'.txt '+surfdir+'mid_'+hemi+'_327680.obj '+surfdir+'smcurvature_'+hemi+'.txt', shell=True)
    #depth = np.loadtxt('depth_'+hemi+'.txt')
#normalise values between 0 and 1
    curv = np.loadtxt(''+surfdir+'smcurvature_'+hemi+'.txt')
    min_value=np.mean(curv)-2*np.std(curv)
    curv = (curv - min_value)
    max_value=np.mean(curv)+2*np.std(curv)
    curv = np.array([curv/max_value]).T
    np.clip(curv, 0,1,out=curv)
#load in surfaces
    g=io_mesh.load_mesh_geometry(''+surfdir+'gray_' + hemi + '_327680.obj')
    w=io_mesh.load_mesh_geometry(''+surfdir+'white_' + hemi + '_327680.obj')
    mid = g['coords']*(1-curv) + w['coords']*curv
    g['coords'] = mid
    io_mesh.save_mesh_geometry(''+surfdir+'weighted_mid_' + hemi + '_327680.obj',g)

