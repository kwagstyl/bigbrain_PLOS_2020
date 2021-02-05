#not used

surfdir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/
orig_surfs=/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/
hemis="left right"

for h in $hemis
do
sphere_surf_interpolate.pl  "$orig_surfs"equi_"$h"_0.50.obj "$surfdir"equi_"$h"_0.50_iso.obj
#adapt_object_mesh_taubin  "$surfdir"smlayer4_"$h"_327680_iso.obj 99999999 5
subdivide_polygons "$surfdir"equi_"$h"_0.50_iso.obj "$surfdir"equi_"$h"_0.50_up_iso.obj
check_self_intersect "$surfdir"equi_"$h"_0.50_up_iso.obj

done

