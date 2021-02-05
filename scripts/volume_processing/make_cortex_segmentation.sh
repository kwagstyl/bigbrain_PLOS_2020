
##script for turning surfaces into a voxel-wise segmentation
voldir=/data1/users/kwagstyl/bigbrain/volumes/
outdir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/voxel_dir/whole_brain/
surfdir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/surfs_20microns/
volname=full8_300um_geo.mnc
mkdir $outdir

#layer1
s=1
surface_mask2 -binary_mask "$voldir""$volname" "$surfdir"masked_surfs_april2019_left_20_layer"$s".obj \
"$outdir"tmp_left.mnc

surface_mask2 -binary_mask "$voldir""$volname" "$surfdir"masked_surfs_april2019_right_20_layer"$s".obj \
"$outdir"tmp_right.mnc

minccalc -clobber -int -quiet -expression 'if(A[0]>0.2) {1} else if(A[1]>0.2) {1}' "$outdir"tmp_left.mnc \
"$outdir"tmp_right.mnc "$outdir"tmp"$s".mnc

#white

s=6
surface_mask2 -binary_mask "$voldir""$volname" "$surfdir"masked_surfs_april2019_left_20_layer"$s".obj \
"$outdir"tmp_left.mnc

surface_mask2 -binary_mask "$voldir""$volname" "$surfdir"masked_surfs_april2019_right_20_layer"$s".obj \
"$outdir"tmp_right.mnc

minccalc -clobber -int -quiet -expression 'if(A[0]>0.2) {1} else if(A[1]>0.2) {1}' "$outdir"tmp_left.mnc \
"$outdir"tmp_right.mnc "$outdir"tmp"$s".mnc


minccalc -clobber -int -quiet -expression '{A[0] + A[1]}' \
"$outdir"tmp1.mnc  "$outdir"tmp6.mnc \
"$outdir"segmentation"$section".mnc
