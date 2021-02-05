
##script for turning surfaces into a voxel-wise segmentation
voldir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/voxel_dir/
outdir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/voxel_dir/sections/
surfdir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/surfs_20microns/
## set coordinates
#for section in $(seq -f "%04g" 1 7404)
#do
section=$1
volname=pm"$section"o.mnc

#Coronal Axial Sagittal
plane=$2 
echo $section
rm "$voldir""$volname"
#download from the ftp
wget  ftp://bigbrain.loris.ca/BigBrainRelease.2015/2D_Final_Sections/"$plane"/Minc/pm"$section"o.mnc \
-P  "$voldir"
#on the FTP y is always set to zero. Others are correct.
#reset y either to coronal coordinate or -70.01
#Inverse is s= (coord-$start)/step
y_coord=$( bc <<< "-70.01 + ("$section"-1) * 0.02 ")
x_coord=$( bc <<< "-70.6666 + ("$section"-1) * 0.0211667 ")
z_coord=$( bc <<< "-58.7777 + ("$section"-1) * 0.0211667 ")
if [ "$plane" == "Coronal" ]
then
minc_modify_header -dinsert yspace:start="$y_coord" "$voldir""$volname"
elif [ "$plane" == "Axial" ]
then 
#flipyz for axial due to dodgy bug in surface_mask2
y_coord="-70.01"
minc_modify_header -dinsert yspace:start="$y_coord" "$voldir""$volname"
cp "$voldir""$volname" "$voldir"tmp_"$volname"
/data1/users/kwagstyl/bigbrain/NeuralNetworks/scripts/histology_2d/flipyz.pl "$voldir"tmp_"$volname"
 
else
y_coord="-70.01"
minc_modify_header -dinsert yspace:start="$y_coord" "$voldir""$volname"
fi

#loop through layers. axial done flipped then flipped back, due to bug in surface_mask2
if [ "$plane" == "Axial" ]
then 
for s in {6..0}
do
transform_objects "$surfdir"masked_surfs_april2019_right_20_layer"$s".obj "$outdir"flipyz.xfm "$surfdir"right_flipped.obj
transform_objects "$surfdir"masked_surfs_april2019_left_20_layer"$s".obj "$outdir"flipyz.xfm "$surfdir"left_flipped.obj

surface_mask2 -binary "$voldir"tmp_"$volname" "$surfdir"right_flipped.obj "$outdir"tmp_right.mnc
surface_mask2 -binary "$voldir"tmp_"$volname" "$surfdir"left_flipped.obj "$outdir"tmp_left.mnc
minccalc -clobber -int -quiet -expression 'if(A[0]>0.2) {1} else if(A[1]>0.2) {1}' "$outdir"tmp_left.mnc \
"$outdir"tmp_right.mnc "$outdir"tmp"$s".mnc


done
minccalc -clobber -int -quiet -expression '{A[0] + A[1] + A[2] + A[3] + A[4] + A[5] + A[6]}' "$outdir"tmp0.mnc \
"$outdir"tmp1.mnc "$outdir"tmp2.mnc "$outdir"tmp3.mnc "$outdir"tmp4.mnc "$outdir"tmp5.mnc "$outdir"tmp6.mnc \
"$outdir"segmentation"$section".mnc
/data1/users/kwagstyl/bigbrain/NeuralNetworks/scripts/histology_2d/flipyz.pl "$outdir"segmentation"$section".mnc
else
for s in {6..0}
do
echo $s
surface_mask2 -binary_mask "$voldir""$volname" "$surfdir"masked_surfs_april2019_left_20_layer"$s".obj \
"$outdir"tmp_left.mnc

surface_mask2 -binary_mask "$voldir""$volname" "$surfdir"masked_surfs_april2019_right_20_layer"$s".obj \
"$outdir"tmp_right.mnc

minccalc -clobber -int -quiet -expression 'if(A[0]>0.2) {1} else if(A[1]>0.2) {1}' "$outdir"tmp_left.mnc \
"$outdir"tmp_right.mnc "$outdir"tmp"$s".mnc

done

minccalc -clobber -int -quiet -expression '{A[0] + A[1] + A[2] + A[3] + A[4] + A[5] + A[6]}' "$outdir"tmp0.mnc \
"$outdir"tmp1.mnc "$outdir"tmp2.mnc "$outdir"tmp3.mnc "$outdir"tmp4.mnc "$outdir"tmp5.mnc "$outdir"tmp6.mnc \
"$outdir"segmentation"$section".mnc

fi
