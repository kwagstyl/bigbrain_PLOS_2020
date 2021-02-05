
##script for turning surfaces into a voxel-wise segmentation
voldir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/voxel_dir/
outdir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/voxel_dir/sections/
surfdir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/surfs_20microns/
volname=dummy_section.mnc
mkdir $outdir
## set coordinates
#for section in $(seq -f "%04g" 1 7404)
#do
section="1066"
if [ -f "$outdir"segmentation"$section".mnc ];
then
echo "$section exists"
else
echo $section
y_coord=$( bc <<< "-70.01 + ("$section"-1) * 0.02 ")
minc_modify_header -dinsert yspace:start="$y_coord" "$voldir""$volname"

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

#done
