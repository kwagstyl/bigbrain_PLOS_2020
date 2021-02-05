#
#SliceDir="/data1/users/kwagstyl/bigbrain/Slice_Verification/"
Slices="1066 2807 3300 3863 4366 4892 5431 0301 1066v1 1582 1600 4080 3380 5431a 6316"
#Slices="1582 1600  3380 5431a 6316"
#Slices="1600"
#original section: 1066 2807 3300 3863 4366 4892 5431
#added in march: 1582 1600 4080
#NOTE TO SELF: FOR SOME REASON, ADDING THE SLICES BELOW WORSENED NETWORK PERFORMANCE
#CONSIDER REGENERATING TRAINING DATA WITHOUT THESE? CHECK TO MAKE SURE REGISTRATION AND SEGMENTATION ARE OK
#Slices="3380 5431a 6316"
# ALSO, now that 5431 is included twice, this might need dealing with, possibly as 5431a
#Slices="1066v1"
#create manually labelled sample
#Slices="3380"
SD="/data1/users/kwagstyl/bigbrain/NeuralNetworks/"
PyDir="/data1/users/kwagstyl/KWquarantines/"
bindir="/data1/users/kwagstyl/quarantines/Linux-x86_64/bin/"
TrainingDir="$SD"TrainingData/
Force="True"
mkdir "$TrainingDir"

for s in $Slices
  do
#cls_label is for the duplicate sections to have a unique name on output
cls_label=$s
if [[ "$s" != "1582" && "$s" != "1600" && "$s" != "4080" && "$s" != "3380" && "$s" != "5431a" && "$s" != "6316" && "$s" != "0301" && "$s" != "1066v1" ]]
then
    SliceDir=/data1/users/kwagstyl/bigbrain/Slice_Verification/Slice_"$s"/
    NNSliceDir="$TrainingDir"Slice_"$s"/

mkdir "$NNSliceDir"
cd "$NNSliceDir"
echo "$s"
#convert coordinates to tags

echo "registering coordinates"
#register coordinates to pm nl space with transform tags
rm  "$NNSliceDir"white_coordinates.txt "$NNSliceDir"gray_coordinates.txt

cat "$SliceDir"Region_*_coordinates_white.txt >> "$NNSliceDir"white_coordinates.txt
cat "$SliceDir"Region_*_coordinates_gray.txt >> "$NNSliceDir"gray_coordinates.txt


python "$PyDir"txt2tag_straight.py "$NNSliceDir"white_coordinates.txt "$NNSliceDir"white_coordinates.tag
python "$PyDir"txt2tag_straight.py "$NNSliceDir"gray_coordinates.txt "$NNSliceDir"gray_coordinates.tag
#register
transform_tags "$NNSliceDir"gray_coordinates.tag "$NNSliceDir"nl"$s".xfm "$NNSliceDir"gray_coordinates_nl.tag
transform_tags "$NNSliceDir"gray_coordinates_nl.tag "$NNSliceDir"pm"$s"_mri.xfm "$NNSliceDir"gray_coordinates_mri.tag
transform_tags "$NNSliceDir"gray_coordinates_mri.tag "$NNSliceDir"pm"$s"_nl.xfm "$NNSliceDir"gray_coordinates_bigbrain.tag

transform_tags "$NNSliceDir"white_coordinates.tag "$NNSliceDir"nl"$s".xfm "$NNSliceDir"white_coordinates_nl.tag
transform_tags "$NNSliceDir"white_coordinates_nl.tag "$NNSliceDir"pm"$s"_mri.xfm "$NNSliceDir"white_coordinates_mri.tag
transform_tags "$NNSliceDir"white_coordinates_mri.tag "$NNSliceDir"pm"$s"_nl.xfm "$NNSliceDir"white_coordinates_bigbrain.tag
      cd $SliceDir
cp pm"$s"_nl.xfm "$NNSliceDir"/
cp pm"$s"_mri.xfm "$NNSliceDir"/
cp nl"$s".xfm "$NNSliceDir"/
cp pm"$s"_nl_grid_0.mnc "$NNSliceDir"/
cp pm"$s"_mri_grid_0.mnc "$NNSliceDir"/
cp nl"$s"_grid_0.mnc "$NNSliceDir"/
cp pm"$s"Mask_nl.mnc "$NNSliceDir"/

if [ ! -f "$NNSliceDir"pm"$s"_nl_classifiedsixlayers.mnc ] || [ $Force = "True" ]; then
      cd $SliceDir
        minccalc -clobber -short -quiet -expression 'if(abs(A[0]-11051)<50) {1} else if(abs(A[0]-21845)<50) {2} else if(abs(A[0]-32896)<0.5) {3} else if(abs(A[0]-43690)<50) {4} else if(abs(A[0]-54741)<50) {5} else if(abs(A[0]-65535)<50) {6} else {0}'     pm"$s"sixlayers_nl.mnc "$NNSliceDir"pm"$s"_nl_classifiedsixlayers.mnc
fi

elif [[ "$s" != "1582" && "$s" != "1600" && "$s" != "4080" && "$s" != "3380" && "$s" != "5431a" && "$s" != "6316" ]]
then
 if [[ "$s" = "1066v1" ]]
    then
        s="1066"
    fi

   TrainingDir="$SD"TrainingData/
    SliceDir=/data1/users/kwagstyl/bigbrain/Slice_Verification/V1/Slice_"$s"/
    NNSliceDir="$TrainingDir"V1/Slice_"$s"/
mkdir "$NNSliceDir"
      cd $SliceDir
if [ ! -f  "$NNSliceDir"pm"$s"_nl_classifiedsixlayers.mnc ] || [ $Force = "True" ] ; then
        minccalc -clobber -short -quiet -expression       'if(abs(A[0]-11)<0.5) {1} else if(abs(A[0]-22)<0.5) {2} else if(abs(A[0]-33.5)<0.5) {3} else if(abs(A[0]-44.5)<0.5) {4} else if(abs(A[0]-55.5)<0.5) {4} else if(abs(A[0]-66.5)<0.5) {4} else if(abs(A[0]-77.5)<0.5) {4} else if(abs(A[0]-88.5)<0.5) {5} else if(abs(A[0]-100)<0.5) {5} else if(abs(A[0]-111)<0.5) {6} else if(abs(A[0]-122)<0.5) {6} else {0}'          pm"$s"_alllayers_nl.mnc "$NNSliceDir"pm"$s"_nl_classifiedsixlayers.mnc
fi
cp pm"$s"_nl.xfm "$NNSliceDir"/
cp pm"$s"_mri.xfm "$NNSliceDir"/
cp nl"$s".xfm "$NNSliceDir"/
cp pm"$s"_nl_grid_0.mnc "$NNSliceDir"/
cp pm"$s"_mri_grid_0.mnc "$NNSliceDir"/
cp nl"$s"_grid_0.mnc "$NNSliceDir"/
#cp pm"$s"Mask_nl.mnc "$NNSliceDir"/
rm  "$NNSliceDir"white_coordinates.txt "$NNSliceDir"gray_coordinates.txt

cat "$SliceDir"Region_*_coordinates_white.txt >> "$NNSliceDir"white_coordinates.txt
cat "$SliceDir"Region_*_coordinates_gray.txt >> "$NNSliceDir"gray_coordinates.txt

python "$PyDir"txt2tag_straight.py "$NNSliceDir"white_coordinates.txt "$NNSliceDir"white_coordinates.tag
python "$PyDir"txt2tag_straight.py "$NNSliceDir"gray_coordinates.txt "$NNSliceDir"gray_coordinates.tag
#register
transform_tags "$NNSliceDir"gray_coordinates.tag "$NNSliceDir"nl"$s".xfm "$NNSliceDir"gray_coordinates_nl.tag
transform_tags "$NNSliceDir"gray_coordinates_nl.tag "$NNSliceDir"pm"$s"_mri.xfm "$NNSliceDir"gray_coordinates_mri.tag
transform_tags "$NNSliceDir"gray_coordinates_mri.tag "$NNSliceDir"pm"$s"_nl.xfm "$NNSliceDir"gray_coordinates_bigbrain.tag

transform_tags "$NNSliceDir"white_coordinates.tag "$NNSliceDir"nl"$s".xfm "$NNSliceDir"white_coordinates_nl.tag
transform_tags "$NNSliceDir"white_coordinates_nl.tag "$NNSliceDir"pm"$s"_mri.xfm "$NNSliceDir"white_coordinates_mri.tag
transform_tags "$NNSliceDir"white_coordinates_mri.tag "$NNSliceDir"pm"$s"_nl.xfm "$NNSliceDir"white_coordinates_bigbrain.tag


else
    if [[ "$s" = "5431a" ]]
    then
        s="5431"
    fi
V1="False"
    TrainingDir="$SD"TrainingData/
    SliceDir=/data1/users/kwagstyl/bigbrain/Slice_Verification/new_sections_03-18/Slice_"$s"/
    NNSliceDir="$TrainingDir"new_sections_03-18/Slice_"$s"/
native=/data1/users/kwagstyl/bigbrain/Slice_Verification/new_sections_03-18/native_sections/
mri=/data1/users/kwagstyl/bigbrain/Slice_Verification/new_sections_03-18/mri_transforms/
nl=/data1/users/kwagstyl/bigbrain/Slice_Verification/new_sections_03-18/nl_transforms/

#registration step
mkdir "$NNSliceDir"
cd "$NNSliceDir"

#register coordinates to pm nl space with transform tags
rm  "$NNSliceDir"white_coordinates.txt "$NNSliceDir"gray_coordinates.txt

cat "$SliceDir"Region_*_coordinates_white.txt >> "$NNSliceDir"white_coordinates.txt
cat "$SliceDir"Region_*_coordinates_gray.txt >> "$NNSliceDir"gray_coordinates.txt
cp "$SliceDir"pm"$s"Mask_nl.mnc "$NNSliceDir"/


python "$PyDir"txt2tag_straight.py "$NNSliceDir"white_coordinates.txt "$NNSliceDir"white_coordinates.tag
python "$PyDir"txt2tag_straight.py "$NNSliceDir"gray_coordinates.txt "$NNSliceDir"gray_coordinates.tag
#register
transform_tags "$NNSliceDir"gray_coordinates.tag "$native"nl"$s".xfm "$NNSliceDir"gray_coordinates_nl.tag
transform_tags "$NNSliceDir"gray_coordinates_nl.tag "$mri"pm"$s".xfm "$NNSliceDir"gray_coordinates_mri.tag
transform_tags "$NNSliceDir"gray_coordinates_mri.tag "$nl"pm"$s".xfm "$NNSliceDir"gray_coordinates_bigbrain.tag

transform_tags "$NNSliceDir"white_coordinates.tag "$native"nl"$s".xfm "$NNSliceDir"white_coordinates_nl.tag
transform_tags "$NNSliceDir"white_coordinates_nl.tag "$mri"pm"$s".xfm "$NNSliceDir"white_coordinates_mri.tag
transform_tags "$NNSliceDir"white_coordinates_mri.tag "$nl"pm"$s".xfm "$NNSliceDir"white_coordinates_bigbrain.tag


if [ ! -f "$NNSliceDir"pm"$s"_nl_classifiedsixlayers.mnc ]  || [ $Force = "True" ] ; then
      cd $SliceDir
        minccalc -clobber -short -quiet -expression 'if(abs(A[0]-11051)<50) {1} else if(abs(A[0]-21845)<50) {2} else if(abs(A[0]-32896)<0.5) {3} else if(abs(A[0]-43947)<1000) {4} else if(abs(A[0]-54741)<50) {5} else if(abs(A[0]-65535)<50) {6} else {0}'     pm"$s"sixlayers_nl.mnc "$NNSliceDir"pm"$s"_nl_classifiedsixlayers.mnc
fi
fi
cd "$NNSliceDir"
#convert tags to dat
rm "$NNSliceDir"white_coordinates_bigbrain.dat  "$NNSliceDir"gray_coordinates_bigbrain.dat
python "$PyDir"tag2csv.py "$NNSliceDir"white_coordinates_bigbrain.tag "$NNSliceDir"white_coordinates_bigbrain.dat
python "$PyDir"tag2csv.py "$NNSliceDir"gray_coordinates_bigbrain.tag "$NNSliceDir"gray_coordinates_bigbrain.dat
#create lines for each pair
#create 3D profiles
python "$PyDir"create_profiles.py "$NNSliceDir"gray_coordinates_bigbrain.dat "$NNSliceDir"white_coordinates_bigbrain.dat "$s" all_profiles_3d.dat


blockdir="$NNSliceDir"block_blocks/
if [ ! -d "$blockdir" ]; then
mkdir "$blockdir"
fi

if [ ! -f "$blockdir"slice_blocks.mnc  ] ; then

rm glim
echo " /data1/users/kwagstyl/bigbrain/volumes/legend1000.mnc" > "$NNSliceDir"glim
echo " /data1/users/kwagstyl/bigbrain/volumes/legend1000.mnc" >> "$NNSliceDir"glim

"$bindir"print_world_values_nearest glim all_profiles_3d.dat all_profiles_3d_blocks.dat >/dev/null
bn="$(python "$PyDir"convert_printed_intensities_NN_print_blocks.py all_profiles_3d_blocks.dat all_profiles_3d_blocks_data.dat)"

#clear slices
rm slice_blocks.mnc slice_blocks_geo.mnc "$blockdir"slice_blocks.mnc "$blockdir"slice_blocks_geo.mnc

#extract slice from blocks

for b in $bn;
  do
while [[ ${#b} -lt 4 ]] ; do
b="0${b}"
done
cd  "$blockdir"

block="/data1/users/kwagstyl/bigbrain/blocks20/"
#make blank full sized section to fill in with bits of blocks
xstart="$(mincinfo -attvalue xspace:start "$block"block20-0001.mnc)"
zstart="$(mincinfo -attvalue zspace:start "$block"block20-0001.mnc)"
xel=$( bc <<< "1482 + 1582 + 1581 + 1582 + 1483")
zel=$( bc <<< "1300 + 1400 + 1399 + 1399 + 1299")
ystart=$( bc <<< "-70 + ("$s"-1) * 0.02 ")
yel=1

#if [ ! -f block20-"$b"_slice.mnc ]; then
mincresample -clobber -quiet -start $xstart $ystart $zstart -nelements $xel $yel $zel "$block"block20-"$b".mnc block20-"$b"_slice.mnc
mincresample -clobber -quiet -start $xstart $ystart $zstart -nelements $xel $yel $zel "$block"block20-"$b"_geo.mnc block20-"$b"_geo_slice.mnc
#fi

if [ ! -f "$blockdir"slice_blocks.mnc ]; then
cp "$blockdir"block20-"$b"_slice.mnc "$blockdir"slice_blocks.mnc
cp "$blockdir"block20-"$b"_geo_slice.mnc "$blockdir"slice_blocks_geo.mnc
fi

minccalc -clobber -quiet -expr 'if(A[0]>0) {A[0]} else {A[1]}' block20-"$b"_slice.mnc slice_blocks.mnc tmp.mnc

mv tmp.mnc slice_blocks.mnc
#make minislab otherwise slight coordinate errors mess things up

minccalc -clobber -quiet -expr 'if(A[0]>0) {A[0]} else {A[1]}' block20-"$b"_geo_slice.mnc slice_blocks_geo.mnc tmp.mnc
mv tmp.mnc slice_blocks_geo.mnc


  done
fi

#cd ..
#align histo to bigbrain space by flipping
ystart=$( bc <<< "-70 + ("$s"-1) * 0.02 ")
cd "$NNSliceDir"

cp "$NNSliceDir"pm"$s"_nl_classifiedsixlayers.mnc "$NNSliceDir"tmp.mnc
minc_modify_header -dinsert zspace:start=$ystart "$NNSliceDir"tmp.mnc
"$TrainingDir"flipyz.pl "$NNSliceDir"tmp.mnc
mincresample -clob -nearest_neighbour -like "$blockdir"slice_blocks.mnc "$NNSliceDir"tmp.mnc "$NNSliceDir"pm"$s"_nl_classifiedsixlayers_aligned.mnc 


cp "$NNSliceDir"pm"$s"Mask_nl.mnc "$NNSliceDir"tmp.mnc
minc_modify_header -dinsert zspace:start=$ystart "$NNSliceDir"tmp.mnc
"$TrainingDir"flipyz.pl "$NNSliceDir"tmp.mnc
mincresample -clob -nearest_neighbour -like "$blockdir"slice_blocks.mnc  "$NNSliceDir"tmp.mnc "$NNSliceDir"pm"$s"Mask_nl_aligned.mnc


cd "$blockdir"
rm glim
rm "$blockdir"glim
echo " "$NNSliceDir"pm"$s"_nl_classifiedsixlayers_aligned.mnc " > "$blockdir"glim
echo " "$NNSliceDir"pm"$s"_nl_classifiedsixlayers_aligned.mnc " >> "$blockdir"glim
"$bindir"print_world_values_nearest glim "$NNSliceDir"all_profiles_3d.dat concat_values.dat >/dev/null
python "$PyDir"convert_printed_intensities_NN.py concat_values.dat concat_values_data.dat
python "$PyDir"columns_2_profiles.py concat_values_data.dat training_"$cls_label"_cls.txt training_"$cls_label"_cls.txt 200

rm glimblock
echo "  "$blockdir"slice_blocks.mnc " > glimblock
echo "  "$blockdir"slice_blocks_geo.mnc " >> glimblock
"$bindir"print_world_values_nearest glimblock "$NNSliceDir"all_profiles_3d.dat concat_values_blocks.dat >/dev/null
python "$PyDir"convert_printed_intensities_NN.py concat_values_blocks.dat concat_blocks_data.dat
python "$PyDir"columns_2_profiles.py concat_blocks_data.dat training_"$cls_label"_raw.txt training_"$cls_label"_geo.txt 200

rm glimmask
echo " "$NNSliceDir"pm"$s"Mask_nl_aligned.mnc " > glimmask
echo " "$NNSliceDir"pm"$s"Mask_nl_aligned.mnc " >> glimmask
"$bindir"print_world_values_nearest glimmask "$NNSliceDir"all_profiles_3d.dat concat_values_masks.dat > /dev/null
python "$PyDir"convert_printed_intensities_NN.py concat_values_masks.dat concat_masks_data.dat



python "$PyDir"clean_up_profiles.py training_"$cls_label"_cls.txt training_"$cls_label"_cls.txt
python "$PyDir"mask_columns_to_vector.py concat_masks_data.dat training_"$cls_label"_masks.txt 200 "$cls_label"

mkdir "$TrainingDir"TrainingData
cd "$TrainingDir"TrainingData
python "$PyDir"clean_up_profiles_filter_sixlayers.py "$blockdir"training_"$cls_label"_cls.txt "$blockdir"training_"$cls_label"_raw.txt "$blockdir"training_"$cls_label"_geo.txt \
"$blockdir"training_"$cls_label"_masks.txt training_"$cls_label"_cls.txt training_"$cls_label"_raw.txt training_"$cls_label"_geo.txt \
training_"$cls_label"_masks.txt "$NNSliceDir"filtered_indices.txt

mv training_"$cls_label"_cls.txt prewhite.txt
python "$PyDir"convert_profiles_white.py prewhite.txt training_"$cls_label"_cls.txt
done

#concatenate training data
cd "$TrainingDir"TrainingData
rm training_20_raw.txt training_20_geo.txt training_cls.txt training_regions.txt
cat training*cls.txt >> training_cls.txt
cat training*masks.txt >> training_regions.txt
cat training_*_geo.txt >> training_20_geo.txt
cat training_*_raw.txt >> training_20_raw.txt
cp training_20_geo.txt "$TrainingDir"TrainingData_lowres/
cp training_20_raw.txt "$TrainingDir"TrainingData_lowres/
