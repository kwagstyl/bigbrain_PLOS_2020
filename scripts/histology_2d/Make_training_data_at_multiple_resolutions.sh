#create training data at lower resolutions. 
#to run this script, you need to have already run the main create training data scripts to generate the profiles etc.

#SliceDir="/data1/users/kwagstyl/bigbrain/Slice_Verification/"
#Slices="1066 2807 3300 3863 4366 4892 5431 1582 1600 4080"
#Slices="1066 2807 3300 3863 4366 4892 5431 1582 1600 4080 0301 1066v1"
Slices="1066 2807 3300 3863 4366 4892 5431 0301 1066v1 1582 1600 4080 3380 5431a 6316"
#Slices="4080"
#original section: 1066 2807 3300 3863 4366 4892 5431
#added in march 2018: 1582 1600 4080
#V1 only 0301 1066v1
#Slices="0301
#NOTE TO SELF: FOR SOME REASON, ADDING THE SLICES BELOW WORSENED NETWORK PERFORMANCE
#CONSIDER REGENERATING TRAINING DATA WITHOUT THESE? CHECK TO MAKE SURE REGISTRATION AND SEGMENTATION ARE OK
#Slices="3380 5431a 6316"
#create manually labelled sample

SD="/data1/users/kwagstyl/bigbrain/NeuralNetworks/"
TrainingDir="$SD"TrainingData/
PyDir="/data1/users/kwagstyl/KWquarantines/"
resolutions="20 40 100 200 300 400 1000"
bindir="/data1/users/kwagstyl/quarantines/Linux-x86_64/bin/"

#resolutions="20"
#resolutions="100 200 300 400 1000"
#resolutions="100"
for s in $Slices
  do
echo "running section :"
echo "$s"
#keep cls label for duplicate sections
cls_label=$s
if [[ "$s" != "1582" && "$s" != "1600" && "$s" != "4080" && "$s" != "3380" && "$s" != "5431a" && "$s" != "6316" && "$s" != "0301" && "$s" != "1066v1" ]]

then
    SliceDir=/data1/users/kwagstyl/bigbrain/Slice_Verification/Slice_"$s"/
    NNSliceDir="$TrainingDir"Slice_"$s"/
elif [[ "$s" != "1582" && "$s" != "1600" && "$s" != "4080" && "$s" != "3380" && "$s" != "5431a" && "$s" != "6316" ]]
then
#V1 sections

 if [[ "$s" == "1066v1" ]]
    then
        s="1066"
    fi
    SliceDir=/data1/users/kwagstyl/bigbrain/Slice_Verification/V1/Slice_"$s"/
    NNSliceDir="$TrainingDir"V1/Slice_"$s"/
else
#march 2018 sections
 if [[ "$s" == "5431a" ]]
    then
        s="5431"
    fi
    SliceDir=/data1/users/kwagstyl/bigbrain/Slice_Verification/new_sections_03-18/Slice_"$s"/
    NNSliceDir="$TrainingDir"new_sections_03-18/Slice_"$s"/
fi
for r in $resolutions
do
echo "processing "$r"um resolution"
volumedir="$NNSliceDir"resolution_"$r"um/
if [ ! -d "$volumedir" ]; then
mkdir "$volumedir"
fi

blockdir="$NNSliceDir"block_blocks/
if [ ! -d "$blockdir" ]; then
mkdir "$blockdir"
fi

cd "$NNSliceDir"


if [[ $r == "40" ]] || [[ $r == "20" ]]  ; then

rm glim
echo " /data1/users/kwagstyl/bigbrain/volumes/legend1000.mnc" > "$NNSliceDir"glim
echo " /data1/users/kwagstyl/bigbrain/volumes/legend1000.mnc" >> "$NNSliceDir"glim
"$bindir"print_world_values_nearest glim all_profiles_3d.dat all_profiles_3d_blocks.dat >/dev/null
bn="$(python "$PyDir"convert_printed_intensities_NN_print_blocks.py all_profiles_3d_blocks.dat all_profiles_3d_blocks_data.dat)"

#clear slices
rm slice_blocks.mnc slice_blocks_geo.mnc "$volumedir"slice_blocks.mnc "$volumedir"slice_blocks_geo.mnc

#crop section for each

for b in $bn;
  do
while [[ ${#b} -lt 4 ]] ; do
b="0${b}"
done
cd  "$volumedir"

if [[ $r == "40" ]];
then
block="/data2/blocks40/"
inv="-inv"
else
block="/data1/users/kwagstyl/bigbrain/blocks20/"
inv=""
fi

#make blank full sized section to fill in with bits of blocks
xstart="$(mincinfo -attvalue xspace:start "$block"block"$r"-0001"$inv".mnc)"
zstart="$(mincinfo -attvalue zspace:start "$block"block"$r"-0001"$inv".mnc)"
xel=6572
zel=5711
ystart=$( bc <<< "-70 + ("$s"-1) * 0.02 ")
yel=1

#if [ ! -f block20-"$b"_slice.mnc ]; then
mincresample -clobber -quiet -start $xstart $ystart $zstart -nelements $xel $yel $zel "$block"block"$r"-"$b""$inv".mnc "$volumedir"block"$r"-"$b"_slice.mnc
mincresample -clobber -quiet -start $xstart $ystart $zstart -nelements $xel $yel $zel "$block"block"$r"-"$b""$inv"_geo.mnc "$volumedir"block"$r"-"$b"_geo_slice.mnc
#fi

if [ ! -f "$volumedir"slice_blocks.mnc ]; then
cp "$volumedir"block"$r"-"$b"_slice.mnc "$volumedir"slice_blocks.mnc
cp "$volumedir"block"$r"-"$b"_geo_slice.mnc "$volumedir"slice_blocks_geo.mnc
fi
minccalc -clobber -quiet -expr 'if(A[0]>0) {A[0]} else {A[1]}' block"$r"-"$b"_slice.mnc slice_blocks.mnc tmp.mnc
mv tmp.mnc slice_blocks.mnc
#make minislab otherwise slight coordinate errors mess things up

minccalc -clobber -quiet -expr 'if(A[0]>0) {A[0]} else {A[1]}' block"$r"-"$b"_geo_slice.mnc slice_blocks_geo.mnc tmp.mnc
mv tmp.mnc slice_blocks_geo.mnc


  done
#fi

#cd ..

rm glimblock
echo "  "$volumedir"slice_blocks.mnc " > glimblock
echo "  "$volumedir"slice_blocks_geo.mnc " >> glimblock
"$bindir"print_world_values_nearest glimblock "$NNSliceDir"all_profiles_3d.dat concat_values_blocks.dat >/dev/null
python "$PyDir"convert_printed_intensities_NN.py concat_values_blocks.dat concat_blocks_data.dat
python "$PyDir"columns_2_profiles.py concat_blocks_data.dat "$volumedir"training_"$cls_label"_"$r"_raw.txt "$volumedir"training_"$cls_label"_"$r"_geo.txt 200

else

#volume data
cd "$volumedir"
volumes="/data1/users/kwagstyl/bigbrain/volumes/"

rm glimblock
echo " "$volumes"full8_"$r"um.mnc " > glimblock
echo " "$volumes"full8_"$r"um_geo.mnc " >> glimblock
"$bindir"print_world_values_nearest glimblock "$NNSliceDir"all_profiles_3d.dat concat_values_blocks.dat >/dev/null
python "$PyDir"convert_printed_intensities_NN.py concat_values_blocks.dat concat_blocks_data.dat
python "$PyDir"columns_2_profiles.py concat_blocks_data.dat "$volumedir"training_"$cls_label"_"$r"_raw.txt "$volumedir"training_"$cls_label"_"$r"_geo.txt 200


fi

cd ..
if [ ! -d "$TrainingDir"TrainingData_lowres ]; then
mkdir "$TrainingDir"TrainingData_lowres
fi
cd "$TrainingDir"TrainingData_lowres
#take classified profiles and masks from blockdir as they're the same independent of resolution
python "$PyDir"clean_up_profiles_filter_sixlayers.py "$blockdir"training_"$cls_label"_cls.txt "$volumedir"training_"$cls_label"_"$r"_raw.txt "$volumedir"training_"$cls_label"_"$r"_geo.txt \
"$blockdir"training_"$cls_label"_masks.txt training_"$cls_label"_cls.txt training_"$cls_label"_"$r"_raw.txt training_"$cls_label"_"$r"_geo.txt \
training_"$cls_label"_masks.txt "$NNSliceDir"filtered_indices.txt

# change white to highest index, not zeros.
mv training_"$cls_label"_cls.txt prewhite.txt
python "$PyDir"convert_profiles_white.py prewhite.txt training_"$cls_label"_cls.txt
#create combined training dataset
done
done

cd "$TrainingDir"TrainingData_lowres
for r in $resolutions
do
rm training_"$r"_raw.txt training_"$r"_geo.txt
cat training*_"$r"_geo.txt > training_"$r"_geo.txt
cat training*_"$r"_raw.txt > training_"$r"_raw.txt
done
rm training_cls.txt  training_regions.txt
cat training*cls.txt > training_cls.txt
cat training*masks.txt > training_regions.txt
