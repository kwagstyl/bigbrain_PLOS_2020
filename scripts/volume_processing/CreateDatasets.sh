#for some reason this has to be written out in the terminal
PyDir="/data1/users/kwagstyl/bigbrain/NeuralNetworks/scripts/surfaces/"
qPyDir="/data1/users/kwagstyl/KWquarantines/"
#mkdir TrainingData
#mkdir ActualData
DataDir="/data1/users/kwagstyl/bigbrain/NeuralNetworks/BigBrainData/"
surfdir="/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/"
hemis="left right"
hemis="right"
mkdir $DataDir
resolutions="20 40 100 200 300 400 1000"
resolutions="1000 400 300 200 100 40 20"
#resolutions="40 20"
#resolutions="1000"
resolutions="20"
resolutions="1000"
for hemi in $hemis
do
cd "$surfdir"

fwhm="2.0"
#tmp_"$fwhm"_gray_right_up_iso.obj
#expand surfaces
python "$PyDir"ExpandGrayWhiteSurfaces.py tmp_"$fwhm"_gray_"$hemi"_up_iso.obj tmp_"$fwhm"_white_"$hemi"_up_iso.obj 0.5

graysurf="$surfdir"outer_tmp_"$fwhm"_gray_"$hemi"_up_iso.obj
whitesurf="$surfdir"inner_tmp_"$fwhm"_white_"$hemi"_up_iso.obj

for res in $resolutions
do

if [ $res = "20" ] || [ $res = "40" ]; 
then

if [ "$hemi" = "left" ]; then
  tens="03 04 07 8 9 10 12 13 14 15 17 18 19 20 22 \
23 24 27 28 29 32 33 34 35 36 37 38 39 40 41 42 43 \
44 45 47 48 49 50 52 53 54 57 58 59 60 62 63 64 65 \
67 68 69 70 72 73 74 75"
else
  tens="53 54 55 58 59 60 62 63 64 65 67 68 69 70 72 \
73 74 75 77 78 79 80 82 83 84 85 86 87 88 89 90 91 92 93 \
94 95 97 98 99 103 104 107 108 109 110 111 112 113 114 115 \
117 118 119 122 123 124"
fi


for t in $tens; do
if [ "$t" = "08" ]; then
  nt="008"
elif [ "$t" = "09" ]; then
  nt="009"  
else
  printf -v nt "%03d" "$t"
fi

if [ $res = "20" ]; then
blockdir="/data1/users/kwagstyl/bigbrain/blocks20/"
block_geo=block20-0"$nt"_geo.mnc
block=block20-0"$nt".mnc


elif [ $res = "40" ]; then
blockdir="/data2/blocks40/"
block_geo=block40-0"$nt"-inv_geo.mnc
block=block40-0"$nt"-inv.mnc

fi

echo "$block"
/data1/users/kwagstyl/bigbrain/bin/create_cortical_profile -linear -samples 200 "$blockdir""$block" \
"$graysurf" "$whitesurf" "$DataDir""$block"_"$hemi"_profiles.txt


/data1/users/kwagstyl/bigbrain/bin/create_cortical_profile -linear -samples 200 "$blockdir""$block_geo" \
"$graysurf" "$whitesurf" "$DataDir""$block_geo"_"$hemi"_profiles_geo.txt

done
cd "$DataDir"
python "$PyDir"concatenate_profiles.py "$hemi"_profiles.txt WhiteGray_"$hemi".txt
python "$PyDir"concatenate_profiles.py "$hemi"_profiles_geo.txt WhiteGray_"$hemi"_geo.txt
rm *"$hemi"_profiles.txt *"$hemi"_profiles_geo.txt

python "$PyDir"convert_profiles2matrix.py WhiteGray_"$hemi".txt raw_"$hemi"_"$res".txt
python "$PyDir"convert_profiles2matrix.py WhiteGray_"$hemi"_geo.txt geo_"$hemi"_"$res".txt
rm WhiteGray_"$hemi".txt WhiteGray_"$hemi"_geo.txt


else
voldir="/data1/users/kwagstyl/bigbrain/volumes/"
volume="full8_"$res"um.mnc"
volume_geo="full8_"$res"um_geo.mnc"
/data1/users/kwagstyl/bigbrain/bin/create_cortical_profile -linear -samples 200 "$voldir""$volume" \
"$graysurf" "$whitesurf" \
"$DataDir""$volume"_"$hemi"_profiles_raw.txt
/data1/users/kwagstyl/bigbrain/bin/create_cortical_profile -linear -samples 200 "$voldir""$volume_geo" \
"$graysurf" "$whitesurf" \
"$DataDir""$volume_geo"_"$hemi"_profiles_geo.txt

python "$PyDir"convert_profiles2matrix.py "$DataDir""$volume"_"$hemi"_profiles_raw.txt "$DataDir"raw_"$hemi"_"$res".txt
python "$PyDir"convert_profiles2matrix.py "$DataDir""$volume_geo"_"$hemi"_profiles_geo.txt "$DataDir"geo_"$hemi"_"$res".txt

if [ ! "$res" = "1000" ]
then
rm "$DataDir""$volume"_"$hemi"_profiles_raw.txt "$DataDir""$volume_geo"_"$hemi"_profiles_geo.txt
fi
fi


done
done



