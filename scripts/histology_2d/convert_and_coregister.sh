#script for convert manually segmented sections to mnc and registering them to bigbrain

slice_dir="/data1/users/kwagstyl/bigbrain/Slice_Verification/"

#original sections
slices="1066 2807 3300 3863 4366 4892 5431"

cd "$slice_dir"
#python "$slice_dir"generate_profiles.py

for s in $slices
do
cd "$slice_dir"Slice_"$s"
#bash "$slice_dir"convertsixlayers.sh "$s"
bash "$slice_dir"stepssixlayers.sh "$s"
done



#added v1 sections
slice_dir="/data1/users/kwagstyl/bigbrain/Slice_Verification/V1/"
slices="0301 1066"

cd "$slice_dir"
#python "$slice_dir"generate_profiles.py


for s in $slices
do
cd "$slice_dir"Slice_"$s"
#bash "$slice_dir"convertV1.sh "$s"
bash "$slice_dir"steps_alllayers.sh "$s"
done

#sections added in march 17
#make sure latest tifs are in the correct directory. They are initially written to sections/Slice_...	
slice_dir="/data1/users/kwagstyl/bigbrain/Slice_Verification/new_sections_03-18/"
slices="1582 1600 3380 4080 5431 6316"

cd "$slice_dir"
#python "$slice_dir"generate_profiles.py


for slice in $slices;
do
#cd "$slice_dir"Slice_"$slice"
#bash convertsixlayers.sh $slice
bash stepssixlayers.sh $slice
done


