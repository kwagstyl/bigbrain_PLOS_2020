surfdir=/data1/users/kwagstyl/bigbrain/NeuralNetworks/surfdir/
hemis="left right"
pydir=/data1/users/kwagstyl/KWquarantines/
for hemi in $hemis
do
echo "defragging label"
python "$pydir"surface_defrag.py "$surfdir"archi_"$hemi".txt "$surfdir"white_"$hemi"_327680.obj "$surfdir"defragged_archi_"$hemi".txt
echo "reducing white surface"
python "$pydir"label2obj_reduce.py "$surfdir"white_"$hemi"_327680.obj "$surfdir"defragged_archi_"$hemi".txt "$surfdir"white_archi_"$hemi".obj
echo "reducing gray surface"
python "$pydir"label2obj_reduce.py "$surfdir"gray_"$hemi"_327680.obj "$surfdir"defragged_archi_"$hemi".txt  "$surfdir"gray_archi_"$hemi".obj
echo "making layer obj from file"
python "$pydir"make_layer.py "$surfdir"white_archi_"$hemi".obj "$surfdir"gray_archi_"$hemi".obj "$surfdir"combi_archi_"$hemi".obj
done

objconcat "$surfdir"combi_archi_left.obj "$surfdir"combi_archi_right.obj none none "$surfdir"combi_archi_both.obj none

surface_mask2 -binary_mask /data1/users/kwagstyl/bigbrain/volumes/full8_200um.mnc "$surfdir"combi_archi_both.obj "$surfdir"archi_mask.mnc


volume_object_evaluate -nearest_neighbour /data2/blocks40/legend1000.mnc "$surfdir"combi_archi_both.obj "$surfdir"blocks_archi.txt

