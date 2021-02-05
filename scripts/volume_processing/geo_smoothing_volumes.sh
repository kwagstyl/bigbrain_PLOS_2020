#smoothing volumes. effective fwhm 0.163mm
#above 100um as single volumes, below as blocks. for 20 micron smooth repeat the 40 micron step.

res="100 200 300 400 1000"
vol_dir="/data1/users/kwagstyl/bigbrain/volumes/"
#for r in $res
#do
#geo_smooth 0.0004 6 "$vol_dir"full8_"$r"um.mnc "$vol_dir"full8_"$r"um_geo.mnc
#done


vol_dir="/data2/blocks40/"

cd "$vol_dir"
for f in block*inv.mnc;
do

geo_file=`echo "${f%????}"_geo.mnc`
if [ ! -f $vol_dir$geo_file ];
then
echo "creating " 
echo $vol_dir$geo_file
geo_smooth 0.0004 6 "$vol_dir""$f" "$vol_dir""$geo_file"
fi

done
