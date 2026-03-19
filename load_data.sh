set -e

zenodo_get 3775983
tar -xvf imojie_data.tar.gz

mkdir dataset
mv data dataset/data