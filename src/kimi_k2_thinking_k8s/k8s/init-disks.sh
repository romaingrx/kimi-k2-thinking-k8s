for i in {1..6}; do sudo pvcreate /dev/nvme${i}n1; done

sudo vgcreate data_vg /dev/nvme{1..6}n1

sudo lvcreate -l 100%FREE -n data_lv data_vg

sudo mkfs.ext4 /dev/data_vg/data_lv
sudo mkdir /data
sudo mount /dev/data_vg/data_lv /data
sudo chown $(whoami):$(whoami) /data
