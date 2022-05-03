


home=$(dirname $(readlink -f "$0"))
cd $home
source "${home}"/settings.sh
#conda init bash
#source ~/.bashrc
#conda activate $env


script=${mode}.py

if [ $mode == train ]
then
  configs=$train_configs$default_train_configs
fi

if [ $mode == test ]
then
  configs=$test_configs$default_test_configs
fi
echo python $script "$configs"

python $script $configs

