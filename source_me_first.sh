conda activate py310
module unload gcc
module load gcc/7.3.0
export LD_LIBRARY_PATH=/data/d10b/users/ylin/quda/build/lib:/data/d10b/users/ylin/quda/build/usqcd/lib:$LD_LIBRARY_PATH
