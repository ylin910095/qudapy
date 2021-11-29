conda activate py310
module unload gcc
module load gcc/7.3.0
export LD_LIBRARY_PATH=/data/d10b/users/ylin/quda/build_1080/lib:/data/d10b/users/ylin/quda/build_1080/usqcd/lib:$LD_LIBRARY_PATH
