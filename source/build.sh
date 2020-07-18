
TF_INCLUDE=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIBRARY=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') 
#g++ -std=c++11 -shared decode.cpp decode_op.cpp -o ./decode_op.so -fPIC -I$TF_INCLUDE -I$TF_INCLUDE/external/nsync/public -L $TF_LIBRARY -ltensorflow_framework  -lmkl_rt -liomp5 -O2
g++ -std=c++11 -shared decode.cpp encode.cpp translate_op.cpp -o ./lib/translate_op.so -fPIC -I$TF_INCLUDE -I$TF_INCLUDE/external/nsync/public -L$TF_LIBRARY -ltensorflow_framework  -lmkl_rt -liomp5 -O2 --verbose
#icpc -std=c++11 -shared decode.cpp encode.cpp translate_op.cpp -o ./translate_op.so -fPIC -I$TF_INCLUDE -I$TF_INCLUDE/external/nsync/public -L$TF_LIBRARY -ltensorflow_framework  -lmkl_rt -liomp5 -O2 --verbose

#g++ -std=c++11 -shared decode.cpp encode.cpp translate_op.cpp -o ./translate_op.so -fPIC -I/opt/app/.venv/lib64/python3.6/site-packages/tensorflow/include:/opt/intel/mkl/include/  -I$TF_INCLUDE/external/nsync/public -L$TF_LIBRARY -ltensorflow_framework  -lmkl_rt -liomp5 -O2 --verbose
