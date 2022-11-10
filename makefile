all:
	nvcc -Xcompiler -fPIC -Xcompiler -shared -I/usr/include/python3.8 -I/home/samil/.local/lib/python3.8/site-packages/pybind11/include \
	-I /mnt/c/Users/pc/repos/CUDA/Layers -I /mnt/c/Users/pc/repos/CUDA/Optimizers -I /mnt/c/Users/pc/repos/CUDA/Regularizers \
	Layers/src/Base.cu Layers/src/Dense.cu \
	Optimizers/src/Optimizer.cu Optimizers/src/SGD.cu Optimizers/src/Adam.cu \
	Regularizers/src/Regularizer.cu Regularizers/src/L1.cu Regularizers/src/L2.cu \
	bind_config.cu -o FullyConnected.cpython-38-x86_64-linux-gnu.so

clean:
	rm -f *.so