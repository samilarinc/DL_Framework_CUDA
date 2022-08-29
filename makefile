all:
	nvcc \
	-I /mnt/c/Users/pc/CUDA/framework/Layers -I /mnt/c/Users/pc/CUDA/framework/Optimizers -I /mnt/c/Users/pc/CUDA/framework/Regularizers \
	-I /mnt/c/Users/pc/CUDA/framework/Initializers \
	Layers/src/Base.cu Layers/src/Dense.cu Layers/src/Conv.cu \
	Optimizers/src/Optimizer.cu Optimizers/src/SGD.cu Optimizers/src/Adam.cu \
	Regularizers/src/Regularizer.cu Regularizers/src/L1.cu Regularizers/src/L2.cu \
	Initializers/src/Initializer.cu Initializers/src/Constant.cu \
	tester.cu -o test
	
	./test

clean:
	rm test