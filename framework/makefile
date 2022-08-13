all:
	nvcc -I /mnt/c/Users/pc/CUDA/framework/Layers -I /mnt/c/Users/pc/CUDA/framework/Optimizers -I /mnt/c/Users/pc/CUDA/framework/Regularizers \
	Layers/src/Base.cu Layers/src/Dense.cu Layers/src/Conv.cu Optimizers/src/Optimizer.cu Optimizers/src/SGD.cu Regularizers/src/L1.cu Regularizers/src/L2.cu \
	Regularizers/src/Regularizer.cu \
	tester.cu -o test
	
	./test

clean:
	rm test