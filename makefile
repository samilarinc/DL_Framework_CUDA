all:
	nvcc \
	-I /mnt/c/Users/pc/CUDA/Layers -I /mnt/c/Users/pc/CUDA/Optimizers -I /mnt/c/Users/pc/CUDA/Regularizers \
	-I /mnt/c/Users/pc/CUDA/Initializers -I /mnt/c/Users/pc/CUDA/Activations -I /mnt/c/Users/pc/CUDA/Loss \
	Layers/src/Base.cu Layers/src/Dense.cu Layers/src/Conv.cu \
	Optimizers/src/Optimizer.cu Optimizers/src/SGD.cu Optimizers/src/Adam.cu \
	Regularizers/src/Regularizer.cu Regularizers/src/L1.cu Regularizers/src/L2.cu \
	Initializers/src/Initializer.cu Initializers/src/Constant.cu \
	Activations/src/Activation.cu Activations/src/Sigmoid.cu \
	Loss/src/Loss.cu Loss/src/CrossEntropyLoss.cu \
	tester.cu -o test
	
	./test

clean:
	rm test