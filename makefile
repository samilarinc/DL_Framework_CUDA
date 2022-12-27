all:
	nvcc -Xcompiler -fPIC -Xcompiler -shared -I /usr/include/python3.8 -I /home/samil/.local/lib/python3.8/site-packages/pybind11/include \
	-I ./ -I /usr/local/cuda/include Tensor.cu bind_config.cu -o pyflow.cpython-38-x86_64-linux-gnu.so

clean:
	rm -f *.so