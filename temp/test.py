from FullyConnected import DenseTr, Dense, Tensor
layer = Dense(8, 5)
t = Tensor([0,1,2,3,4,5,6,7])
print(t)
print("Here")
out = layer.forward(t)
print("Here")
print(t)
print(out)