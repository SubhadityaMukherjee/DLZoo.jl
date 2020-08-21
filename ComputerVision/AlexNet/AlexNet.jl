
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using CuArrays

# DATA

imgs = MNIST.images()

X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()

Y = onehotbatch(labels, 0:9) |> gpu

# MODEL

## Convinence Functions
"""
Conv -> Relu -> Max Pool with parameters to make the code shorter and less redundant
"""
conv_block(in_channels, out_channels, padding = 1, stride = 2,ks=3) = (
    Conv(
        (ks, ks),
        in_channels => out_channels,
        relu,
        pad = (padding, padding),
        stride = (stride, stride),
    ),
    MaxPool((3, 3), stride = 2),
)

"""
AdaptiveMeanPool(target_out) 
Adaptive mean pooling layer. 'target_out' stands for the size of one layer(channel) of output.
Irrespective of Input size and pooling window size, it returns the target output dimensions.

Taken from : https://github.com/FluxML/Flux.jl/blob/2f8d281748b234b11a79f604f6ee85751724bc7c/src/layers/conv.jl
"""
struct AdaptiveMeanPool{N}
  target_out::NTuple{N, Int}
end

AdaptiveMeanPool(target_out::NTuple{N, Integer}) where N = 
  AdaptiveMeanPool(target_out)

function (m::AdaptiveMeanPool)(x)
  w = size(x, 1) - m.target_out[1] + 1
  h = size(x, 2) - m.target_out[2] + 1
  return meanpool(x, (w, h); pad = (0, 0), stride = (1, 1))
end

m = Chain(
    conv_block(3, 64, 2, 4, 11)...,
    conv_block(64, 192, 2)...,
    Conv((3,3), 192 => 384, pad = 1), 
    Conv((3,3), 384 => 256, pad = 1),
    conv_block(256, 256)...,
    AdaptiveMeanPool((6,6)),
    flatten,
    Dropout,
    Dense(256*6*6, 4096, relu),
    Dropout,
    Dense(4096, 4096, relu), 
    Dense(4096, length(unique(Y))) 
) |> gpu

# TRAINING

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

dataset = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))
opt = ADAM()

Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))

# TESTING

accuracy(X, Y)

tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

accuracy(tX, tY)
