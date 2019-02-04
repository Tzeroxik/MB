using Flux
using Flux: @epochs, onehotbatch, mse, throttle

struct AutoEncoder
    _encoder       # the hidden_layer 
    _decoder       # the  output_layer
    _model         
end 

function AutoEncoder(inout_dim::Int32, encoded_size::Int32)
    encoder = Dense(inout_dim, encoded_size, leakyrelu)
    decoder = Dense(encoded_size, inout_dim, leakyrelu)
    model   = Chain(encoder, decoder)
    AutoEncoder(encoder, decoder, model)
end

function train(autoencoder::AutoEncoder, data) 
    
    loss(x) = mse(autoencoder._model(x), x)
    evalcb = throttle(() -> @show(loss(data[1])), 5)
    optimizer = ADAM()
    
    @epochs 10 Flux.train!(loss, params(autoencoder._model), zip(data), optimizer, cb = evalcb)
end



