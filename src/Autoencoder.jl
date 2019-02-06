using Flux
using Flux: @epochs, onehotbatch, mse, throttle

struct AutoEncoder
    _encoder       # the hidden_layer 
    _decoder       # the  output_layer
    _model         
end 

function AutoEncoder(inout_dim::Int64, encoded_size::Int64)
    encoder = Dense(inout_dim, encoded_size, leakyrelu)
    decoder = Dense(encoded_size, inout_dim, leakyrelu)
    model   = Chain(encoder, decoder)
    AutoEncoder(encoder, decoder, model)
end

function train(autoencoder::AutoEncoder, data,nepochs = 10::Int64) 
    
    loss(x) = mse(autoencoder._model(x), x)
    evalcb = throttle(() -> @show(loss(data[1])), 5)
    opt = ADAM(params(autoencoder._model))
    
    @epochs nepochs Flux.train!(loss, zip(data), opt, cb = evalcb)
end



