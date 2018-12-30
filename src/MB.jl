module MB
    export correlation_matrix, covariance_matrix, pca
    using LinearAlgebra
    using Statistics
    
    correlation_matrix(mat::Array{Float64 ,2})::Array{Float64 ,2} = (mat .- mean(mat,dims=1)) ./ std(mat,dims=1)   
    covariance_matrix(corrmat::Array{Float64 ,2})::Array{Float64 ,2} = corrmat |> length |> float |> n -> corrmat' * corrmat / (n  - 1.)
    pca(mat::Array{Float64 ,2})::SVD = mat |> correlation_matrix |> svd
    
end 
