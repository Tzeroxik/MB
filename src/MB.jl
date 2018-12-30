module MB
    export center_matrix, cov_center_matrix, pca
    using LinearAlgebra
    using Statistics

    center_matrix(mat::Array{Float64 ,2})::Array{Float64 ,2} = (matrix .- mean(mat,dims=1)) ./ std(mat,dims=1)   
    cov_center_matrix(centermat::Array{Float64 ,2})::Array{Float64 ,2} = centermat |> length |> float |> n -> centermat' * centermat / (n  - 1.)
    pca(mat::Array{Float64 ,2})::SVD = mat |> center_matrix |> svd
    
end 
