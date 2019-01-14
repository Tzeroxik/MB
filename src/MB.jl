module MB
    export correlation_matrix, covariance_matrix, pca, principal_components, reduce_dimensionality, count_principal_components
    using LinearAlgebra
    using Statistics

    correlation_matrix(mat::Array{Float64 ,2})::Array{Float64 ,2} = (mat .- mean(mat,dims=1)) ./ std(mat,dims=1)   
    
    covariance_matrix(corrmat::Array{Float64 ,2})::Array{Float64 ,2} = corrmat |> length |> float |> n -> corrmat' * corrmat / (n  - 1.)

    pca(mat::Array{Float64 ,2})::SVD = mat |> correlation_matrix |> svd

    function count_principal_components(dataset_pca::SVD)::UInt32
        U,_,_ = dataset_pca
        for i in axes(U,2) 
            if (norm(U[:,i]) <= 1.) return i end 
        end
    end

    function principal_components(dataset_pca::SVD; k) 
        U,S,_ = dataset_pca
        U[:, 1:k] * S[1:k, 1:k]
    end

    function reduce_dimensionality(dataset_pca::SVD; k::UInt32 = 0) 
        if k == 0 
            k = count_principal_components(dataset_pca) 
        end 
        principal_components(dataset_pca, k) * dataset_pca.Vt[1:k, 1:k]
    end
    
end