module MB
    export correlation_matrix, covariance_matrix, pca, principal_components, reduce_dimensionality, count_principal_components
    using LinearAlgebra
    using Statistics
    
    correlation_matrix(mat::Array{Float64 ,2})::Array{Float64 ,2} = (mat .- mean(mat,dims=1)) ./ std(mat,dims=1)   
    
    covariance_matrix(corrmat::Array{Float64 ,2})::Array{Float64 ,2} = corrmat |> length |> float |> n -> corrmat' * corrmat / (n  - 1.)

    pca(mat::Array{Float64 ,2})::SVD = mat |> correlation_matrix |> svd

    #Counts how many columns of the unitary matrix have a norm > 1
    #the caller must provide a pca object returned from the pca function
    function count_principal_components(dataset_pca::SVD)::UInt32
        U,_,_ = dataset_pca
        for i in axes(U,2) 
            if (norm(U[:,i]) <= 1.) return i end 
        end
    end

    #Calculates a principal component matrix using the PCA
    #provided by the pca function, k must be positive integer
    #if one wants to reduce the dimensionality of the original data
    #the correct function is reduce_dimensionality
    #*this is used as an internal or if one wants to observe Principal components* 
    function principal_components(dataset_pca, k)         
        dataset_pca.U[:, 1:k] * Diagonal(dataset_pca.S)[1:k, 1:k]
    end

    #uses PCA from the pca function to calculate the new 
    #reduced dimensionality matrix if k = 0 calculates
    #the number of components considered on the PCA
    #by the columns whose norm > 1
    function reduce_dimensionality(dataset_pca, k = 0) 
        if k == 0 
            k = count_principal_components(dataset_pca) 
        end 
        principal_components(dataset_pca, k) * dataset_pca.Vt[1:k, 1:k]
    end
    
end