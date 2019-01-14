module MB
    export correlation_matrix, covariance_matrix, pca, select_principal_components
    using LinearAlgebra
    using Statistics

    correlation_matrix(mat::Array{Float64 ,2})::Array{Float64 ,2} = (mat .- mean(mat,dims=1)) ./ std(mat,dims=1)   
    
    covariance_matrix(corrmat::Array{Float64 ,2})::Array{Float64 ,2} = corrmat |> length |> float |> n -> corrmat' * corrmat / (n  - 1.)

    pca(mat::Array{Float64 ,2})::SVD = mat |> correlation_matrix |> svd

    function select_principal_components((U,S,_)::SVD; ncomponents::Float64 = 0.)::Array{Float64 ,2}
        
        if ncomponents != 0.
            for i in axes(U,2)
                if norm(U[:,i]) <= 1.
                    ncomponents = i
                    break
                end
            end
        end
        K = 1:ncomponents

        U[:,K] * S[K,K]
    end



end 