module MB

function covariance_matrix(mat::Array{Float64 ,2})::Float64 
    out::Array{Float64 ,2} = mat |> size |> zeros
    
    local col::Array{Float64 ,2}

    for i in 1:length(mat[:,1])
        col = mat[:,i]
        s = sum(col)        
        out[:,i] = map(x -> x - s, col)
    end 
    out
end

function svd(mat::Array{Float64,2})::Float64 
    mat |> covariance_matrix 
        |> covmat -> (mat - covmat)' * (mat - covmat) 
        |> eig
end 
 
end 