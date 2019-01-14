
struct Vec2D
    x::Float64
    y::Float64
end

function Soma2D(u::Vec2D , v::Vec2D)
    out = Vec2D(u.x + v.x , u.y +v.y)
    return out
end

function ProdByScalar2D(lambda::Float64,u::Vec2D)
    out = Vec2D(lambda * u.x , lambda * u.y)
    return out
end


struct Vec3D
    x::Float64
    y::Float64
    z::Float64
end

struct Plane3D
    point::Vec3D
    normal::Vec3D
end

function Vec3DToArray(u::Vec3D)
    out = [u.x,u.y,u.z]
    return out
end

# this is a very useful function to create views
# with it you can tune the vectors by successive random generation
function ArrayToVec3D(u::Array{Float64,1})
    out = Vec3D(u[1],u[2],u[3])
    return out
end

function Soma3D(u::Vec3D , v::Vec3D)
    out = Vec3D(u.x + v.x , u.y + v.y , u.z + v.z)
    return out
end

function ProdByScalar3D(lambda::Float64,u::Vec3D)
    out = Vec3D(lambda * u.x , lambda * u.y , lambda * u.z)
    return out
end

function VectorProd(u::Vec3D,v::Vec3D)
    out = Vec3D(u.y*v.z - u.z*v.y,u.z*v.x - u.x*v.z , u.x * v.y - u.y * v.x)
    return out
end

function DotProd3D(u::Vec3D , v::Vec3D)
    out = u.x * v.x + u.y * v.y + u.z * v.z
    return out
end

function Soma3D(u::Vec3D , v::Vec3D)
    out = Vec3D(u.x + v.x , u.y + v.y , u.z + v.z)
    return out
end

function ProdByScalar3D(lambda::Float64,u::Vec3D)
    out = Vec3D(lambda * u.x , lambda * u.y , lambda * u.z)
    return out
end

function Versor3D(u::Vec3D)

    scalar = 1.0/sqrt(DotProd3D(u,u))
    out = ProdByScalar3D(scalar,u)

    return out
end

# the method we prefer to initialize a plane is by a point and 2 vectors
function Plane3DInit(point::Vec3D , veca::Vec3D , vecb::Vec3D)

    normal = VectorProd(veca,vecb)
    scalar = 1.0/sqrt(DotProd3D(normal,normal))

    # we want that the normal vector has norm 1
    normal = ProdByScalar3D(scalar , normal)

    out = Plane3D(point , normal)

    return out
end


# we create a 3D point that is the orthogonal projection of a point in a Plane3D
function ProjOverPlane3D(point::Vec3D , plano::Plane3D)

    # we create the vector from a point in the plane to the point that we want to make the projection
    vectorpos = Soma3D(point , ProdByScalar3D(-1.0,plano.point))
    dist = DotProd3D(vectorpos , plano.normal)
    tmp = ProdByScalar3D(-dist,plano.normal)

    out = Soma3D(point , tmp)

    return out
end


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

######### The generalization of the previous instructions for lower dimensional cases #############

####################################################################################################
####################################################################################################
####################################################################################################


# the basic data type is a vector with N dimensions.
struct VecND
    dim::Int
    values::Array{Float64,1}
end

struct HyperPlaneND
    point::VecND
    normal::VecND
end

function VecNDInit(dim::Int)
    tmp = zeros(dim)
    out = VecND(dim,tmp)

    return out
end


# the usual way to initialize a point in N-dimensions is to carry it from an array
function ArrayToVecND(u::Array{Float64,1})
    out = VecND(size(u)[1],u)
    return out
end

function SomaND(u::VecND , v::VecND)
    tmp = u.values + v.values
    dim = size(u.values)[1]
    out = VecND(dim,tmp)
    return out
end

function ProdByScalarND(lambda::Float64,u::VecND)
    dim = u.dim
    tmp = lambda * u.values
    out = VecND(dim,tmp)
    return out
end

function DotND(u::VecND , v::VecND)
    out = dot(u.values,v.values)
    return out
end

function VersorND(u::VecND)
    invnorm = 1.0 / sqrt(dot(u.values,u.values))
    dim = u.dim
    tmp = invnorm * u.values
    out = VecND(dim,tmp)
    return out
end

# from an array of VecND's creates a matrix with the values
function VecNDToMatrix(uarra::Array{VecND,1})
    # check if all dimensions match
    for i in 1:length(uarra)
        if isequal(uarra[1].dim ,uarra[i].dim)
            continue
        end
    end

    tamanho = length(uarra) ;
    dimen = uarra[1].dim

    matriz = Array{Float64}(tamanho,dimen)

    for i in 1:tamanho
        for j in 1:dimen
            matriz[i,j] = uarra[i].values[j]
        end
    end

    return matriz
end

# from N-1 linearly independent VecND's computes a vector
# that is perpendicular to all the others and it preserves
# orientation (det > 0). Its norm it is equal to the norm
# hyper-paralelogram of the input VecND's.
# the input is an array of VecND's with the same dimension
function VectorProdND(uarr::Array{VecND,1})

    # check if all dimensions match
    for i in 1:length(uarr)
        if isequal(uarr[1].dim ,uarr[i].dim)
            continue
        end
    end

    dimension = uarr[1].dim

    # the easiest way to proceed is to put everything in a matrix
    matrix = VecNDToMatrix(uarr)

    # we will append a row at the end.
    soma = []

    for i in 1:dimension
        tmp = 0
        for j in 1:size(matrix)[1]
            tmp += matrix[j,i]
        end
        push!(soma,tmp)
    end

    #equivalent to argmax
    switmax = indmax(abs.(soma))
    switmin = indmin(abs.(soma))
    # we exchange the signal of the entry with the highest abs value
    # this way we guarantee that soma is linearly independent
    soma[switmax] *= -1.0

    transfer = soma[switmin]
    soma[switmin] = soma[switmax]
    soma[switmax] = transfer

    tmp = VecND(dimension,soma)

    # at this moment we will create a square matrix
    matrix = [matrix;reshape(tmp.values,(1,dimension))]

    # if A is the matrix whose rows contain the VecND's, we have appended X,
    # This X will be the orthogonal vector to all rows of A
    # AX = b where b = [0 \dots 0 1]. This way we impose the orthogonality and
    # that norm(X) = 1. In Julia X = A \ b
    matb = reshape(zeros(dimension),(dimension,1))
    matb[dimension,1] = 1.0

    tmpout_mat = matrix \ matb

    values = Array{Float64}(dimension)
    for i in 1:dimension
        values[i] = tmpout_mat[i,1]
    end

    out = ArrayToVecND(values)

    if (det(matrix) < 0)
        out = ProdByScalarND(-1.0 * abs(det(matrix)),out)
    else
        out = ProdByScalarND(det(matrix),out)
    end

    return out

end

# now things are quite easy to define
function HyperPlaneNDInit(point::VecND , uarr::Array{VecND,1})

    normal = VersorND(VectorProdND(uarr))
    out = HyperPlaneND(point , normal)
    return out

end

# the natural generalization of the 3-dimensional case
function ProjOverHyperPlaneND(point::VecND , hplane::HyperPlaneND)

    vectorpos = SomaND(point , ProdByScalarND(-1.0 , hplane.point))
    signdist = DotND(vectorpos,hplane.normal)
    tmp = ProdByScalarND(-signdist , hplane.normal)

    out = SomaND(point,tmp)

end
