module DirectSums

export direct_sum

using SparseArrays
using TestItems

using ..QuantumSystems
using ..Isomorphisms


"""
    direct_sum(A::AbstractMatrix, B::AbstractMatrix)

Returns the direct sum of two matrices.
"""
function direct_sum(A::AbstractMatrix, B::AbstractMatrix)
    return [A spzeros((size(A, 1), size(B, 2))); spzeros((size(B, 1), size(A, 2))) B]
end

"""
    direct_sum(A::SparseMatrixCSC, B::SparseMatrixCSC)

Returns the direct sum of two sparse matrices.
"""
function direct_sum(A::SparseMatrixCSC, B::SparseMatrixCSC)
    return blockdiag(A, B)
end

"""
    direct_sum(Ã⃗::AbstractVector, B̃⃗::AbstractVector)

Returns the direct sum of two iso_vec operators.
"""
function direct_sum(Ã⃗::AbstractVector, B̃⃗::AbstractVector)
    return operator_to_iso_vec(
        direct_sum(
            iso_vec_to_operator(Ã⃗),
            iso_vec_to_operator(B̃⃗)
        )
    )
end

"""
    direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)

Returns the direct sum of two `QuantumSystem` objects.
"""
function direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)
    @assert sys1.n_drives == sys2.n_drives "System 1 drives ($(sys1.n_drives)) must equal System 2 drives ($(sys2.n_drives))"
    n_drives = sys1.n_drives
    H = a -> direct_sum(sys1.H(a), sys2.H(a))
    direct_sum_params = Dict{Symbol, Dict{Symbol, Any}}()
    if haskey(sys1.params, :system_1)
        n_systems = length(keys(sys1.params))
        direct_sum_params = sys1.params
        if haskey(sys2.params, :system_1)
            for i = 1:length(keys(sys2.params))
                direct_sum_params[Symbol("system_$(n_systems + i)")] =
                    sys2.params[Symbol("system_$(i)")]
            end
        else
            direct_sum_params[Symbol("system_$(n_systems + 1)")] = sys2.params
        end
    else
        direct_sum_params[:system_1] = sys1.params
        if haskey(sys2.params, :system_1)
            n_systems = length(keys(sys2.params))
            for i = 1:length(keys(sys2.params))
                direct_sum_params[Symbol("system_$(1 + i)")] =
                    sys2.params[Symbol("system_$(i)")]
            end
        else
            direct_sum_params[:system_2] = sys2.params
        end
    end
    return QuantumSystem(H, n_drives; params=direct_sum_params)
end

direct_sum(systems::AbstractVector{<:QuantumSystem}) = reduce(direct_sum, systems)


# *************************************************************************** #

@testitem "Test vector direct sum" begin
    A = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    B = [0, 1, 0, 1, 1, 0, -1, 0]
    @test direct_sum(A, B) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, -1, 0]
end

@testitem "Test matrix direct sum" begin
    using SparseArrays
    A = [1 2; 3 4]
    B = [5 6; 7 8]
    @test direct_sum(A, B) == [1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]

    A = sparse([1 2; 3 4])
    B = sparse([5 6; 7 8])
    @test direct_sum(A, B) == sparse([1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8])
end

@testitem "Test vector of quantum system direct sum" begin
    sys1 = QuantumSystem([1 2; 3 4])
    sys2 = QuantumSystem([5 6; 7 8])
    sys = direct_sum([sys1, sys2])
    @test sys.H(0) == [1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]
    @test sys.n_drives == 0
    # created new keys for the direct sum
    @test sys.params[:system_1] == Dict()
    @test sys.params[:system_2] == Dict()
end

@testitem "Test quantum system direct sum" begin
    sys1 = QuantumSystem([1 2; 3 4])
    sys2 = QuantumSystem([5 6; 7 8])
    sys = direct_sum(sys1, sys2)
    @test sys.H(0) == [1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]
    @test sys.n_drives == 0
    # created new keys for the direct sum
    @test sys.params[:system_1] == Dict()
    @test sys.params[:system_2] == Dict()

    sys1 = QuantumSystem([1 2; 3 4]; params=Dict(:a => 1))
    sys2 = QuantumSystem([5 6; 7 8]; params=Dict(:b => 2))
    sys = direct_sum(sys1, sys2)
    @test sys.H(0) == [1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]
    @test sys.n_drives == 0
    @test sys.params == Dict{Symbol, Dict{Symbol, Any}}(:system_1 => Dict(:a => 1), :system_2 => Dict(:b => 2))
end

@testitem "Test quantum system with params" begin
    sys1 = QuantumSystem([1 2; 3 4], params=Dict(:system_1 => Dict(:a => 1)))
    sys2 = QuantumSystem([5 6; 7 8])
    sys = direct_sum(sys1, sys2)
    @test sys.H(0) == [1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]
    @test sys.n_drives == 0
    @test sys.params[:system_1] == Dict(:a=>1)
    @test sys.params[:system_2] == Dict()

    sys1 = QuantumSystem([1 2; 3 4])
    sys2 = QuantumSystem([5 6; 7 8], params=Dict(:system_1 => Dict(:a => 1)))
    sys = direct_sum(sys1, sys2)
    @test sys.H(0) == [1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]
    @test sys.n_drives == 0
    @test sys.params[:system_1] == Dict()
    @test sys.params[:system_2] == Dict(:a=>1)
end

end
