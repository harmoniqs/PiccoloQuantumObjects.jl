using NamedTrajectories


"""
    dense(vals, structure, shape)

Convert sparse data to dense matrix.

# Arguments
- `vals`: vector of values
- `structure`: vector of tuples of indices
- `shape`: tuple of matrix dimensions
"""
function dense(vals, structure, shape)

    M = zeros(shape)

    for (v, (k, j)) in zip(vals, structure)
        M[k, j] += v
    end

    if shape[1] == shape[2]
        return Symmetric(M)
    else
        return M
    end
end

"""
    show_diffs(A::Matrix, B::Matrix)

Show differences between matrices.
"""
function show_diffs(A::AbstractMatrix, B::AbstractMatrix; atol=0)
    @assert size(A) == size(B)
    matrix_is_square = size(A, 1) == size(A, 2)
    for (i, (a, b)) in enumerate(zip(A, B))
        inds = Tuple(CartesianIndices(A)[i])
        if matrix_is_square
            if !isapprox(a, b; atol=atol) && inds[1] ≤ inds[2]
                println((a, b), " @ ", inds)
            end
        else
            if !isapprox(a, b; atol=atol)
                println((a, b), " @ ", inds)
            end
        end
    end
end


function named_trajectory_type_1(; free_time=false)
    # Hadamard gate, two dda controls (random), Δt = 0.2
    data = [
        1.0          0.957107     0.853553     0.75         0.707107;
        0.0          0.103553     0.353553     0.603553     0.707107;
        0.0          0.103553     0.146447     0.103553     1.38778e-17;
        0.0         -0.25        -0.353553    -0.25        -1.52656e-16;
        0.0          0.103553     0.353553     0.603553     0.707107;
        1.0          0.75         0.146447    -0.457107    -0.707107;
        0.0         -0.25        -0.353553    -0.25        -1.249e-16;
        0.0          0.603553     0.853553     0.603553     4.16334e-16;
        0.0         -0.243953     0.959151    -0.665253     0.0;
        0.0          0.0139165    0.668917     0.625329     0.0;
        0.00393491   0.0240775   -0.00942396   0.00329391   0.00941354;
       -0.00223794  -0.0105816    0.00328457   0.0204239    0.0253415;
        0.0058186    0.00686586  -0.00422555   0.00442631   0.000319156;
       -0.00134597  -0.00120682   0.0114915    0.00189333  -0.0251649;
        0.2          0.2          0.2          0.2          0.2
    ]

    if free_time
        components = (
            Ũ⃗ = data[1:8, :],
            a = data[9:10, :],
            da = data[11:12, :],
            dda = data[13:14, :],
            Δt = data[15:15, :]
        )
        controls = (:dda, :Δt)
        timestep = :Δt
        bounds = (
            a = ([-1.0, -1.0], [1.0, 1.0]),
            dda = ([-1.0, -1.0], [1.0, 1.0]),
            Δt = ([0.1], [0.30000000000000004])
        )
    else
        components = (
            Ũ⃗ = data[1:8, :],
            a = data[9:10, :],
            da = data[11:12, :],
            dda = data[13:14, :]
        )
        controls = (:dda,)
        timestep = 0.2
        bounds = (
            a = ([-1.0, -1.0], [1.0, 1.0]),
            dda = ([-1.0, -1.0], [1.0, 1.0]),
        )
    end

    initial = (
        Ũ⃗ = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        a = [0.0, 0.0]
    )
    final = (a = [0.0, 0.0],)
    goal = (Ũ⃗ = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],)

    return NamedTrajectory(
        components;
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
end

function named_trajectory_type_2()
    # X gate (initial states: |0⟩, |1⟩), two dda controls (solved), Δt = 0.2
    data = [
        0.0     0.0    -0.0     0.057   0.205   0.405   0.58    0.677   0.705   0.707
        1.0     1.0     1.0     0.997   0.957   0.82    0.573   0.29    0.08    0.0
        0.0     0.0    -0.0    -0.057  -0.205  -0.405  -0.58   -0.677  -0.705  -0.707
        0.0     0.0     0.0     0.0    -0.0    -0.0    -0.0    -0.0    -0.0    -0.0
        1.0     1.0     1.0     0.997   0.957   0.82    0.573   0.29    0.08    0.0
        0.0     0.0    -0.0    -0.057  -0.205  -0.405  -0.58   -0.677  -0.705  -0.707
        0.0     0.0     0.0    -0.0     0.0     0.0     0.0     0.0     0.0     0.0
        0.0     0.0    -0.0    -0.057  -0.205  -0.405  -0.58   -0.677  -0.705  -0.707
        0.0    -0.0     0.145   0.387   0.57    0.634   0.57    0.387   0.145   0.0
        0.0     0.0    -0.145  -0.387  -0.57   -0.634  -0.57   -0.387  -0.145   0.0
        0.0     0.369   0.618   0.467   0.163  -0.163  -0.467  -0.618  -0.369   0.0
        0.0    -0.369  -0.618  -0.467  -0.163   0.163   0.467   0.618   0.369   0.0
        0.943   0.636  -0.386  -0.777  -0.833  -0.777  -0.386   0.636   0.943   0.0
       -0.943  -0.636   0.386   0.777   0.833   0.777   0.386  -0.636  -0.943   0.0
        0.392   0.392   0.392   0.392   0.392   0.392   0.392   0.392   0.392   0.392
    ]

    components = (
        ψ̃1 = data[1:4, :],
        ψ̃2 = data[5:8, :],
        a = data[9:10, :],
        da = data[11:12, :],
        dda = data[13:14, :],
        Δt = data[15:15, :]
    )

    controls = (:dda, :Δt)

    timestep = :Δt

    bounds = (
        a = ([-1.0, -1.0], [1.0, 1.0]), 
        da = ([-Inf, -Inf], [Inf, Inf]), 
        dda = ([-1.0, -1.0], [1.0, 1.0]), 
        Δt = ([0.0002], [0.4])
    )

    initial = (
        ψ̃1 = [0.0, 1.0, 0.0, 0.0], 
        ψ̃2 = [1.0, 0.0, 0.0, 0.0], 
        a = [0.0, 0.0], 
        da = [0.0, 0.0]
    )

    final = (
        a = [0.0, 0.0],
        da = [0.0, 0.0]
    )

    goal = (
        ψ̃1 = [1.0, 0.0, 0.0, 0.0], 
        ψ̃2 = [0.0, 1.0, 0.0, 0.0]
    )

    return NamedTrajectory(
        components;
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
end
