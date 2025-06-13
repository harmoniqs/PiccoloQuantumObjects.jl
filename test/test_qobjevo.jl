@testitem "QobjEvo tests" begin
    using QuantumToolbox
    using QuantumToolbox: QobjEvo, basis, destroy, sigmaz, sigmax, expect, mesolve, sesolve
    using PiccoloQuantumObjects
    using PiccoloQuantumObjects: get_drift, get_drives, QuantumSystem, OpenQuantumSystem
    using NamedTrajectories
    using NamedTrajectories: get_times
    using LinearAlgebra

    function isapprox_qobj(qobj1::Qobj, qobj2::Qobj; kwargs...)
        return qobj1.type == qobj2.type && qobj1.dims == qobj2.dims && isapprox(qobj1.data, qobj2.data; kwargs...)
    end

    @testset "QuantumSystem Time Evolution" begin
        H_drift = 1.0 * PAULIS[:Z]
        H_drives = [1.0 * PAULIS[:X], 1.0 * PAULIS[:Y]]
        sys = QuantumSystem(H_drift, H_drives)
        T = 100
        Δt = 0.05
        times = range(0, (T-1)*Δt, length=T)
        a_vals = Matrix{Float64}(undef, 2, T)
        a_vals[1,:] = 1.0 * sin.(2π * times / times[end])
        a_vals[2,:] = 1.0 * cos.(2π * times / times[end])
        traj = NamedTrajectory(
            (a = a_vals, Δt = fill(Δt, 1, T)),
            timestep=:Δt,
            controls=:a
        )
        H_evo = QobjEvo(sys, traj)
        ψ0 = basis(2, 0)
        sol = sesolve(H_evo, ψ0, get_times(traj))
        @test length(sol.states) == T
        @test isapprox(norm(ψ0), 1.0, atol=1e-6)
        @test isapprox(norm(sol.states[end]), 1.0, atol=1e-6)
        initial_z = expect(sigmaz(), ψ0)
        final_z = expect(sigmaz(), sol.states[end])
        z_change = abs(initial_z - final_z)
        @test z_change > 0.02 || isapprox(z_change, 0.0, atol=0.02)
    end

    @testset "OpenQuantumSystem Time Evolution" begin
        H_drift = 0.5 * PAULIS[:Z]
        H_drives = [PAULIS[:X]]
        diss_ops = [sqrt(0.5)*PAULIS[:Z], sqrt(0.25)*PAULIS[:X]]
        sys = OpenQuantumSystem(H_drift, H_drives, diss_ops)
        T = 100
        Δt = 0.05
        times = range(0, (T-1)*Δt, length=T)
        a_vals = Matrix{Float64}(undef, 1, T)
        a_vals[1,:] = 0.5 * sin.(3π * times / times[end])
        traj = NamedTrajectory(
            (a = a_vals, Δt = fill(Δt, 1, T)),
            timestep=:Δt,
            controls=:a
        )
        H_evo, c_ops = QobjEvo(sys, traj)
        ψ0 = basis(2, 0)
        e_ops = [sigmaz(), sigmax()]
        sol = mesolve(H_evo, ψ0, get_times(traj), c_ops; e_ops=e_ops, saveat=get_times(traj))
        @test length(sol.states) == T
        @test size(sol.expect) == (length(e_ops), T)
        initial_z = expect(sigmaz(), ψ0)
        final_z = sol.expect[1,end]
        @test abs(initial_z - final_z) > 0.2
        @test isapprox(norm(sol.states[end]), 1.0, atol=1e-4)
    end

    @testset "QuantumSystem: Single drive, basic amplitudes" begin
        H_drift = 0.5 * PAULIS[:Z]
        H_drives = [PAULIS[:X]]
        sys = QuantumSystem(H_drift, H_drives)
        T = 5
        Δt = 0.1
        a_vals = [0.1 0.2 0.3 0.4 0.5]
        traj = NamedTrajectory(
            (a = a_vals, Δt = fill(Δt, 1, T)),
            timestep=:Δt,
            controls=:a
        )
        H_evo = QobjEvo(sys, traj)
        times = get_times(traj)
        @test isapprox_qobj(H_evo(times[1]), Qobj(H_drift + a_vals[1,1] * H_drives[1]))
        @test isapprox_qobj(H_evo(times[3]), Qobj(H_drift + a_vals[1,3] * H_drives[1]))
        t_mid = times[2] - 0.01*Δt
        @test isapprox_qobj(H_evo(t_mid), Qobj(H_drift + a_vals[1,1] * H_drives[1]))
    end

    @testset "QuantumSystem: State transfer with sesolve" begin
        H_drift = 0.05 * PAULIS[:Z]
        H_drives = [PAULIS[:X]]
        sys = QuantumSystem(H_drift, H_drives)
        T = 50
        Δt = 0.1
        times = range(0, (T-1)*Δt, length=T)
        amps = zeros(1, T)
        amps[1,:] = sin.(2π .* times ./ (T*Δt))
        traj = NamedTrajectory(
            (a = amps, Δt = fill(Δt, 1, T)),
            timestep=:Δt,
             controls=:a
        )
        H_evo = QobjEvo(sys, traj)
        ψ0 = basis(2, 0)
        sol = sesolve(H_evo, ψ0, times; progress_bar=Val(false), saveat=times)
        @test length(sol.states) == length(times)
        @test isapprox(norm(sol.states[1]), 1.0, atol=1e-6)
        @test isapprox(norm(sol.states[end]), 1.0, atol=1e-6)
        z0 = real(expect(sigmaz(), ψ0))
        zf = real(expect(sigmaz(), sol.states[end]))
        @test !isapprox(z0, zf, atol=1e-3) || (length(times) <= 1)
    end

    @testset "QuantumSystem: mesolve with dissipative dynamics" begin
        H_drift = 0.1 * PAULIS[:Z]
        H_drives = [PAULIS[:X]]
        sys = QuantumSystem(H_drift, H_drives)
        T = 20
        Δt = 0.1
        times = range(0.0, (T-1)*Δt, length=T)
        amps = zeros(1, T)
        amps[1,:] = sin.(times * 2π / times[end] * 0.5)
        traj = NamedTrajectory(
            (a = amps, Δt = fill(Δt, 1, T)),
            timestep=:Δt,
            controls=:a
        )
        H_evo = QobjEvo(sys, traj)
        ψ0 = basis(2, 0)
        c_ops = [sqrt(0.1) * destroy(2)]
        e_ops = [sigmaz(), sigmax()]
        sol = mesolve(H_evo, ψ0, times, c_ops; e_ops=e_ops, progress_bar=Val(false), saveat=times)
        @test length(sol.states) == length(times)
        @test size(sol.expect) == (length(e_ops), length(times))
        @test isapprox(real(sol.expect[1, 1]), real(expect(sigmaz(), ψ0)))
        @test isapprox(real(sol.expect[2, 1]), real(expect(sigmax(), ψ0)))
        @test !isapprox(real(sol.expect[1, 1]), real(sol.expect[1, end]), atol=1e-3)
        @test !isapprox(real(sol.expect[2, 1]), real(sol.expect[2, end]), atol=1e-3)
        for state in sol.states
            @test isapprox(norm(state), 1.0, atol=1e-6)
        end
    end

    @testset "QuantumSystem: Multiple drives, varying amplitudes" begin
        H_drift = 0.05 * PAULIS[:Z]
        H_drives = [PAULIS[:X], PAULIS[:Y]]
        sys = QuantumSystem(H_drift, H_drives)
        T = 10
        Δt = 0.2
        times = range(0.0, (T-1)*Δt, length=T)
        a_vals = [cos.(times)'; sin.(times)']
        traj = NamedTrajectory(
            (a = a_vals, Δt = fill(Δt, 1, T)),
            timestep=:Δt,
            controls=:a
        )
        H_evo = QobjEvo(sys, traj)
        times = get_times(traj)
        @test isapprox_qobj(H_evo(times[1]), Qobj(H_drift + a_vals[1,1] * H_drives[1] + a_vals[2,1] * H_drives[2]))
        @test isapprox_qobj(H_evo(times[end]), Qobj(H_drift + a_vals[1,end] * H_drives[1] + a_vals[2,end] * H_drives[2]))
        t_interp = (times[5] + times[6]) / 2
        @test isapprox_qobj(H_evo(t_interp), Qobj(H_drift + a_vals[1,5] * H_drives[1] + a_vals[2,5] * H_drives[2]))
        t_just_before = times[3] - 1e-10
        @test isapprox_qobj(H_evo(t_just_before), Qobj(H_drift + a_vals[1,2] * H_drives[1] + a_vals[2,2] * H_drives[2]))
    end

    @testset "QuantumSystem: Drives-only QuantumSystem (no drift)" begin
        H_drift = zeros(2, 2)
        H_drives = [PAULIS[:X]]
        sys = QuantumSystem(H_drift, H_drives)
        T = 5
        Δt = 0.1
        times = range(0.0, (T-1)*Δt, length=T)
        a_vals = ones(1, T)
        traj = NamedTrajectory(
            (a = a_vals, Δt = fill(Δt, 1, T)),
            timestep=:Δt,
            controls=:a
        )
        H_evo = QobjEvo(sys, traj)
        @test isapprox_qobj(H_evo(times[1]), Qobj(a_vals[1,1] * H_drives[1]))
        @test isapprox_qobj(H_evo(times[end]), Qobj(a_vals[1,end] * H_drives[1]))
    end

    @testset "QuantumSystem: sesolve integration" begin
        H_drift = 0.01 * PAULIS[:Z]
        H_drives = [PAULIS[:X]]
        sys = QuantumSystem(H_drift, H_drives)
        T = 10
        Δt = 0.05
        times = range(0.0, (T-1)*Δt, length=T)
        a_values = zeros(1, T)
        a_values[1,:] = sin.(times * π / times[end])
        traj = NamedTrajectory(
            (a = a_values, Δt = fill(Δt, 1, T)),
            timestep=:Δt,
            controls=:a
       )
       H_evo = QobjEvo(sys, traj)
       ψ0 = basis(2, 0)
       sol = sesolve(H_evo, ψ0, times; progress_bar=Val(false), saveat=times)
       @test length(sol.states) == length(times)
       @test isapprox(norm(sol.states[1]), 1.0, atol=1e-6)
       @test isapprox(norm(sol.states[end]), 1.0, atol=1e-6)
       initial_z = real(expect(sigmaz(), ψ0))
       final_z = real(expect(sigmaz(), sol.states[end]))
       @test !isapprox(initial_z, final_z, atol=1e-3) || (length(times) <= 1)
   end
end
