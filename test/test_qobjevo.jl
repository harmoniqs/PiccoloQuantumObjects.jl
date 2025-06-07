@testitem "QobjEvo tests" begin
    using QuantumToolbox
    using QuantumToolbox: Qobj, length, size
    using PiccoloQuantumObjects
    using PiccoloQuantumObjects: get_drift, get_drives
    using NamedTrajectories
    using NamedTrajectories: get_times
    using Interpolations
    using LinearAlgebra

    function isapprox_qobj(qobj1::Qobj, qobj2::Qobj; kwargs...)
        return qobj1.type == qobj2.type && qobj1.dims == qobj2.dims && isapprox(qobj1.data, qobj2.data; kwargs...)
    end

    @testset "Single drive, simple amplitudes" begin
        H_drift_test1 = 0.5 * PAULIS.Z
        H_drives_test1 = [PAULIS.X]
        sys1 = QuantumSystem(H_drift_test1, H_drives_test1)
        T1 = 5
        Δt1 = 0.1
        a_values1 = [0.1 0.2 0.3 0.4 0.5]
        prob_test1 = UnitarySmoothPulseProblem(sys1, GATES.X, T1, Δt1; a_bound=1.0, dda_bound=1.0)
        solve!(prob_test1; max_iter=1)
        traj1 = prob_test1.trajectory
        a_idx_range1 = traj1.components.a
        traj1.data[a_idx_range1, :] = a_values1
        H_evo1 = QobjEvo(sys1, traj1)
        times1 = get_times(traj1)
        @test isapprox_qobj(H_evo1(times1[1]), Qobj(H_drift_test1 + a_values1[1] * H_drives_test1[1]))
        @test isapprox_qobj(H_evo1(times1[3]), Qobj(H_drift_test1 + a_values1[3] * H_drives_test1[1]))
        @test isapprox_qobj(H_evo1(times1[end]), Qobj(H_drift_test1 + a_values1[end] * H_drives_test1[1]))
        t_interp1 = (times1[1] + times1[2]) / 2
        expected_a_interp1 = (a_values1[1] + a_values1[2]) / 2
        @test isapprox_qobj(H_evo1(t_interp1), Qobj(H_drift_test1 + expected_a_interp1 * H_drives_test1[1]))
    end

    @testset "using mesolve" begin
        H_drift_ms = 0.1 * PAULIS.Z
        H_drives_ms = [PAULIS.X]
        sys_ms = QuantumSystem(H_drift_ms, H_drives_ms)
        T_ms = 20
        Δt_ms = 0.1
        times_ms_expected = collect(0.0:Δt_ms:(T_ms-1)*Δt_ms)
        a_values_ms = reshape(sin.(times_ms_expected * 2π / times_ms_expected[end] * 0.5), 1, :)
        prob_ms = UnitarySmoothPulseProblem(sys_ms, GATES.X, T_ms, Δt_ms; a_bound=1.0, dda_bound=1.0)
        solve!(prob_ms; max_iter=10)
        traj_ms = prob_ms.trajectory
        a_idx_range_ms = traj_ms.components.a
        traj_ms.data[a_idx_range_ms, :] = a_values_ms
        H_evo_ms = QobjEvo(sys_ms, traj_ms)
        times_ms = get_times(traj_ms)
        ψ0_ms = basis(2, 0)
        c_ops_ms = [sqrt(0.1) * destroy(2)]
        e_ops_ms = [sigmaz(), sigmax()]
        sol_ms = mesolve(H_evo_ms, ψ0_ms, times_ms, c_ops_ms; e_ops=e_ops_ms, progress_bar = Val(false), saveat = times_ms)

        @test length(sol_ms.states) == length(times_ms)
        @test size(sol_ms.expect) == (length(e_ops_ms), length(times_ms))
        @test isapprox(real(sol_ms.expect[1, 1]), real(expect(sigmaz(), ψ0_ms)))
        @test isapprox(real(sol_ms.expect[2, 1]), real(expect(sigmax(), ψ0_ms)))

        initial_expect_z = real(expect(sigmaz(), ψ0_ms))
        final_expect_z = real(sol_ms.expect[1, end])
        @test !isapprox(initial_expect_z, final_expect_z, atol=1e-3) || (length(times_ms) <= 1)

        initial_expect_x = real(expect(sigmax(), ψ0_ms))
        final_expect_x = real(sol_ms.expect[2, end])
        @test !isapprox(initial_expect_x, final_expect_x, atol=1e-3) || (length(times_ms) <= 1)
        for i in 1:length(sol_ms.states)
            @test isapprox(norm(sol_ms.states[i]), 1.0, atol=1e-6)
        end
    end
end
