@testitem "QobjEvo tests" begin
    using QuantumCollocation: UnitarySmoothPulseProblem, UnitaryInfidelityObjective, UnitaryIntegrator
    using DirectTrajOpt: DirectTrajOptProblem
    using QuantumToolbox
    using QuantumToolbox: QobjEvo, basis, destroy, sigmaz, sigmax, expect, mesolve, sesolve
    using PiccoloQuantumObjects
    using PiccoloQuantumObjects: get_drift, get_drives, QuantumSystem
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

    @testset "sesolve using DirectTrajOptProblem for state transfer" begin
        H_drift_dt = 0.05 * PAULIS.Z
        H_drives_dt = [PAULIS.X]
        sys_dt = QuantumSystem(H_drift_dt, H_drives_dt)
        dim = size(H_drift_dt, 1)
        num_drives = length(H_drives_dt)
        T_intervals = 50
        Δt_init = 0.1
        prob_baseline = UnitarySmoothPulseProblem(
            sys_dt, GATES.X, T_intervals, Δt_init;
            a_bound=1.0, dda_bound=1.0
        )
        solve!(prob_baseline; max_iter=1)
        traj_dt_base = prob_baseline.trajectory
        amplitude_values = reshape(
            sin.(collect(0.0:Δt_init:(T_intervals-1)*Δt_init) * 2π / (T_intervals*Δt_init)),
            num_drives, :
        )
        a_idx_range = traj_dt_base.components.a
        traj_dt_base.data[a_idx_range, :] = amplitude_values
        unitary_state_symbol = nothing
        for s in keys(traj_dt_base.components)
            if startswith(string(s), "Ũ")
                unitary_state_symbol = s
                break
            end
        end
        obj_dt = UnitaryInfidelityObjective(GATES.X, unitary_state_symbol, traj_dt_base)
        integrator_dt = UnitaryIntegrator(sys_dt, traj_dt_base, unitary_state_symbol, :a)
        prob_direct = DirectTrajOptProblem(
            traj_dt_base,
            obj_dt,
            integrator_dt
        )
        solve!(prob_direct; max_iter=1)
        optimized_traj_dt = prob_direct.trajectory
        H_evo_dt = QobjEvo(sys_dt, optimized_traj_dt)
        times_dt = get_times(optimized_traj_dt)
        ψ0_dt = basis(dim, 0)
        sol_dt = sesolve(H_evo_dt, ψ0_dt, times_dt; progress_bar=Val(false), saveat=times_dt)
        @test length(sol_dt.states) == length(times_dt)
        @test isapprox(norm(sol_dt.states[1]), 1.0, atol=1e-6)
        @test isapprox(norm(sol_dt.states[end]), 1.0, atol=1e-6)
        initial_expect_z_dt = real(expect(sigmaz(), ψ0_dt))
        final_expect_z_dt = real(expect(sigmaz(), sol_dt.states[end]))
        @test !isapprox(initial_expect_z_dt, final_expect_z_dt, atol=1e-3) || (length(times_dt) <= 1)
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

    @testset "Multiple drives, varying amplitudes" begin
        H_drift_test2 = 0.05 * PAULIS.Z
        H_drives_test2 = [PAULIS.X, PAULIS.Y]
        sys2 = QuantumSystem(H_drift_test2, H_drives_test2)
        T2 = 10
        Δt2 = 0.2
        a_values2 = vcat(
            reshape(cos.(collect(0.0:Δt2:(T2-1)*Δt2)), 1, :),
            reshape(sin.(collect(0.0:Δt2:(T2-1)*Δt2)), 1, :)
        )
        prob_test2 = UnitarySmoothPulseProblem(sys2, GATES.Z, T2, Δt2; a_bound=1.0, dda_bound=1.0)
        solve!(prob_test2; max_iter=1)
        traj2 = prob_test2.trajectory
        a_idx_range2 = traj2.components.a
        traj2.data[a_idx_range2, :] = a_values2
        H_evo2 = QobjEvo(sys2, traj2)
        times2 = get_times(traj2)
        @test isapprox_qobj(H_evo2(times2[1]), Qobj(H_drift_test2 + a_values2[1, 1] * H_drives_test2[1] + a_values2[2, 1] * H_drives_test2[2]))
        @test isapprox_qobj(H_evo2(times2[end]),Qobj(H_drift_test2 + a_values2[1, end] * H_drives_test2[1] + a_values2[2, end] * H_drives_test2[2]))
        t_interp2 = (times2[5] + times2[6]) / 2
        expected_a1_interp2 = (a_values2[1, 5] + a_values2[1, 6]) / 2
        expected_a2_interp2 = (a_values2[2, 5] + a_values2[2, 6]) / 2
        @test isapprox_qobj(H_evo2(t_interp2), Qobj(H_drift_test2 + expected_a1_interp2 * H_drives_test2[1] + expected_a2_interp2 * H_drives_test2[2]))
    end

    @testset "Drives-only QuantumSystem (no drift)" begin
        H_drift_dr = zeros(2, 2) # No drift
        H_drives_dr = [PAULIS.X]
        sys_dr = QuantumSystem(H_drift_dr, H_drives_dr)
        T_dr = 5
        Δt_dr = 0.1
        times_dr = collect(0.0:Δt_dr:(T_dr-1)*Δt_dr)
        a_values_dr = reshape(ones(length(times_dr)), 1, :)
        prob_dr = UnitarySmoothPulseProblem(sys_dr, GATES.X, T_dr, Δt_dr; a_bound=1.0, dda_bound=1.0)
        solve!(prob_dr; max_iter=1)
        traj_dr = prob_dr.trajectory
        a_idx_range_dr = traj_dr.components.a
        traj_dr.data[a_idx_range_dr, :] = a_values_dr
        H_evo_dr = QobjEvo(sys_dr, traj_dr)
        @test isapprox_qobj(H_evo_dr(times_dr[1]), Qobj(a_values_dr[1, 1] * H_drives_dr[1]))
        @test isapprox_qobj(H_evo_dr(times_dr[end]), Qobj(a_values_dr[1, end] * H_drives_dr[1]))
    end

    @testset "sesolve integration (closed system)" begin
        H_drift_se = 0.01 * PAULIS.Z
        H_drives_se = [PAULIS.X]
        sys_se = QuantumSystem(H_drift_se, H_drives_se)
        T_se = 10
        Δt_se = 0.05
        times_se_expected = collect(0.0:Δt_se:(T_se-1)*Δt_se)
        a_values_se = reshape(sin.(times_se_expected * π / times_se_expected[end]), 1, :)
        prob_se = UnitarySmoothPulseProblem(sys_se, GATES.X, T_se, Δt_se; a_bound=1.0, dda_bound=1.0)
        solve!(prob_se; max_iter=10)
        traj_se = prob_se.trajectory
        a_idx_range_se = traj_se.components.a
        traj_se.data[a_idx_range_se, :] = a_values_se
        H_evo_se = QobjEvo(sys_se, traj_se)
        times_se = get_times(traj_se)
        ψ0_se = basis(2, 0)
        sol_se = sesolve(H_evo_se, ψ0_se, times_se; progress_bar = Val(false), saveat = times_se)
        @test length(sol_se.states) == length(times_se)
        @test isapprox(norm(sol_se.states[1]), 1.0, atol=1e-6)
        @test isapprox(norm(sol_se.states[end]), 1.0, atol=1e-6)
        @test !isapprox(real(expect(QuantumToolbox.sigmaz(), ψ0_se)), real(expect(QuantumToolbox.sigmaz(), sol_se.states[end])), atol=1e-3) || (length(times_se) <= 1)
    end
end
