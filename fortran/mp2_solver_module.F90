module mp2_solver_module
    use hartree_fock_module
    implicit none

contains

    !===========================================================================
    ! MP2 Solver - 2nd order perturbation theory (non-iterative)
    ! Simplest test of the normal-ordered Hamiltonian
    !===========================================================================
    subroutine mp2_solver(no_ham, e_corr)
        type(normal_ordered_hamiltonian), intent(in) :: no_ham
        real(8), intent(out) :: e_corr
        
        integer :: nocc, nvir, n_states
        integer :: a, b, i, j
        real(8) :: denom, v_abij, t2_elem
        real(8), allocatable :: eps_o(:), eps_v(:)
        real(8), allocatable :: t2(:,:,:,:)
        integer :: count_nonzero
        real(8) :: max_t2
        
        write(*,*)
        write(*,*) '========================================='
        write(*,*) '    MP2 Solver (Direct Formula)'
        write(*,*) '========================================='
        
        nocc = no_ham%nocc
        nvir = no_ham%nvir
        n_states = no_ham%n_states
        
        write(*,'(A,I5)') ' Number of occupied states:', nocc
        write(*,'(A,I5)') ' Number of virtual states:', nvir
        write(*,*)
        
        ! Extract orbital energies
        allocate(eps_o(nocc), eps_v(nvir))
        allocate(t2(nvir, nvir, nocc, nocc))
        
        do i = 1, nocc
            eps_o(i) = no_ham%f_hf(i, i)
        end do
        do a = 1, nvir
            eps_v(a) = no_ham%f_hf(nocc + a, nocc + a)
        end do
        
        write(*,'(A)') ' Fock orbital energies (occupied):'
        do i = 1, min(nocc, 8)
            write(*,'(A,I2,A,F12.6)') '   eps_o(', i, ') = ', eps_o(i)
        end do
        write(*,'(A)') ' Fock orbital energies (virtual):'
        do a = 1, nvir
            write(*,'(A,I2,A,F12.6)') '   eps_v(', a, ') = ', eps_v(a)
        end do
        write(*,*)
        
        ! Compute MP2 amplitudes and correlation energy
        e_corr = 0.0d0
        t2 = 0.0d0
        count_nonzero = 0
        max_t2 = 0.0d0
        
        write(*,'(A)') ' Computing MP2 amplitudes...'
        
        do a = 1, nvir
            do b = 1, nvir
                do i = 1, nocc
                    do j = 1, nocc
                        ! Energy denominator
                        denom = eps_o(i) + eps_o(j) - eps_v(a) - eps_v(b)
                        
                        if (abs(denom) > 1.0d-10) then
                            ! Antisymmetrized coupling from normal-ordered Hamiltonian
                            ! Gamma_hf is indexed in FULL HF basis (1:n_states, 1:n_states)
                            ! Virtual indices must have nocc offset: (nocc+a, nocc+b, i, j)
                            v_abij = no_ham%Gamma_hf(nocc + a, nocc + b, i, j)
                            t2_elem = v_abij / denom
                            t2(a, b, i, j) = t2_elem
                            
                            ! Accumulate energy: E_corr = sum_abij Gamma_abij * t2_abij
                            e_corr = e_corr + v_abij * t2_elem
                            
                            if (abs(t2_elem) > 1.0d-8) then
                                count_nonzero = count_nonzero + 1
                                if (abs(t2_elem) > max_t2) max_t2 = abs(t2_elem)
                            end if
                        end if
                    end do
                end do
            end do
        end do
        
        write(*,'(A,I6)') ' Non-zero T2 amplitudes: ', count_nonzero
        write(*,'(A,ES12.4)') ' Maximum |T2|: ', max_t2
        write(*,*)
        
        ! Print some sample values for comparison with Python
        write(*,'(A)') ' Sample T2 values (compare with Python):'
        do i = 1, min(nocc, 2)
            do j = 1, min(nocc, 2)
                do a = 1, min(nvir, 2)
                    do b = 1, min(nvir, 2)
                        if (abs(t2(a, b, i, j)) > 1.0d-8) then
                            write(*,'(A,I1,A,I1,A,I1,A,I1,A,ES12.5)') &
                                '   T2(', a, ',', b, ',', i, ',', j, ') = ', t2(a, b, i, j)
                        end if
                    end do
                end do
            end do
        end do
        write(*,*)
        
        write(*,'(A,F20.10,A)') ' MP2 Correlation Energy: ', e_corr, ' MeV'
        write(*,'(A,F20.10,A)') ' Total Energy (E0 + E_corr): ', no_ham%E0 + e_corr, ' MeV'
        write(*,*)
        
        deallocate(eps_o, eps_v, t2)
        
    end subroutine mp2_solver

end module mp2_solver_module
