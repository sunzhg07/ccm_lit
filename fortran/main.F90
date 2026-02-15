program ccm_main
    use talsh
    use snt_io_module
    use m_scheme_module
    use hartree_fock_module
    use mp2_solver_module
    use ccd_solver_module
    use ccsd_solver_module
    use ccsdt_solver_module
    use ccsdtq_solver_module
    use amplitude_storage_module
    implicit none
    
    type(single_particle_orbits) :: j_orbits, m_orbits
    type(potential) :: j_potential
    type(normal_ordered_hamiltonian) :: no_ham
    
    character(len=256) :: snt_file
    real(8) :: scale_factor
    real(8), allocatable :: m_v1b(:,:)
    real(8), allocatable :: m_v2b(:,:,:,:)  ! 4D m-scheme interaction
    
    integer :: n_p_val, n_n_val, n_occ
    integer, allocatable :: occ_indices(:)
    real(8) :: hf_energy
    real(8), allocatable :: sp_energies(:), rho(:,:), sp_coeffs(:,:)
    
    integer :: ierr, i, n_states, a, b, c, d
    real*8:: tmp
    logical:: restrict_sort
    
    ! Initialize TALSH
    write(*,*) '========================================='
    write(*,*) '   Nuclear Structure CCM with TALSH'
    write(*,*) '========================================='
    write(*,*)
    
    ierr = talsh_init()
    if (ierr /= 0) then
        write(*,*) 'Error initializing TALSH'
        stop
    end if
    write(*,*) 'TALSH initialized successfully'
    write(*,*)
    
    ! Setup parameters
    snt_file = 'gxpf1a_copy.snt'
    scale_factor = (42.0d0/56.0d0)**0.30d0
    snt_file = 'p.snt'
    scale_factor=1.
    
    ! Read SNT file
    write(*,*) '--- 1. Reading SNT Interaction ---'
    call read_snt(snt_file, scale_factor, j_orbits, j_potential)
    write(*,'(A,I4,A)') '   Read ', j_orbits%total_orbits, ' j-scheme orbits'
    write(*,*)
    
    ! Generate M-scheme basis
    write(*,*) '--- 2. Generating M-Scheme Basis ---'
    call generate_m_scheme(j_orbits, m_orbits)
    n_states = m_orbits%total_orbits
    write(*,'(A,I4,A)') '   Total m-scheme states: ', n_states
    write(*,*)
    
    ! Print m-scheme basis
    write(*,*) '--- M-Scheme Basis ---'
    write(*,'(A4,A4,A4,A5,A5,A5)') 'Idx', 'n', 'l', '2j', '2jz', '2tz'
    do i = 1,  n_states  ! Print first 20
        write(*,'(I4,I4,I4,I5,I5,I5)') i, m_orbits%n(i), m_orbits%l(i), &
                                        m_orbits%j(i), m_orbits%jz(i), m_orbits%tz(i)
    end do
    
    ! Decouple interaction to M-scheme
    write(*,*) '--- 3. Decoupling Interaction to M-Scheme ---'
    call decouple_1b(j_potential, m_orbits, m_v1b)
    call decouple_2b(j_potential, m_orbits, m_v2b)
    
!    ! Debug: print some matrix elements
    write(*,*) 'Sample m_v1b diagonal elements:'
    do i = 1, n_states
        write(*,'(A,I3,A,I3,A,F12.6)') '   m_v1b(', i, ',', i, ') = ', m_v1b(i,i)
    end do


   ! n_p_val = 8
   ! n_n_val = 8
   ! n_occ = n_p_val + n_n_val
   ! 
   ! allocate(occ_indices(n_occ))
   ! ! Proton indices (first 8 m-scheme states)
   ! do i = 1, n_p_val
   !     occ_indices(i) = i
   ! end do
   ! ! Neutron indices (8 states starting from index 21, which is Python index 20)
   ! do i = 1, n_n_val
   !     occ_indices(n_p_val + i) = 20 + i
   ! end do

    n_p_val = 4
    n_n_val = 4
    n_occ = n_p_val + n_n_val
    
    allocate(occ_indices(n_occ))
    ! Proton indices (first 8 m-scheme states)
    do i = 1, n_p_val
        occ_indices(i) = i
    end do
    ! Neutron indices (8 states starting from index 21, which is Python index 20)
    do i = 1, n_n_val
        occ_indices(n_p_val + i) = 6 + i
    end do
    
    ! Print occupation indices for verification
    write(*,*) '   Occupation indices (Fortran 1-based):'
    write(*,'(A)',advance='no') '   '
    tmp = 0.0d0
    do i = 1, n_occ
        write(*,'(I3)',advance='no') occ_indices(i)
        if (mod(i, 10) == 0) then
            write(*,*)
            write(*,'(A)',advance='no') '   '
        end if
        tmp = tmp + m_v1b(occ_indices(i),occ_indices(i))
    end do
    write(*,*)tmp
    write(*,*)
    
    ! Perform Hartree-Fock
    write(*,'(A,I2,A,I2,A)') '--- 4. Performing Hartree-Fock (Z=', n_p_val, ', N=', n_n_val, ') ---'
    restrict_sort = .false.
    call hartree_fock_talsh(m_orbits, m_v1b, m_v2b, n_p_val, n_n_val, &
                            occ_indices, n_occ, hf_energy, sp_energies, rho, sp_coeffs, &
                            mode_in='deformed', allow_sector_reoccupy_in=restrict_sort)
    write(*,*)
    
    ! Normal ordering
    write(*,*) '--- 5. Performing Normal Ordering ---'
    call normal_order_talsh(m_orbits, m_v1b, m_v2b, hf_energy, &
                            sp_energies, rho, sp_coeffs, n_occ, no_ham)
    write(*,*)
    
    ! At this point, no_ham is ready for CCD/CCSD calculations
    write(*,*) '========================================='
    write(*,*) '   Normal-Ordered Hamiltonian Ready!'
    write(*,*) '========================================='
    write(*,'(A,F16.8,A)') ' Reference Energy (E0): ', no_ham%E0, ' MeV'
    write(*,'(A,I4)') ' Number of hole states:  ', no_ham%nocc
    write(*,'(A,I4)') ' Number of particle states: ', no_ham%nvir
    write(*,*)
    
    ! Run MP2 solver (test the normal-ordered Hamiltonian)
    block
        real(8) :: e_corr_mp2
        
        call mp2_solver(no_ham, e_corr_mp2)
        
        write(*,*) '========================================='
        write(*,*) '   MP2 Calculation Complete'
        write(*,*) '========================================='
        write(*,'(A,F16.8,A)') ' Reference energy (E0):    ', no_ham%E0, ' MeV'
        write(*,'(A,F16.8,A)') ' MP2 correlation energy:   ', e_corr_mp2, ' MeV'
        write(*,'(A,F16.8,A)') ' Total MP2 energy:         ', no_ham%E0 + e_corr_mp2, ' MeV'
        write(*,*) 'Expected (Python reference): -94.2213370652 MeV'
        write(*,*)
    end block
    write(*,*)
    
    ! Run CCD solver
    block
        real(8) :: e_corr_ccd
        logical :: ccd_converged
        integer :: max_iter_ccd
        real(8) :: tol_ccd
        
        max_iter_ccd = 50
        tol_ccd = 1.0d-6
        
        call ccd_solver(no_ham, max_iter_ccd, tol_ccd, e_corr_ccd, ccd_converged)
        
        if (ccd_converged) then
            write(*,*) '========================================='
            write(*,*) '   CCD Calculation Successful!'
            write(*,*) '========================================='
            write(*,'(A,F16.8,A)') ' Reference energy (E0):    ', no_ham%E0, ' MeV'
            write(*,'(A,F16.8,A)') ' Correlation energy:       ', e_corr_ccd, ' MeV'
            write(*,'(A,F16.8,A)') ' Total CCD energy:         ', no_ham%E0 + e_corr_ccd, ' MeV'
        else
            write(*,*) 'CCD did not converge'
        end if
    end block
    write(*,*)
    
    ! Run CCSD solver
    block
        real(8) :: e_corr_ccsd
        logical :: ccsd_converged
        integer :: max_iter_ccsd
        real(8) :: tol_ccsd
        
        max_iter_ccsd = 50
        tol_ccsd = 1.0d-6
        
        call ccsd_solver(no_ham, max_iter_ccsd, tol_ccsd, e_corr_ccsd, ccsd_converged)
        
        if (ccsd_converged) then
            write(*,*) '========================================='
            write(*,*) '   CCSD Calculation Successful!'
            write(*,*) '========================================='
            write(*,'(A,F16.8,A)') ' Reference energy (E0):    ', no_ham%E0, ' MeV'
            write(*,'(A,F16.8,A)') ' Correlation energy:       ', e_corr_ccsd, ' MeV'
            write(*,'(A,F16.8,A)') ' Total CCSD energy:        ', no_ham%E0 + e_corr_ccsd, ' MeV'
        else
            write(*,*) 'CCSD did not converge'
        end if
    end block
    write(*,*)

    ! Run CCSDT solver
    block
        real(8) :: e_corr_ccsdt
        logical :: ccsdt_converged
        integer :: max_iter_ccsdt
        real(8) :: tol_ccsdt

        max_iter_ccsdt = 50
        tol_ccsdt = 1.0d-6

        call ccsdt_solver(no_ham, max_iter_ccsdt, tol_ccsdt, e_corr_ccsdt, ccsdt_converged)

        if (ccsdt_converged) then
            write(*,*) '========================================='
            write(*,*) '   CCSDT Calculation Successful!'
            write(*,*) '========================================='
            write(*,'(A,F16.8,A)') ' Reference energy (E0):    ', no_ham%E0, ' MeV'
            write(*,'(A,F16.8,A)') ' Correlation energy:       ', e_corr_ccsdt, ' MeV'
            write(*,'(A,F16.8,A)') ' Total CCSDT energy:       ', no_ham%E0 + e_corr_ccsdt, ' MeV'
        else
            write(*,*) 'CCSDT did not converge'
        end if
    end block
    write(*,*)

    ! Run CCSDTQ solver
    block
        real(8) :: e_corr_ccsdtq
        logical :: ccsdtq_converged
        integer :: max_iter_ccsdtq
        real(8) :: tol_ccsdtq

        max_iter_ccsdtq = 50
        tol_ccsdtq = 1.0d-6

        call ccsdtq_solver(no_ham, max_iter_ccsdtq, tol_ccsdtq, e_corr_ccsdtq, ccsdtq_converged)

        if (ccsdtq_converged) then
            write(*,*) '========================================='
            write(*,*) '   CCSDTQ Calculation Successful!'
            write(*,*) '========================================='
            write(*,'(A,F16.8,A)') ' Reference energy (E0):    ', no_ham%E0, ' MeV'
            write(*,'(A,F16.8,A)') ' Correlation energy:       ', e_corr_ccsdtq, ' MeV'
            write(*,'(A,F16.8,A)') ' Total CCSDTQ energy:      ', no_ham%E0 + e_corr_ccsdtq, ' MeV'
        else
            write(*,*) 'CCSDTQ did not converge'
        end if
    end block
    write(*,*)
    
    ! Cleanup
    call deallocate_sp_orbits(j_orbits)
    call deallocate_sp_orbits(m_orbits)
    call deallocate_potential(j_potential)
    if (allocated(m_v1b)) deallocate(m_v1b)
    if (allocated(m_v2b)) deallocate(m_v2b)
    if (allocated(occ_indices)) deallocate(occ_indices)
    if (allocated(sp_energies)) deallocate(sp_energies)
    if (allocated(rho)) deallocate(rho)
    if (allocated(sp_coeffs)) deallocate(sp_coeffs)
    
    call deallocate_no_ham(no_ham)
    
    ierr = talsh_shutdown()
    write(*,*) 'Program completed successfully'
    
end program ccm_main

