module ccd_solver_module
    use talsh
    use tensor_algebra
    use, intrinsic:: ISO_C_BINDING
    use hartree_fock_module
    use amplitude_storage_module
    implicit none
    
    complex(8), parameter :: ZERO=(0.D0,0.D0), ONE=(1.D0,0.D0)
    
contains

    !===========================================================================
    ! CCD Solver - iterative solver for Coupled Cluster Doubles
    ! 
    ! This mimics the Python ccd_tenpi_solver.py structure
    ! Uses talsh_tenpi_ccd.F90 equations to compute residuals
    !===========================================================================
    subroutine ccd_solver(no_ham, max_iter, tol, e_corr, converged)
        type(normal_ordered_hamiltonian), intent(in) :: no_ham
        integer, intent(in) :: max_iter
        real(8), intent(in) :: tol
        real(8), intent(out) :: e_corr
        logical, intent(out) :: converged
        
        ! Local variables
        integer :: nocc, nvir, n_states
        integer :: ierr, iter, a, b, i, j, p, q, r, s
        real(8) :: e_corr_old, delta_e, r_max, r_rms, denom
        integer :: diis_start, diis_max, diis_count, diis_pos, diis_size, diis_k, diis_l
        real(8), allocatable :: eps_o(:), eps_v(:)
        real(8), allocatable :: t2_real(:,:,:,:), r2_real(:,:,:,:)
        real(8), allocatable :: t2_new(:,:,:,:)
        real(8), allocatable :: diis_t2(:,:,:,:,:), diis_err(:,:,:,:,:)
        real(8), allocatable :: diis_b(:,:), diis_rhs(:), diis_coeff(:)
        integer, allocatable :: diis_order(:)
        complex(8), allocatable, target :: t2_cmplx(:,:,:,:), r2_cmplx(:,:,:,:)
        complex(8), allocatable, target :: z0_cmplx(:)
        
        ! Fock blocks
        real(8), allocatable :: F_oo(:,:), F_vv(:,:)
        complex(8), allocatable, target :: F_oo_cmplx(:,:), F_vv_cmplx(:,:)
        
        ! Gamma blocks  
        real(8), allocatable :: V_oooo(:,:,:,:), V_vvoo(:,:,:,:)
        real(8), allocatable :: V_voov(:,:,:,:), V_oovv(:,:,:,:), V_vvvv(:,:,:,:)
        complex(8), allocatable, target :: V_oooo_cmplx(:,:,:,:), V_vvoo_cmplx(:,:,:,:)
        complex(8), allocatable, target :: V_voov_cmplx(:,:,:,:), V_oovv_cmplx(:,:,:,:)
        complex(8), allocatable, target :: V_vvvv_cmplx(:,:,:,:)
        
        ! TALSH tensor objects for Fock blocks
        type(talsh_tens_t) :: F1_oo      ! hole-hole
        type(talsh_tens_t) :: F4_vv      ! particle-particle
        
        ! TALSH tensor objects for Gamma blocks
        type(talsh_tens_t) :: V1_oooo    ! hole-hole-hole-hole
        type(talsh_tens_t) :: V3_vvoo    ! particle-particle-hole-hole
        type(talsh_tens_t) :: V5_voov    ! particle-hole-hole-particle
        type(talsh_tens_t) :: V7_oovv    ! hole-hole-particle-particle
        type(talsh_tens_t) :: V9_vvvv    ! particle-particle-particle-particle
        
        ! TALSH tensor objects for amplitudes and residuals
        type(talsh_tens_t) :: T2         ! T2 amplitudes (VVOO)
        type(talsh_tens_t) :: Z0         ! Scalar (energy)
        type(talsh_tens_t) :: Z2         ! T2 residual (VVOO)
        
        ! Extract dimensions
        n_states = no_ham%n_states
        nocc = no_ham%nocc
        nvir = no_ham%nvir
        
        write(*,*)
        write(*,*) '========================================='
        write(*,*) '   CCD Solver (TALSH-accelerated)'
        write(*,*) '========================================='
        write(*,'(A,I4)') ' Number of occupied states:  ', nocc
        write(*,'(A,I4)') ' Number of virtual states:   ', nvir
        write(*,'(A,I4)') ' Maximum iterations:         ', max_iter
        write(*,'(A,ES10.2)') ' Convergence tolerance:      ', tol
        write(*,*)
        
        ! Allocate work arrays
        allocate(eps_o(nocc), eps_v(nvir))
        allocate(t2_real(nvir, nvir, nocc, nocc))
        allocate(r2_real(nvir, nvir, nocc, nocc))
        allocate(t2_new(nvir, nvir, nocc, nocc))
        allocate(t2_cmplx(nvir, nvir, nocc, nocc))
        allocate(r2_cmplx(nvir, nvir, nocc, nocc))
        allocate(z0_cmplx(1))

        diis_start = 3
        diis_max = 6
        diis_count = 0
        diis_pos = 0
        allocate(diis_t2(nvir, nvir, nocc, nocc, diis_max))
        allocate(diis_err(nvir, nvir, nocc, nocc, diis_max))
        allocate(diis_order(diis_max))
        
        ! Extract orbital energies (diagonal of Fock matrix in HF basis)
        do i = 1, nocc
            eps_o(i) = no_ham%f_hf(i, i)
        end do
        do a = 1, nvir
            eps_v(a) = no_ham%f_hf(nocc + a, nocc + a)
        end do
        
        write(*,*) 'Sample occupied orbital energies:'
        write(*,'(5F12.6)') (eps_o(i), i=1, min(5,nocc))
        write(*,*) 'Sample virtual orbital energies:'
        write(*,'(5F12.6)') (eps_v(a), a=1, min(5,nvir))
        write(*,*)
        write(*,*) 'Checking Fock matrix structure:'
        write(*,'(A,2I5)') ' Fock shape: ', n_states, n_states
        write(*,'(A,F12.6)') ' Fock(1,1): ', no_ham%f_hf(1,1)
        write(*,'(A,F12.6)') ' Fock(nocc,nocc): ', no_ham%f_hf(nocc,nocc)
        write(*,'(A,F12.6)') ' Fock(nocc+1,nocc+1): ', no_ham%f_hf(nocc+1,nocc+1)
        write(*,'(A,F12.6)') ' Fock(n_states,n_states): ', no_ham%f_hf(n_states,n_states)
        write(*,*)
        
        ! Extract Fock blocks
        allocate(F_oo(nocc, nocc), F_vv(nvir, nvir))
        allocate(F_oo_cmplx(nocc, nocc), F_vv_cmplx(nvir, nvir))
        
        F_oo = no_ham%f_hf(1:nocc, 1:nocc)
        F_vv = no_ham%f_hf(nocc+1:n_states, nocc+1:n_states)
        
        F_oo_cmplx = cmplx(F_oo, 0.0d0, kind=8)
        F_vv_cmplx = cmplx(F_vv, 0.0d0, kind=8)
        
        ! Extract Gamma blocks from HF basis
        allocate(V_oooo(nocc, nocc, nocc, nocc))
        allocate(V_vvoo(nvir, nvir, nocc, nocc))
        allocate(V_voov(nvir, nocc, nocc, nvir))
        allocate(V_oovv(nocc, nocc, nvir, nvir))
        allocate(V_vvvv(nvir, nvir, nvir, nvir))
        
        allocate(V_oooo_cmplx(nocc, nocc, nocc, nocc))
        allocate(V_vvoo_cmplx(nvir, nvir, nocc, nocc))
        allocate(V_voov_cmplx(nvir, nocc, nocc, nvir))
        allocate(V_oovv_cmplx(nocc, nocc, nvir, nvir))
        allocate(V_vvvv_cmplx(nvir, nvir, nvir, nvir))
        
        ! OOOO block
        do i = 1, nocc
            do j = 1, nocc
                do p = 1, nocc
                    do q = 1, nocc
                        V_oooo(i,j,p,q) = no_ham%Gamma_hf(i, j, p, q)
                    end do
                end do
            end do
        end do
        
        ! VVOO block
        do a = 1, nvir
            do b = 1, nvir
                do i = 1, nocc
                    do j = 1, nocc
                        V_vvoo(a,b,i,j) = no_ham%Gamma_hf(nocc+a, nocc+b, i, j)
                    end do
                end do
            end do
        end do
        
        write(*,*) 'Sample Gamma_hf elements (original indexing):'
        write(*,'(A,4I4,A,F12.6)') ' Gamma_hf(',nocc+1,nocc+1,1,1,') = ', no_ham%Gamma_hf(nocc+1,nocc+1,1,1)
        write(*,'(A,4I4,A,F12.6)') ' Gamma_hf(',1,1,1,1,') = ', no_ham%Gamma_hf(1,1,1,1)
        write(*,'(A,F12.6)') ' Max |Gamma_hf|: ', maxval(abs(no_ham%Gamma_hf))
        write(*,*)
        
        ! VOOV block
        do a = 1, nvir
            do i = 1, nocc
                do j = 1, nocc
                    do b = 1, nvir
                        V_voov(a,i,j,b) = no_ham%Gamma_hf(nocc+a, i, j, nocc+b)
                    end do
                end do
            end do
        end do
        
        ! OOVV block
        do i = 1, nocc
            do j = 1, nocc
                do a = 1, nvir
                    do b = 1, nvir
                        V_oovv(i,j,a,b) = no_ham%Gamma_hf(i, j, nocc+a, nocc+b)
                    end do
                end do
            end do
        end do
        
        ! VVVV block
        do a = 1, nvir
            do b = 1, nvir
                do p = 1, nvir
                    do q = 1, nvir
                        V_vvvv(a,b,p,q) = no_ham%Gamma_hf(nocc+a, nocc+b, nocc+p, nocc+q)
                    end do
                end do
            end do
        end do
        
        ! Convert to complex
        V_oooo_cmplx = cmplx(V_oooo, 0.0d0, kind=8)
        V_vvoo_cmplx = cmplx(V_vvoo, 0.0d0, kind=8)
        V_voov_cmplx = cmplx(V_voov, 0.0d0, kind=8)
        V_oovv_cmplx = cmplx(V_oovv, 0.0d0, kind=8)
        V_vvvv_cmplx = cmplx(V_vvvv, 0.0d0, kind=8)
        
        write(*,*) 'Sample V_vvoo elements:'
        write(*,'(A,F12.6)') ' V_vvoo(1,1,1,1) = ', V_vvoo(1,1,1,1)
        write(*,'(A,F12.6)') ' V_vvoo(1,2,1,2) = ', V_vvoo(1,2,1,2)
        write(*,'(A,F12.6)') ' Max |V_vvoo|: ', maxval(abs(V_vvoo))
        write(*,'(A,F12.6)') ' Max |V_oovv|: ', maxval(abs(V_oovv))
        write(*,*)
        
        ! Initialize TALSH tensors (let TALSH allocate memory, then we'll copy data in)
        write(*,*) 'Constructing TALSH tensors with TALSH-managed memory...'
        ierr = talsh_tensor_construct(F1_oo, C8, (/nocc, nocc/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing F1_oo'
        
        ierr = talsh_tensor_construct(F4_vv, C8, (/nvir, nvir/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing F4_vv'
        
        ierr = talsh_tensor_construct(V1_oooo, C8, (/nocc, nocc, nocc, nocc/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing V1_oooo'
        
        ierr = talsh_tensor_construct(V3_vvoo, C8, (/nvir, nvir, nocc, nocc/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing V3_vvoo'
        
        ierr = talsh_tensor_construct(V5_voov, C8, (/nvir, nocc, nocc, nvir/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing V5_voov'
        
        ierr = talsh_tensor_construct(V7_oovv, C8, (/nocc, nocc, nvir, nvir/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing V7_oovv'
        
        ierr = talsh_tensor_construct(V9_vvvv, C8, (/nvir, nvir, nvir, nvir/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing V9_vvvv'
        
        ierr = talsh_tensor_construct(T2, C8, (/nvir, nvir, nocc, nocc/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing T2'
        
        ! Z0 is a scalar (rank-0 tensor) - pass empty dimensions array
        block
            integer, dimension(0) :: empty_dims
            ierr = talsh_tensor_construct(Z0, C8, empty_dims, init_val=ZERO)
        end block
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing Z0'
        
        ierr = talsh_tensor_construct(Z2, C8, (/nvir, nvir, nocc, nocc/), init_val=ZERO)
        if (ierr /= TALSH_SUCCESS) stop 'Error constructing Z2'
        
        write(*,*) 'All TALSH tensors constructed successfully'
        write(*,*)
        
        ! Copy data into TALSH tensors
        write(*,*) 'Copying data into TALSH tensors...'
        call copy_to_talsh_2d(F1_oo, F_oo_cmplx, nocc, nocc)
        call copy_to_talsh_2d(F4_vv, F_vv_cmplx, nvir, nvir)
        call copy_to_talsh_4d(V1_oooo, V_oooo_cmplx, nocc, nocc, nocc, nocc)
        call copy_to_talsh_4d(V3_vvoo, V_vvoo_cmplx, nvir, nvir, nocc, nocc)
        call copy_to_talsh_4d(V5_voov, V_voov_cmplx, nvir, nocc, nocc, nvir)
        call copy_to_talsh_4d(V7_oovv, V_oovv_cmplx, nocc, nocc, nvir, nvir)
        call copy_to_talsh_4d(V9_vvvv, V_vvvv_cmplx, nvir, nvir, nvir, nvir)
        write(*,*) 'Data copied into input tensors'
        write(*,*)
        
        ! Initialize T2 with MP2 guess: T2(a,b,i,j) = V(a,b,i,j) / (eps_i + eps_j - eps_a - eps_b)
        write(*,*) 'Initializing T2 amplitudes with MP2 guess...'
        t2_real = 0.0d0
        do a = 1, nvir
            do b = 1, nvir
                do i = 1, nocc
                    do j = 1, nocc
                        denom = eps_o(i) + eps_o(j) - eps_v(a) - eps_v(b)
                        if (abs(denom) > 1.0d-12) then
                            t2_real(a,b,i,j) = V_vvoo(a,b,i,j) / denom
                        end if
                    end do
                end do
            end do
        end do
        
        t2_cmplx = cmplx(t2_real, 0.0d0, kind=8)
        
        ! Copy initial T2 into TALSH tensor
        call copy_to_talsh_4d(T2, t2_cmplx, nvir, nvir, nocc, nocc)
        
        write(*,'(A,ES12.4)') ' Max |T2| initial: ', maxval(abs(t2_real))
        write(*,*)
        
        ! Main iteration loop
        write(*,'(A4,A18,A14,A14,A14)') 'Iter', 'E_corr', 'Delta_E', '|R|_max', '|R|_rms'
        write(*,*) '-------------------------------------------------------------------'
        
        e_corr_old = 0.0d0
        converged = .false.
        
        do iter = 0, max_iter - 1
            ! Reset Z0 and Z2 to zero before accumulating residuals
            ierr = talsh_tensor_init(Z0, ZERO)
            ierr = talsh_tensor_init(Z2, ZERO)
            
            ! Call the tenpi-generated residual routine
            call talsh_tenpi_ccd(nocc, nvir, F1_oo, F4_vv, T2, V1_oooo, V7_oovv, &
                                 V5_voov, V3_vvoo, V9_vvvv, Z0, Z2)
            
            ! Extract results from TALSH tensors
            ! Z0 is a scalar - extract directly
            call copy_from_talsh_scalar(Z0, z0_cmplx(1))
            call copy_from_talsh_4d(Z2, r2_cmplx, nvir, nvir, nocc, nocc)
            
            e_corr = real(z0_cmplx(1), 8)
            r2_real = real(r2_cmplx, 8)
            
            ! Compute convergence metrics
            delta_e = abs(e_corr - e_corr_old)
            r_max = maxval(abs(r2_real))
            r_rms = sqrt(sum(r2_real**2) / real(nvir*nvir*nocc*nocc, 8))
            
            write(*,'(I4,F18.10,3ES14.4)') iter, e_corr, delta_e, r_max, r_rms
            
            ! Check convergence
            if (delta_e < tol .and. r_rms < tol) then
                converged = .true.
                write(*,*)
                write(*,*) 'CCD converged!'
                exit
            end if
            
            e_corr_old = e_corr
            
            ! Compute T2 update: t2_new = t2_old + r2 / D
            ! where D(a,b,i,j) = eps_i + eps_j - eps_a - eps_b
            t2_new = t2_real
            do a = 1, nvir
                do b = 1, nvir
                    do i = 1, nocc
                        do j = 1, nocc
                            denom = eps_o(i) + eps_o(j) - eps_v(a) - eps_v(b)
                            if (abs(denom) > 1.0d-12) then
                                t2_new(a,b,i,j) = t2_real(a,b,i,j) + r2_real(a,b,i,j) / denom
                            end if
                        end do
                    end do
                end do
            end do

            ! DIIS acceleration to stabilize CCD iterations
            if (iter >= diis_start) then
                diis_pos = diis_pos + 1
                if (diis_pos > diis_max) diis_pos = 1

                if (diis_count < diis_max) then
                    diis_count = diis_count + 1
                    diis_order(diis_count) = diis_pos
                else
                    do diis_k = 1, diis_max - 1
                        diis_order(diis_k) = diis_order(diis_k + 1)
                    end do
                    diis_order(diis_max) = diis_pos
                end if

                diis_t2(:,:,:,:,diis_pos) = t2_new
                diis_err(:,:,:,:,diis_pos) = r2_real

                if (diis_count >= 2) then
                    diis_size = diis_count
                    allocate(diis_b(diis_size + 1, diis_size + 1))
                    allocate(diis_rhs(diis_size + 1))
                    allocate(diis_coeff(diis_size + 1))

                    diis_b = 0.0d0
                    do diis_k = 1, diis_size
                        do diis_l = 1, diis_size
                            diis_b(diis_k, diis_l) = sum(diis_err(:,:,:,:,diis_order(diis_k)) * &
                                                        diis_err(:,:,:,:,diis_order(diis_l)))
                        end do
                        diis_b(diis_k, diis_size + 1) = -1.0d0
                        diis_b(diis_size + 1, diis_k) = -1.0d0
                    end do
                    diis_b(diis_size + 1, diis_size + 1) = 0.0d0

                    diis_rhs = 0.0d0
                    diis_rhs(diis_size + 1) = -1.0d0

                    call solve_linear_system(diis_b, diis_rhs, diis_size + 1, diis_coeff)

                    t2_real = 0.0d0
                    do diis_k = 1, diis_size
                        t2_real = t2_real + diis_coeff(diis_k) * diis_t2(:,:,:,:,diis_order(diis_k))
                    end do

                    deallocate(diis_b, diis_rhs, diis_coeff)
                else
                    t2_real = t2_new
                end if
            else
                t2_real = t2_new
            end if

            ! Update t2_cmplx and copy back into TALSH tensor for next iteration
            t2_cmplx = cmplx(t2_real, 0.0d0, kind=8)
            call copy_to_talsh_4d(T2, t2_cmplx, nvir, nvir, nocc, nocc)
            
        end do
        
        if (.not. converged) then
            write(*,*)
            write(*,*) 'Warning: CCD did not converge within maximum iterations!'
        end if
        
        write(*,*)
        write(*,'(A,F16.8,A)') ' Final CCD correlation energy: ', e_corr, ' MeV'
        write(*,'(A,F16.8,A)') ' Total energy (E0 + E_corr):   ', no_ham%E0 + e_corr, ' MeV'
        write(*,*)
        
        ! Cleanup
        ierr = talsh_tensor_destruct(F1_oo)
        ierr = talsh_tensor_destruct(F4_vv)
        ierr = talsh_tensor_destruct(V1_oooo)
        ierr = talsh_tensor_destruct(V3_vvoo)
        ierr = talsh_tensor_destruct(V5_voov)
        ierr = talsh_tensor_destruct(V7_oovv)
        ierr = talsh_tensor_destruct(V9_vvvv)
        ierr = talsh_tensor_destruct(T2)
        ierr = talsh_tensor_destruct(Z0)
        ierr = talsh_tensor_destruct(Z2)
        
        ! Store T2 for use by higher-order methods
        call store_t2(t2_real)
        
        deallocate(eps_o, eps_v, t2_real, r2_real, t2_new, t2_cmplx, r2_cmplx, z0_cmplx)
        deallocate(diis_t2, diis_err, diis_order)
        deallocate(F_oo, F_vv, F_oo_cmplx, F_vv_cmplx)
        deallocate(V_oooo, V_vvoo, V_voov, V_oovv, V_vvvv)
        deallocate(V_oooo_cmplx, V_vvoo_cmplx, V_voov_cmplx, V_oovv_cmplx, V_vvvv_cmplx)
        
    end subroutine ccd_solver

    !===========================================================================
    ! Helper routines to copy data to/from TALSH tensors
    !===========================================================================
    
    subroutine copy_to_talsh_2d(tens, data, n1, n2)
        type(talsh_tens_t), intent(inout) :: tens
        complex(8), intent(in) :: data(n1, n2)
        integer, intent(in) :: n1, n2
        type(C_PTR) :: body_p
        complex(8), pointer :: body(:)
        integer :: ierr, vol, i, j, idx
        
        ierr = talsh_tensor_get_body_access(tens, body_p, C8, 0, DEV_HOST)
        vol = talsh_tensor_volume(tens)
        call c_f_pointer(body_p, body, (/vol/))
        
        idx = 1
        do j = 1, n2
            do i = 1, n1
                body(idx) = data(i, j)
                idx = idx + 1
            end do
        end do
    end subroutine copy_to_talsh_2d
    
    subroutine copy_to_talsh_4d(tens, data, n1, n2, n3, n4)
        type(talsh_tens_t), intent(inout) :: tens
        complex(8), intent(in) :: data(n1, n2, n3, n4)
        integer, intent(in) :: n1, n2, n3, n4
        type(C_PTR) :: body_p
        complex(8), pointer :: body(:)
        integer :: ierr, vol, i, j, k, l, idx
        
        ierr = talsh_tensor_get_body_access(tens, body_p, C8, 0, DEV_HOST)
        vol = talsh_tensor_volume(tens)
        call c_f_pointer(body_p, body, (/vol/))
        
        idx = 1
        do l = 1, n4
            do k = 1, n3
                do j = 1, n2
                    do i = 1, n1
                        body(idx) = data(i, j, k, l)
                        idx = idx + 1
                    end do
                end do
            end do
        end do
    end subroutine copy_to_talsh_4d
    
    subroutine copy_from_talsh_1d(tens, data, n1)
        type(talsh_tens_t), intent(inout) :: tens
        complex(8), intent(out) :: data(n1)
        integer, intent(in) :: n1
        type(C_PTR) :: body_p
        complex(8), pointer :: body(:)
        integer :: ierr, vol, i
        
        ierr = talsh_tensor_get_body_access(tens, body_p, C8, 0, DEV_HOST)
        vol = talsh_tensor_volume(tens)
        call c_f_pointer(body_p, body, (/vol/))
        
        do i = 1, n1
            data(i) = body(i)
        end do
    end subroutine copy_from_talsh_1d
    
    subroutine copy_from_talsh_4d(tens, data, n1, n2, n3, n4)
        type(talsh_tens_t), intent(inout) :: tens
        complex(8), intent(out) :: data(n1, n2, n3, n4)
        integer, intent(in) :: n1, n2, n3, n4
        type(C_PTR) :: body_p
        complex(8), pointer :: body(:)
        integer :: ierr, vol, i, j, k, l, idx
        
        ierr = talsh_tensor_get_body_access(tens, body_p, C8, 0, DEV_HOST)
        vol = talsh_tensor_volume(tens)
        call c_f_pointer(body_p, body, (/vol/))
        
        idx = 1
        do l = 1, n4
            do k = 1, n3
                do j = 1, n2
                    do i = 1, n1
                        data(i, j, k, l) = body(idx)
                        idx = idx + 1
                    end do
                end do
            end do
        end do
    end subroutine copy_from_talsh_4d
    
    subroutine copy_from_talsh_scalar(tens, scalar_val)
        type(talsh_tens_t), intent(inout) :: tens
        complex(8), intent(out) :: scalar_val
        type(C_PTR) :: body_p
        complex(8), pointer :: body(:)
        integer :: ierr, vol
        
        ierr = talsh_tensor_get_body_access(tens, body_p, C8, 0, DEV_HOST)
        vol = talsh_tensor_volume(tens)
        call c_f_pointer(body_p, body, (/vol/))
        
        scalar_val = body(1)
    end subroutine copy_from_talsh_scalar

    subroutine solve_linear_system(a, b, n, x)
        real(8), intent(inout) :: a(n, n)
        real(8), intent(inout) :: b(n)
        integer, intent(in) :: n
        real(8), intent(out) :: x(n)
        integer :: i, j, k, pivot
        real(8) :: max_val, tmp

        ! Gaussian elimination with partial pivoting
        do k = 1, n - 1
            max_val = abs(a(k, k))
            pivot = k
            do i = k + 1, n
                if (abs(a(i, k)) > max_val) then
                    max_val = abs(a(i, k))
                    pivot = i
                end if
            end do

            if (pivot /= k) then
                do j = k, n
                    tmp = a(k, j)
                    a(k, j) = a(pivot, j)
                    a(pivot, j) = tmp
                end do
                tmp = b(k)
                b(k) = b(pivot)
                b(pivot) = tmp
            end if

            if (abs(a(k, k)) < 1.0d-14) cycle

            do i = k + 1, n
                tmp = a(i, k) / a(k, k)
                do j = k, n
                    a(i, j) = a(i, j) - tmp * a(k, j)
                end do
                b(i) = b(i) - tmp * b(k)
            end do
        end do

        x(n) = b(n) / a(n, n)
        do i = n - 1, 1, -1
            tmp = b(i)
            do j = i + 1, n
                tmp = tmp - a(i, j) * x(j)
            end do
            x(i) = tmp / a(i, i)
        end do
    end subroutine solve_linear_system

end module ccd_solver_module
