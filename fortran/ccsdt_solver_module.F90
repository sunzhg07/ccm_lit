module ccsdt_solver_module
    use hartree_fock_module
    use talsh
    use tensor_algebra
    use, intrinsic :: iso_c_binding
    use amplitude_storage_module
    implicit none
    
    complex(8), parameter :: ZERO=(0.D0,0.D0), ONE=(1.D0,0.D0)
    
contains

    !===========================================================================
    ! CCSDT Solver - iterative solver for Coupled Cluster Singles, Doubles, Triples
    ! Uses talsh_tenpi_ccsdt.F90 equations to compute residuals
    !===========================================================================
    subroutine ccsdt_solver(no_ham, max_iter, tol, e_corr, converged)
        type(normal_ordered_hamiltonian), intent(in) :: no_ham
        integer, intent(in) :: max_iter
        real(8), intent(in) :: tol
        real(8), intent(out) :: e_corr
        logical, intent(out) :: converged
        
        integer :: nocc, nvir, n_states
        integer :: ierr, iter, a, b, c, i, j, k
        integer :: diis_start, diis_max, diis_count, diis_pos, diis_size, diis_k, diis_l
        real(8) :: e_corr_old, delta_e
        real(8) :: r1_max, r1_rms, r2_max, r2_rms, r3_max, r3_rms
        real(8) :: denom
        
        type(talsh_tens_t) :: F1_oo, F2_vo, F3_ov, F4_vv
        type(talsh_tens_t) :: V1_oooo, V2_vooo, V3_vvoo, V4_ooov
        type(talsh_tens_t) :: V5_voov, V6_vvov, V7_oovv, V8_vovv, V9_vvvv
        type(talsh_tens_t) :: T1, T2, T3, Z0, Z1, Z2, Z3
        
        real(8), allocatable :: eps_o(:), eps_v(:)
        real(8), allocatable :: t1_real(:,:), r1_real(:,:)
        real(8), allocatable :: t2_real(:,:,:,:), r2_real(:,:,:,:)
        real(8), allocatable :: t3_real(:,:,:,:,:,:), r3_real(:,:,:,:,:,:)
        real(8), allocatable :: t1_new(:,:), t2_new(:,:,:,:), t3_new(:,:,:,:,:,:)
        real(8), allocatable :: diis_t1(:,:,:), diis_t2(:,:,:,:,:), diis_t3(:,:,:,:,:,:,:)
        real(8), allocatable :: diis_err1(:,:,:), diis_err2(:,:,:,:,:), diis_err3(:,:,:,:,:,:,:)
        real(8), allocatable :: diis_b(:,:), diis_rhs(:), diis_coeff(:)
        integer, allocatable :: diis_order(:)
        complex(8), allocatable :: t1_cmplx(:,:), r1_cmplx(:,:)
        complex(8), allocatable :: t2_cmplx(:,:,:,:), r2_cmplx(:,:,:,:)
        complex(8), allocatable :: t3_cmplx(:,:,:,:,:,:), r3_cmplx(:,:,:,:,:,:)
        complex(8), allocatable :: z0_cmplx(:)
        
        real(8), allocatable :: F_oo(:,:), F_vo(:,:), F_ov(:,:), F_vv(:,:)
        complex(8), allocatable :: F_oo_cmplx(:,:), F_vo_cmplx(:,:)
        complex(8), allocatable :: F_ov_cmplx(:,:), F_vv_cmplx(:,:)
        real(8), allocatable :: V_oooo(:,:,:,:), V_vooo(:,:,:,:), V_vvoo(:,:,:,:)
        real(8), allocatable :: V_ooov(:,:,:,:), V_voov(:,:,:,:), V_vvov(:,:,:,:)
        real(8), allocatable :: V_oovv(:,:,:,:), V_vovv(:,:,:,:), V_vvvv(:,:,:,:)
        complex(8), allocatable :: V_oooo_cmplx(:,:,:,:), V_vooo_cmplx(:,:,:,:)
        complex(8), allocatable :: V_vvoo_cmplx(:,:,:,:), V_ooov_cmplx(:,:,:,:)
        complex(8), allocatable :: V_voov_cmplx(:,:,:,:), V_vvov_cmplx(:,:,:,:)
        complex(8), allocatable :: V_oovv_cmplx(:,:,:,:), V_vovv_cmplx(:,:,:,:)
        complex(8), allocatable :: V_vvvv_cmplx(:,:,:,:)
        
        write(*,*)
        write(*,*) '========================================='
        write(*,*) '   CCSDT Solver (TALSH-accelerated)'
        write(*,*) '========================================='
        
        nocc = no_ham%nocc
        nvir = no_ham%nvir
        n_states = no_ham%n_states
        
        write(*,'(A,I5)') ' Number of occupied states:', nocc
        write(*,'(A,I5)') ' Number of virtual states:', nvir
        write(*,'(A,I5)') ' Maximum iterations:', max_iter
        write(*,'(A,ES12.2)') ' Convergence tolerance:', tol
        write(*,*)
        
        allocate(eps_o(nocc), eps_v(nvir))
        allocate(t1_real(nvir, nocc), r1_real(nvir, nocc))
        allocate(t2_real(nvir, nvir, nocc, nocc), r2_real(nvir, nvir, nocc, nocc))
        allocate(t3_real(nvir, nvir, nvir, nocc, nocc, nocc))
        allocate(r3_real(nvir, nvir, nvir, nocc, nocc, nocc))
        allocate(t1_new(nvir, nocc))
        allocate(t2_new(nvir, nvir, nocc, nocc))
        allocate(t3_new(nvir, nvir, nvir, nocc, nocc, nocc))
        allocate(t1_cmplx(nvir, nocc), r1_cmplx(nvir, nocc))
        allocate(t2_cmplx(nvir, nvir, nocc, nocc), r2_cmplx(nvir, nvir, nocc, nocc))
        allocate(t3_cmplx(nvir, nvir, nvir, nocc, nocc, nocc))
        allocate(r3_cmplx(nvir, nvir, nvir, nocc, nocc, nocc))
        allocate(z0_cmplx(1))

        diis_start = 3
        diis_max = 6
        diis_count = 0
        diis_pos = 0
        allocate(diis_t1(nvir, nocc, diis_max))
        allocate(diis_t2(nvir, nvir, nocc, nocc, diis_max))
        allocate(diis_t3(nvir, nvir, nvir, nocc, nocc, nocc, diis_max))
        allocate(diis_err1(nvir, nocc, diis_max))
        allocate(diis_err2(nvir, nvir, nocc, nocc, diis_max))
        allocate(diis_err3(nvir, nvir, nvir, nocc, nocc, nocc, diis_max))
        allocate(diis_order(diis_max))
        
        do i = 1, nocc
            eps_o(i) = no_ham%f_hf(i, i)
        end do
        do a = 1, nvir
            eps_v(a) = no_ham%f_hf(nocc + a, nocc + a)
        end do
        
        allocate(F_oo(nocc, nocc), F_vo(nvir, nocc))
        allocate(F_ov(nocc, nvir), F_vv(nvir, nvir))
        allocate(F_oo_cmplx(nocc, nocc), F_vo_cmplx(nvir, nocc))
        allocate(F_ov_cmplx(nocc, nvir), F_vv_cmplx(nvir, nvir))
        
        F_oo = no_ham%f_hf(1:nocc, 1:nocc)
        F_vo = no_ham%f_hf(nocc+1:n_states, 1:nocc)
        F_ov = no_ham%f_hf(1:nocc, nocc+1:n_states)
        F_vv = no_ham%f_hf(nocc+1:n_states, nocc+1:n_states)
        
        F_oo_cmplx = cmplx(F_oo, 0.0d0, kind=8)
        F_vo_cmplx = cmplx(F_vo, 0.0d0, kind=8)
        F_ov_cmplx = cmplx(F_ov, 0.0d0, kind=8)
        F_vv_cmplx = cmplx(F_vv, 0.0d0, kind=8)
        
        allocate(V_oooo(nocc, nocc, nocc, nocc))
        allocate(V_vooo(nvir, nocc, nocc, nocc))
        allocate(V_vvoo(nvir, nvir, nocc, nocc))
        allocate(V_ooov(nocc, nocc, nocc, nvir))
        allocate(V_voov(nvir, nocc, nocc, nvir))
        allocate(V_vvov(nvir, nvir, nocc, nvir))
        allocate(V_oovv(nocc, nocc, nvir, nvir))
        allocate(V_vovv(nvir, nocc, nvir, nvir))
        allocate(V_vvvv(nvir, nvir, nvir, nvir))
        
        allocate(V_oooo_cmplx(nocc, nocc, nocc, nocc))
        allocate(V_vooo_cmplx(nvir, nocc, nocc, nocc))
        allocate(V_vvoo_cmplx(nvir, nvir, nocc, nocc))
        allocate(V_ooov_cmplx(nocc, nocc, nocc, nvir))
        allocate(V_voov_cmplx(nvir, nocc, nocc, nvir))
        allocate(V_vvov_cmplx(nvir, nvir, nocc, nvir))
        allocate(V_oovv_cmplx(nocc, nocc, nvir, nvir))
        allocate(V_vovv_cmplx(nvir, nocc, nvir, nvir))
        allocate(V_vvvv_cmplx(nvir, nvir, nvir, nvir))
        
        do i = 1, nocc
            do j = 1, nocc
                do a = 1, nvir
                    do b = 1, nvir
                        V_vvoo(a,b,i,j) = no_ham%Gamma_hf(nocc+a, nocc+b, i, j)
                        V_oovv(i,j,a,b) = no_ham%Gamma_hf(i, j, nocc+a, nocc+b)
                        V_voov(a,i,j,b) = no_ham%Gamma_hf(nocc+a, i, j, nocc+b)
                    end do
                end do
                do k = 1, nocc
                    do a = 1, nvir
                        V_ooov(i,j,k,a) = no_ham%Gamma_hf(i, j, k, nocc+a)
                    end do
                end do
                do b = 1, nocc
                    do a = 1, nvir
                        V_vooo(a,i,j,b) = no_ham%Gamma_hf(nocc+a, i, j, b)
                    end do
                    do a = 1, nocc
                        V_oooo(i,j,a,b) = no_ham%Gamma_hf(i, j, a, b)
                    end do
                end do
            end do
        end do
        
        do a = 1, nvir
            do b = 1, nvir
                do i = 1, nocc
                    do j = 1, nvir
                        V_vvov(a,b,i,j) = no_ham%Gamma_hf(nocc+a, nocc+b, i, nocc+j)
                    end do
                    do j = 1, nvir
                        V_vovv(a,i,b,j) = no_ham%Gamma_hf(nocc+a, i, nocc+b, nocc+j)
                    end do
                end do
                do i = 1, nvir
                    do j = 1, nvir
                        V_vvvv(a,b,i,j) = no_ham%Gamma_hf(nocc+a, nocc+b, nocc+i, nocc+j)
                    end do
                end do
            end do
        end do
        
        V_oooo_cmplx = cmplx(V_oooo, 0.0d0, kind=8)
        V_vooo_cmplx = cmplx(V_vooo, 0.0d0, kind=8)
        V_vvoo_cmplx = cmplx(V_vvoo, 0.0d0, kind=8)
        V_ooov_cmplx = cmplx(V_ooov, 0.0d0, kind=8)
        V_voov_cmplx = cmplx(V_voov, 0.0d0, kind=8)
        V_vvov_cmplx = cmplx(V_vvov, 0.0d0, kind=8)
        V_oovv_cmplx = cmplx(V_oovv, 0.0d0, kind=8)
        V_vovv_cmplx = cmplx(V_vovv, 0.0d0, kind=8)
        V_vvvv_cmplx = cmplx(V_vvvv, 0.0d0, kind=8)
        
        write(*,*) 'Constructing TALSH tensors...'
        ierr = talsh_tensor_construct(F1_oo, C8, (/nocc, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(F2_vo, C8, (/nvir, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(F3_ov, C8, (/nocc, nvir/), init_val=ZERO)
        ierr = talsh_tensor_construct(F4_vv, C8, (/nvir, nvir/), init_val=ZERO)
        ierr = talsh_tensor_construct(V1_oooo, C8, (/nocc, nocc, nocc, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(V2_vooo, C8, (/nvir, nocc, nocc, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(V3_vvoo, C8, (/nvir, nvir, nocc, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(V4_ooov, C8, (/nocc, nocc, nocc, nvir/), init_val=ZERO)
        ierr = talsh_tensor_construct(V5_voov, C8, (/nvir, nocc, nocc, nvir/), init_val=ZERO)
        ierr = talsh_tensor_construct(V6_vvov, C8, (/nvir, nvir, nocc, nvir/), init_val=ZERO)
        ierr = talsh_tensor_construct(V7_oovv, C8, (/nocc, nocc, nvir, nvir/), init_val=ZERO)
        ierr = talsh_tensor_construct(V8_vovv, C8, (/nvir, nocc, nvir, nvir/), init_val=ZERO)
        ierr = talsh_tensor_construct(V9_vvvv, C8, (/nvir, nvir, nvir, nvir/), init_val=ZERO)
        ierr = talsh_tensor_construct(T1, C8, (/nvir, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(T2, C8, (/nvir, nvir, nocc, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(T3, C8, (/nvir, nvir, nvir, nocc, nocc, nocc/), init_val=ZERO)
        
        block
            integer, dimension(0) :: empty_dims
            ierr = talsh_tensor_construct(Z0, C8, empty_dims, init_val=ZERO)
        end block
        
        ierr = talsh_tensor_construct(Z1, C8, (/nvir, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(Z2, C8, (/nvir, nvir, nocc, nocc/), init_val=ZERO)
        ierr = talsh_tensor_construct(Z3, C8, (/nvir, nvir, nvir, nocc, nocc, nocc/), init_val=ZERO)
        
        write(*,*) 'Copying data into TALSH tensors...'
        call copy_to_talsh_2d(F1_oo, F_oo_cmplx, nocc, nocc)
        call copy_to_talsh_2d(F2_vo, F_vo_cmplx, nvir, nocc)
        call copy_to_talsh_2d(F3_ov, F_ov_cmplx, nocc, nvir)
        call copy_to_talsh_2d(F4_vv, F_vv_cmplx, nvir, nvir)
        call copy_to_talsh_4d(V1_oooo, V_oooo_cmplx, nocc, nocc, nocc, nocc)
        call copy_to_talsh_4d(V2_vooo, V_vooo_cmplx, nvir, nocc, nocc, nocc)
        call copy_to_talsh_4d(V3_vvoo, V_vvoo_cmplx, nvir, nvir, nocc, nocc)
        call copy_to_talsh_4d(V4_ooov, V_ooov_cmplx, nocc, nocc, nocc, nvir)
        call copy_to_talsh_4d(V5_voov, V_voov_cmplx, nvir, nocc, nocc, nvir)
        call copy_to_talsh_4d(V6_vvov, V_vvov_cmplx, nvir, nvir, nocc, nvir)
        call copy_to_talsh_4d(V7_oovv, V_oovv_cmplx, nocc, nocc, nvir, nvir)
        call copy_to_talsh_4d(V8_vovv, V_vovv_cmplx, nvir, nocc, nvir, nvir)
        call copy_to_talsh_4d(V9_vvvv, V_vvvv_cmplx, nvir, nvir, nvir, nvir)
        
        write(*,*) 'Initializing amplitudes...'
        
        ! Initialize T1, T2, T3
        if (have_t2 .and. allocated(stored_t2)) then
            ! Use CCD T2 as initial guess for CCSDT T1 and T2
            write(*,*) 'Initializing from CCD T2 amplitudes'
            t1_real = 0.0d0
            t2_real = stored_t2
            if (size(stored_t2, 1) == nvir .and. size(stored_t2, 3) == nocc) then
                ! Correct dimensions
            else
                write(*,*) 'Warning: stored T2 has different dimensions, using MP2 initialization'
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
            end if
        else
            ! Use MP2 initialization (default)
            t1_real = 0.0d0
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
        end if
        t3_real = 0.0d0
        
        t1_cmplx = cmplx(t1_real, 0.0d0, kind=8)
        t2_cmplx = cmplx(t2_real, 0.0d0, kind=8)
        t3_cmplx = cmplx(t3_real, 0.0d0, kind=8)
        
        call copy_to_talsh_2d(T1, t1_cmplx, nvir, nocc)
        call copy_to_talsh_4d(T2, t2_cmplx, nvir, nvir, nocc, nocc)
        call copy_to_talsh_6d(T3, t3_cmplx, nvir, nvir, nvir, nocc, nocc, nocc)
        
        write(*,'(A4,A18,A14,A14,A14,A14,A14)') 'Iter', 'E_corr', 'Delta_E', &
                                                 '|R1|_max', '|R1|_rms', '|R2|_max', '|R2|_rms'
        write(*,*) '---------------------------------------------------------------------------------'
        
        e_corr_old = 0.0d0
        converged = .false.
        
        do iter = 0, max_iter - 1
            ierr = talsh_tensor_init(Z0, ZERO)
            ierr = talsh_tensor_init(Z1, ZERO)
            ierr = talsh_tensor_init(Z2, ZERO)
            ierr = talsh_tensor_init(Z3, ZERO)
            
            call talsh_tenpi_ccsdt(nocc, nvir, F1_oo, F3_ov, F2_vo, F4_vv, T1, T2, T3, &
                                   V1_oooo, V4_ooov, V7_oovv, V2_vooo, V5_voov, V8_vovv, &
                                   V3_vvoo, V6_vvov, V9_vvvv, Z0, Z1, Z2, Z3)
            
            call copy_from_talsh_scalar(Z0, z0_cmplx(1))
            call copy_from_talsh_2d(Z1, r1_cmplx, nvir, nocc)
            call copy_from_talsh_4d(Z2, r2_cmplx, nvir, nvir, nocc, nocc)
            call copy_from_talsh_6d(Z3, r3_cmplx, nvir, nvir, nvir, nocc, nocc, nocc)
            
            e_corr = real(z0_cmplx(1), 8)
            r1_real = real(r1_cmplx, 8)
            r2_real = real(r2_cmplx, 8)
            r3_real = real(r3_cmplx, 8)
            
            delta_e = abs(e_corr - e_corr_old)
            r1_max = maxval(abs(r1_real))
            r1_rms = sqrt(sum(r1_real**2) / real(nvir*nocc, 8))
            r2_max = maxval(abs(r2_real))
            r2_rms = sqrt(sum(r2_real**2) / real(nvir*nvir*nocc*nocc, 8))
            r3_max = maxval(abs(r3_real))
            r3_rms = sqrt(sum(r3_real**2) / real(nvir**3 * nocc**3, 8))
            
            write(*,'(I4,F18.10,5ES14.4)') iter, e_corr, delta_e, r1_max, r1_rms, r2_max, r2_rms
            
            if (delta_e < tol .and. r1_rms < tol .and. r2_rms < tol .and. r3_rms < tol) then
                converged = .true.
                write(*,*)
                write(*,*) 'CCSDT converged!'
                exit
            end if
            
            e_corr_old = e_corr
            
            t1_new = t1_real
            do a = 1, nvir
                do i = 1, nocc
                    denom = eps_o(i) - eps_v(a)
                    if (abs(denom) > 1.0d-12) then
                        t1_new(a,i) = t1_real(a,i) + r1_real(a,i) / denom
                    end if
                end do
            end do
            
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

            t3_new = t3_real
            do a = 1, nvir
                do b = 1, nvir
                    do c = 1, nvir
                        do i = 1, nocc
                            do j = 1, nocc
                                do k = 1, nocc
                                    denom = eps_o(i) + eps_o(j) + eps_o(k) - eps_v(a) - eps_v(b) - eps_v(c)
                                    if (abs(denom) > 1.0d-12) then
                                        t3_new(a,b,c,i,j,k) = t3_real(a,b,c,i,j,k) + r3_real(a,b,c,i,j,k) / denom
                                    end if
                                end do
                            end do
                        end do
                    end do
                end do
            end do

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

                diis_t1(:,:,diis_pos) = t1_new
                diis_t2(:,:,:,:,diis_pos) = t2_new
                diis_t3(:,:,:,:,:,:,diis_pos) = t3_new
                diis_err1(:,:,diis_pos) = r1_real
                diis_err2(:,:,:,:,diis_pos) = r2_real
                diis_err3(:,:,:,:,:,:,diis_pos) = r3_real

                if (diis_count >= 2) then
                    diis_size = diis_count
                    allocate(diis_b(diis_size + 1, diis_size + 1))
                    allocate(diis_rhs(diis_size + 1))
                    allocate(diis_coeff(diis_size + 1))

                    diis_b = 0.0d0
                    do diis_k = 1, diis_size
                        do diis_l = 1, diis_size
                                     diis_b(diis_k, diis_l) = sum(diis_err1(:,:,diis_order(diis_k)) * &
                                                                          diis_err1(:,:,diis_order(diis_l))) + &
                                                                      sum(diis_err2(:,:,:,:,diis_order(diis_k)) * &
                                                                          diis_err2(:,:,:,:,diis_order(diis_l))) + &
                                                                      sum(diis_err3(:,:,:,:,:,:,diis_order(diis_k)) * &
                                                                          diis_err3(:,:,:,:,:,:,diis_order(diis_l)))
                        end do
                        diis_b(diis_k, diis_size + 1) = -1.0d0
                        diis_b(diis_size + 1, diis_k) = -1.0d0
                    end do
                    diis_b(diis_size + 1, diis_size + 1) = 0.0d0

                    diis_rhs = 0.0d0
                    diis_rhs(diis_size + 1) = -1.0d0

                    call solve_linear_system(diis_b, diis_rhs, diis_size + 1, diis_coeff)

                    t1_real = 0.0d0
                    t2_real = 0.0d0
                    t3_real = 0.0d0
                    do diis_k = 1, diis_size
                        t1_real = t1_real + diis_coeff(diis_k) * diis_t1(:,:,diis_order(diis_k))
                        t2_real = t2_real + diis_coeff(diis_k) * diis_t2(:,:,:,:,diis_order(diis_k))
                        t3_real = t3_real + diis_coeff(diis_k) * diis_t3(:,:,:,:,:,:,diis_order(diis_k))
                    end do

                    deallocate(diis_b, diis_rhs, diis_coeff)
                else
                    t1_real = t1_new
                    t2_real = t2_new
                    t3_real = t3_new
                end if
            else
                t1_real = t1_new
                t2_real = t2_new
                t3_real = t3_new
            end if
            
            ! T3 updated via t3_new / DIIS above
            
            t1_cmplx = cmplx(t1_real, 0.0d0, kind=8)
            t2_cmplx = cmplx(t2_real, 0.0d0, kind=8)
            t3_cmplx = cmplx(t3_real, 0.0d0, kind=8)
            call copy_to_talsh_2d(T1, t1_cmplx, nvir, nocc)
            call copy_to_talsh_4d(T2, t2_cmplx, nvir, nvir, nocc, nocc)
            call copy_to_talsh_6d(T3, t3_cmplx, nvir, nvir, nvir, nocc, nocc, nocc)
        end do
        
        if (.not. converged) then
            write(*,*)
            write(*,*) 'Warning: CCSDT did not converge within maximum iterations!'
        end if
        
        write(*,*)
        write(*,'(A,F18.10,A)') ' Final CCSDT correlation energy:', e_corr, ' MeV'
        write(*,'(A,F18.10,A)') ' Total energy (E0 + E_corr):', no_ham%E0 + e_corr, ' MeV'
        write(*,*)
        
        ierr = talsh_tensor_destruct(F1_oo)
        ierr = talsh_tensor_destruct(F2_vo)
        ierr = talsh_tensor_destruct(F3_ov)
        ierr = talsh_tensor_destruct(F4_vv)
        ierr = talsh_tensor_destruct(V1_oooo)
        ierr = talsh_tensor_destruct(V2_vooo)
        ierr = talsh_tensor_destruct(V3_vvoo)
        ierr = talsh_tensor_destruct(V4_ooov)
        ierr = talsh_tensor_destruct(V5_voov)
        ierr = talsh_tensor_destruct(V6_vvov)
        ierr = talsh_tensor_destruct(V7_oovv)
        ierr = talsh_tensor_destruct(V8_vovv)
        ierr = talsh_tensor_destruct(V9_vvvv)
        ierr = talsh_tensor_destruct(T1)
        ierr = talsh_tensor_destruct(T2)
        ierr = talsh_tensor_destruct(T3)
        ierr = talsh_tensor_destruct(Z0)
        ierr = talsh_tensor_destruct(Z1)
        ierr = talsh_tensor_destruct(Z2)
        ierr = talsh_tensor_destruct(Z3)
        
        ! Store T1, T2, T3 for use by CCSDTQ
        call store_ccsdt_amplitudes(t1_real, t2_real, t3_real)
        
        deallocate(eps_o, eps_v, t1_real, r1_real, t2_real, r2_real, t3_real, r3_real, t1_new, t2_new, t3_new)
        deallocate(diis_t1, diis_t2, diis_t3, diis_err1, diis_err2, diis_err3, diis_order)
        deallocate(t1_cmplx, r1_cmplx, t2_cmplx, r2_cmplx, t3_cmplx, r3_cmplx, z0_cmplx)
        deallocate(F_oo, F_vo, F_ov, F_vv, F_oo_cmplx, F_vo_cmplx, F_ov_cmplx, F_vv_cmplx)
        deallocate(V_oooo, V_vooo, V_vvoo, V_ooov, V_voov, V_vvov, V_oovv, V_vovv, V_vvvv)
        deallocate(V_oooo_cmplx, V_vooo_cmplx, V_vvoo_cmplx, V_ooov_cmplx)
        deallocate(V_voov_cmplx, V_vvov_cmplx, V_oovv_cmplx, V_vovv_cmplx, V_vvvv_cmplx)
        
    end subroutine ccsdt_solver

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
    
    subroutine copy_to_talsh_6d(tens, data, n1, n2, n3, n4, n5, n6)
        type(talsh_tens_t), intent(inout) :: tens
        complex(8), intent(in) :: data(n1, n2, n3, n4, n5, n6)
        integer, intent(in) :: n1, n2, n3, n4, n5, n6
        type(C_PTR) :: body_p
        complex(8), pointer :: body(:)
        integer :: ierr, vol, i1, i2, i3, i4, i5, i6, idx
        
        ierr = talsh_tensor_get_body_access(tens, body_p, C8, 0, DEV_HOST)
        vol = talsh_tensor_volume(tens)
        call c_f_pointer(body_p, body, (/vol/))
        
        idx = 1
        do i6 = 1, n6
            do i5 = 1, n5
                do i4 = 1, n4
                    do i3 = 1, n3
                        do i2 = 1, n2
                            do i1 = 1, n1
                                body(idx) = data(i1, i2, i3, i4, i5, i6)
                                idx = idx + 1
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end subroutine copy_to_talsh_6d
    
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
    
    subroutine copy_from_talsh_2d(tens, data, n1, n2)
        type(talsh_tens_t), intent(inout) :: tens
        complex(8), intent(out) :: data(n1, n2)
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
                data(i, j) = body(idx)
                idx = idx + 1
            end do
        end do
    end subroutine copy_from_talsh_2d
    
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
    
    subroutine copy_from_talsh_6d(tens, data, n1, n2, n3, n4, n5, n6)
        type(talsh_tens_t), intent(inout) :: tens
        complex(8), intent(out) :: data(n1, n2, n3, n4, n5, n6)
        integer, intent(in) :: n1, n2, n3, n4, n5, n6
        type(C_PTR) :: body_p
        complex(8), pointer :: body(:)
        integer :: ierr, vol, i1, i2, i3, i4, i5, i6, idx
        
        ierr = talsh_tensor_get_body_access(tens, body_p, C8, 0, DEV_HOST)
        vol = talsh_tensor_volume(tens)
        call c_f_pointer(body_p, body, (/vol/))
        
        idx = 1
        do i6 = 1, n6
            do i5 = 1, n5
                do i4 = 1, n4
                    do i3 = 1, n3
                        do i2 = 1, n2
                            do i1 = 1, n1
                                data(i1, i2, i3, i4, i5, i6) = body(idx)
                                idx = idx + 1
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end subroutine copy_from_talsh_6d

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

end module ccsdt_solver_module
