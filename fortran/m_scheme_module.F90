module m_scheme_module
    use snt_io_module
    use clebsch_gordan
    implicit none
    
contains

    ! Generate m-scheme basis from j-coupled basis
    subroutine generate_m_scheme(j_orbits, m_orbits)
        type(single_particle_orbits), intent(in) :: j_orbits
        type(single_particle_orbits), intent(out) :: m_orbits
        integer :: i, idx, two_j, two_jz, n_p_m, n_n_m
        integer :: jz_min, jz_max, step, total_m
        integer, allocatable :: p_indices(:), n_indices(:)
        integer :: n_p, n_n, ip, in
        
        ! Count proton and neutron orbits in j-scheme
        n_p = 0
        n_n = 0
        do i = 1, j_orbits%total_orbits
            if (j_orbits%tz(i) < 0) n_p = n_p + 1
            if (j_orbits%tz(i) > 0) n_n = n_n + 1
        end do
        
        allocate(p_indices(n_p))
        allocate(n_indices(n_n))
        
        ip = 0
        in = 0
        do i = 1, j_orbits%total_orbits
            if (j_orbits%tz(i) < 0) then
                ip = ip + 1
                p_indices(ip) = i
            else if (j_orbits%tz(i) > 0) then
                in = in + 1
                n_indices(in) = i
            end if
        end do
        
        ! Calculate m-scheme orbit counts
        n_p_m = 0
        do ip = 1, n_p
            i = p_indices(ip)
            n_p_m = n_p_m + (j_orbits%j(i) + 1)  ! 2j+1 states
        end do
        
        n_n_m = 0
        do in = 1, n_n
            i = n_indices(in)
            n_n_m = n_n_m + (j_orbits%j(i) + 1)
        end do
        
        total_m = n_p_m + n_n_m
        
        call init_sp_orbits(m_orbits, j_orbits%n_p_core, j_orbits%n_n_core, n_p_m, n_n_m)
        
        ! Expand each j-orbit into m-substates
        idx = 0
        do i = 1, j_orbits%total_orbits
            two_j = j_orbits%j(i)
            
            ! Generate 2jz values sorted by absolute value (matching Python)
            ! For half-integer j: -1, 1, -3, 3, ...
            ! For integer j: 0, -2, 2, -4, 4, ...
            
            if (mod(two_j, 2) == 0) then
                ! Integer j - handle 0 first
                idx = idx + 1
                m_orbits%n(idx) = j_orbits%n(i)
                m_orbits%l(idx) = j_orbits%l(i)
                m_orbits%j(idx) = j_orbits%j(i)
                m_orbits%jz(idx) = 0
                m_orbits%tz(idx) = j_orbits%tz(i)
                m_orbits%coupling_map(idx) = i
                
                do two_jz = 2, two_j, 2
                    ! -k
                    idx = idx + 1
                    m_orbits%n(idx) = j_orbits%n(i)
                    m_orbits%l(idx) = j_orbits%l(i)
                    m_orbits%j(idx) = j_orbits%j(i)
                    m_orbits%jz(idx) = -two_jz
                    m_orbits%tz(idx) = j_orbits%tz(i)
                    m_orbits%coupling_map(idx) = i
                    
                    ! +k
                    idx = idx + 1
                    m_orbits%n(idx) = j_orbits%n(i)
                    m_orbits%l(idx) = j_orbits%l(i)
                    m_orbits%j(idx) = j_orbits%j(i)
                    m_orbits%jz(idx) = two_jz
                    m_orbits%tz(idx) = j_orbits%tz(i)
                    m_orbits%coupling_map(idx) = i
                end do
            else
                ! Half-integer j
                do two_jz = 1, two_j, 2
                     ! -k
                    idx = idx + 1
                    m_orbits%n(idx) = j_orbits%n(i)
                    m_orbits%l(idx) = j_orbits%l(i)
                    m_orbits%j(idx) = j_orbits%j(i)
                    m_orbits%jz(idx) = -two_jz
                    m_orbits%tz(idx) = j_orbits%tz(i)
                    m_orbits%coupling_map(idx) = i
                    
                    ! +k
                    idx = idx + 1
                    m_orbits%n(idx) = j_orbits%n(i)
                    m_orbits%l(idx) = j_orbits%l(i)
                    m_orbits%j(idx) = j_orbits%j(i)
                    m_orbits%jz(idx) = two_jz
                    m_orbits%tz(idx) = j_orbits%tz(i)
                    m_orbits%coupling_map(idx) = i
                end do
            end if
        end do
        
        deallocate(p_indices)
        deallocate(n_indices)
        
        write(*,'(A,I4,A)') ' Generated ', total_m, ' m-scheme states'
        
    end subroutine generate_m_scheme
    
    ! Get j-scheme index from m-scheme index
    function get_j_index(m_orbits, m_idx) result(j_idx)
        type(single_particle_orbits), intent(in) :: m_orbits
        integer, intent(in) :: m_idx
        integer :: j_idx
        j_idx = m_orbits%coupling_map(m_idx)
    end function get_j_index
    
    ! Decouple 1-body interaction to m-scheme
    subroutine decouple_1b(j_pot, m_orbits, m_v1b)
        type(potential), intent(in) :: j_pot
        type(single_particle_orbits), intent(in) :: m_orbits
        real(8), allocatable, intent(out) :: m_v1b(:,:)
        integer :: n_m, i, j, r, s
        
        n_m = m_orbits%total_orbits
        allocate(m_v1b(n_m, n_m))
        m_v1b = 0.0d0
        
        do i = 1, n_m
            r = m_orbits%coupling_map(i)
            do j = 1, n_m
                s = m_orbits%coupling_map(j)
                
                ! Conservation laws for 1-body operator
                if (m_orbits%l(i) == m_orbits%l(j) .and. &
                    m_orbits%j(i) == m_orbits%j(j) .and. &
                    m_orbits%tz(i) == m_orbits%tz(j) .and. &
                    m_orbits%jz(i) == m_orbits%jz(j)) then
                    m_v1b(i, j) = j_pot%v1b(r, s)
                end if
            end do
        end do
        
        write(*,'(A,I6)') ' Decoupled 1-body interaction, non-zero elements: ', count(abs(m_v1b) > 1.0d-15)
        
    end subroutine decouple_1b
    
    ! Decouple 2-body interaction to m-scheme (dense 4D array)
    ! Based on Python read_snt_io.py::decouple_2b
    subroutine decouple_2b(j_pot, m_orbits, m_v2b)
        type(potential), intent(in) :: j_pot
        type(single_particle_orbits), intent(in) :: m_orbits
        real(8), allocatable, intent(out) :: m_v2b(:,:,:,:)
        integer :: n_m, i_tbme
        integer :: r, s, t, u, J_val, two_J, two_M
        integer :: a_idx, b_idx, c_idx, d_idx
        integer :: two_ma, two_mb, two_mc, two_md
        real(8) :: val, nas_factor, cg_ab, cg_cd, v_val
        integer, allocatable :: m_a_indices(:), m_b_indices(:), m_c_indices(:), m_d_indices(:)
        integer :: n_ma, n_mb, n_mc, n_md, ia, ib, ic, id
        
        n_m = m_orbits%total_orbits
        allocate(m_v2b(n_m, n_m, n_m, n_m))
        m_v2b = 0.0d0
        
        write(*,'(A,I6,A)') ' Decoupling ', j_pot%n_tbme, ' TBMEs to m-scheme...'
        
        ! Loop over all J-scheme TBMEs
        do i_tbme = 1, j_pot%n_tbme
            r = j_pot%v2b(i_tbme)%idx_i
            s = j_pot%v2b(i_tbme)%idx_j
            t = j_pot%v2b(i_tbme)%idx_k  
            u = j_pot%v2b(i_tbme)%idx_l
            J_val = j_pot%v2b(i_tbme)%J_val
            val = j_pot%v2b(i_tbme)%val_me
            
            if (abs(val) < 1.0d-15) cycle
            
            two_J = 2 * J_val
            
            ! Get m-scheme indices for each j-scheme orbit (add 1 for Fortran indexing)
            call get_m_indices_array(m_orbits, r+1, m_a_indices, n_ma)
            call get_m_indices_array(m_orbits, s+1, m_b_indices, n_mb)
            call get_m_indices_array(m_orbits, t+1, m_c_indices, n_mc)
            call get_m_indices_array(m_orbits, u+1, m_d_indices, n_md)
            
            ! NAS normalization factor
            nas_factor = 1.0d0
            if (r == s) nas_factor = nas_factor * sqrt(2.0d0)
            if (t == u) nas_factor = nas_factor * sqrt(2.0d0)
            
            ! Loop over M projections
            do two_M = -two_J, two_J, 2
                ! Loop over (a,b) combinations
                do ia = 1, n_ma
                    a_idx = m_a_indices(ia)
                    two_ma = m_orbits%jz(a_idx)
                    
                    do ib = 1, n_mb
                        b_idx = m_b_indices(ib)
                        two_mb = m_orbits%jz(b_idx)
                        
                        if (two_ma + two_mb /= two_M) cycle
                        
                        cg_ab = get_cg_cached(m_orbits%j(a_idx), two_ma, &
                                             m_orbits%j(b_idx), two_mb, two_J, two_M)
                        if (abs(cg_ab) < 1.0d-10) cycle
                        
                        ! Loop over (c,d) combinations  
                        do ic = 1, n_mc
                            c_idx = m_c_indices(ic)
                            two_mc = m_orbits%jz(c_idx)
                            
                            do id = 1, n_md
                                d_idx = m_d_indices(id)
                                two_md = m_orbits%jz(d_idx)
                                
                                if (two_mc + two_md /= two_M) cycle
                                
                                cg_cd = get_cg_cached(m_orbits%j(c_idx), two_mc, &
                                                     m_orbits%j(d_idx), two_md, two_J, two_M)
                                if (abs(cg_cd) < 1.0d-10) cycle
                                
                                v_val = cg_ab * cg_cd * val * nas_factor
                                
                                ! Antisymmetrize: <ab||cd> = <ab|cd> - <ab|dc>
                                m_v2b(a_idx, b_idx, c_idx, d_idx) = m_v2b(a_idx, b_idx, c_idx, d_idx) + v_val
                                if (r /= s) m_v2b(b_idx, a_idx, c_idx, d_idx) = m_v2b(b_idx, a_idx, c_idx, d_idx) - v_val
                                if (t /= u) m_v2b(a_idx, b_idx, d_idx, c_idx) = m_v2b(a_idx, b_idx, d_idx, c_idx) - v_val
                                if (r /= s .and. t /= u) m_v2b(b_idx, a_idx, d_idx, c_idx) = m_v2b(b_idx, a_idx, d_idx, c_idx) + v_val
                                
                                ! Hermiticity (if bra != ket)
                                if (r /= t .or. s /= u) then
                                    m_v2b(c_idx, d_idx, a_idx, b_idx) = m_v2b(c_idx, d_idx, a_idx, b_idx) + v_val
                                    if (r /= s) m_v2b(c_idx, d_idx, b_idx, a_idx) = m_v2b(c_idx, d_idx, b_idx, a_idx) - v_val
                                    if (t /= u) m_v2b(d_idx, c_idx, a_idx, b_idx) = m_v2b(d_idx, c_idx, a_idx, b_idx) - v_val
                                    if (r /= s .and. t /= u) m_v2b(d_idx, c_idx, b_idx, a_idx) = m_v2b(d_idx, c_idx, b_idx, a_idx) + v_val
                                end if
                            end do
                        end do
                    end do
                end do
            end do
            
            deallocate(m_a_indices, m_b_indices, m_c_indices, m_d_indices)
        end do
        
        write(*,'(A,I10)') ' Decoupled 2-body interaction, non-zero elements: ', count(abs(m_v2b) > 1.0d-15)
        
    end subroutine decouple_2b
    
    ! Helper to get m-scheme indices for a given j-scheme index
    subroutine get_m_indices_array(m_orbits, j_idx, m_indices, n_m)
        type(single_particle_orbits), intent(in) :: m_orbits
        integer, intent(in) :: j_idx
        integer, allocatable, intent(out) :: m_indices(:)
        integer, intent(out) :: n_m
        integer :: i, count_m
        
        ! Count how many m-scheme states correspond to this j-scheme orbit
        count_m = 0
        do i = 1, m_orbits%total_orbits
            if (m_orbits%coupling_map(i) == j_idx) count_m = count_m + 1
        end do
        
        n_m = count_m
        allocate(m_indices(n_m))
        
        ! Collect the indices
        count_m = 0
        do i = 1, m_orbits%total_orbits
            if (m_orbits%coupling_map(i) == j_idx) then
                count_m = count_m + 1
                m_indices(count_m) = i
            end if
        end do
    end subroutine get_m_indices_array

end module m_scheme_module
