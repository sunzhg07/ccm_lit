module hartree_fock_module
    use talsh
    use tensor_algebra
    use, intrinsic:: ISO_C_BINDING
    use snt_io_module
    use m_scheme_module
    implicit none
    
    ! Normal-ordered Hamiltonian structure
    type :: normal_ordered_hamiltonian
        real(8) :: E0                          ! Zero-body term (reference energy)
        real(8), allocatable :: f_hf(:,:)      ! One-body term (Fock matrix in HF basis)
        real(8), allocatable :: Gamma_hf(:,:,:,:)  ! Two-body term in HF basis
        integer :: n_states                     ! Number of single-particle states
        integer :: nocc                         ! Number of occupied states
        integer :: nvir                         ! Number of virtual states
    end type normal_ordered_hamiltonian
    
    ! Symmetry sector type for block diagonalization
    type :: symmetry_sector
        integer :: parity      ! l mod 2
        integer :: j           ! 2*j (Total angular momentum)
        integer :: jz          ! 2*m_z
        integer :: tz          ! 2*t_z
        integer :: n_states    ! number of states in this sector
        integer :: n_occ       ! number to occupy in this sector
        integer, allocatable :: indices(:)  ! state indices in this sector
        real(8), allocatable :: eigvals(:)          ! Eigenvalues for this sector
        real(8), allocatable :: eigvecs(:,:)        ! Eigenvectors for this sector (local basis)
    end type symmetry_sector
    
contains

    ! Helper to sort eigenvalues and track indices
    subroutine sort_eigenvalues(val, idx, n)
        real(8), intent(inout) :: val(:)
        integer, intent(inout) :: idx(:)
        integer, intent(in) :: n
        integer :: i, k
        real(8) :: tmp_val
        integer :: tmp_idx
        
        do i = 1, n - 1
            do k = i + 1, n
                if (val(k) < val(i)) then
                    tmp_val = val(i)
                    val(i) = val(k)
                    val(k) = tmp_val
                    
                    tmp_idx = idx(i)
                    idx(i) = idx(k)
                    idx(k) = tmp_idx
                end if
            end do
        end do
    end subroutine

    ! Hartree-Fock solver with symmetry-preserving block diagonalization
    ! Matches Python implementation
    subroutine hartree_fock_talsh(m_orbits, m_v1b, m_v2b, n_p_val, n_n_val, &
                                  occ_indices, n_occ_in, hf_energy, sp_energies, rho, sp_coeffs, mode_in, &
                                  allow_sector_reoccupy_in)
        type(single_particle_orbits), intent(in) :: m_orbits
        real(8), intent(in) :: m_v1b(:,:)
        real(8), intent(in) :: m_v2b(:,:,:,:)  ! 4D m-scheme interaction
        integer, intent(in) :: n_p_val, n_n_val
        integer, intent(in) :: occ_indices(:)
        integer, intent(in) :: n_occ_in
        real(8), intent(out) :: hf_energy
        real(8), allocatable, intent(out) :: sp_energies(:)
        real(8), allocatable, intent(out) :: rho(:,:)
        real(8), allocatable, intent(out) :: sp_coeffs(:,:)
        character(len=*), intent(in), optional :: mode_in
        logical, intent(in), optional :: allow_sector_reoccupy_in
        
        character(len=20) :: mode
        logical :: allow_sector_reoccupy
        integer :: n_states, n_occ, i, j, a, b, c, d, iter, isec, k, idx, idx_a, idx_b
        real(8), allocatable :: fock(:,:), new_rho(:,:)
        real(8), allocatable :: eigvec(:,:), eigval(:)
        real(8) :: energy, old_energy, delta_e, e1b, e2b
        integer :: max_iter
        real(8) :: tol
        
        ! Symmetry sector variables
        type(symmetry_sector), allocatable :: sectors(:)
        integer :: n_sectors
        integer, allocatable :: state_to_sector(:)
        integer, allocatable :: temp_indices(:)
        integer :: parity, jz, tz, two_j
        logical :: found, match
        
        ! Block diagonalization variables
        real(8), allocatable :: sub_fock(:,:), sub_eigval(:), sub_eigvec(:,:)
        real(8), allocatable :: full_vec(:)
        integer :: sub_n
        
        ! Sorting variables
        real(8), allocatable :: p_evals(:), n_evals(:)
        integer, allocatable :: p_map(:,:), n_map(:,:) ! (2, n) : 1=sector, 2=index
        integer :: n_p_total, n_n_total, p_count, n_count
        
        if (present(mode_in)) then
            mode = mode_in
        else
            mode = 'deformed'  ! Default to deformed to match Python
        end if

        if (present(allow_sector_reoccupy_in)) then
            allow_sector_reoccupy = allow_sector_reoccupy_in
        else
            allow_sector_reoccupy = .true.
        end if
        
        n_states = m_orbits%total_orbits
        n_occ = n_occ_in
        max_iter = 100
        tol = 1.0d-8
        
        allocate(rho(n_states, n_states))
        allocate(new_rho(n_states, n_states))
        allocate(fock(n_states, n_states))
        allocate(eigvec(n_states, n_states))
        allocate(eigval(n_states))
        allocate(sp_energies(n_states))
        allocate(sp_coeffs(n_states, n_states))
        allocate(state_to_sector(n_states))
        allocate(temp_indices(n_states))
        
        ! Sorting arrays
        allocate(p_evals(n_states), n_evals(n_states))
        allocate(p_map(2, n_states), n_map(2, n_states))
        
        ! --- Identify symmetry sectors ---
        if (trim(mode) == 'deformed') then
            write(*,*) '--- Identifying Symmetry Sectors (Deformed: parity, jz, tz) ---'
        else
            write(*,*) '--- Identifying Symmetry Sectors (Spherical: parity, j, jz, tz) ---'
        end if
        
        ! First pass: count unique sectors
        n_sectors = 0
        state_to_sector = 0
        
        do i = 1, n_states
            parity = mod(m_orbits%l(i), 2)
            two_j = m_orbits%j(i)
            jz = m_orbits%jz(i)
            tz = m_orbits%tz(i)
            
            found = .false.
            do isec = 1, n_sectors
                ! Check matching based on mode
                if (trim(mode) == 'deformed') then
                    ! Deformed: match (parity, jz, tz) only - allow different j to mix
                    match = (sectors(isec)%parity == parity) .and. &
                            (sectors(isec)%jz == jz) .and. &
                            (sectors(isec)%tz == tz)
                else
                    ! Spherical: match (parity, j, jz, tz) - j is conserved
                    match = (sectors(isec)%parity == parity) .and. &
                            (sectors(isec)%j == two_j) .and. &
                            (sectors(isec)%jz == jz) .and. &
                            (sectors(isec)%tz == tz)
                end if
                
                if (match) then
                    found = .true.
                    state_to_sector(i) = isec
                    exit
                end if
            end do
            
            if (.not. found) then
                n_sectors = n_sectors + 1
                if (n_sectors == 1) then
                    allocate(sectors(1))
                else
                    ! Reallocate sectors array
                    block
                        type(symmetry_sector), allocatable :: temp_sectors(:)
                        allocate(temp_sectors(n_sectors))
                        temp_sectors(1:n_sectors-1) = sectors(1:n_sectors-1)
                        call move_alloc(temp_sectors, sectors)
                    end block
                end if
                sectors(n_sectors)%parity = parity
                sectors(n_sectors)%j = two_j  ! Always store j
                sectors(n_sectors)%jz = jz
                sectors(n_sectors)%tz = tz
                sectors(n_sectors)%n_states = 0
                sectors(n_sectors)%n_occ = 0
                state_to_sector(i) = n_sectors
            end if
        end do
        
        ! Second pass: fill indices
        do isec = 1, n_sectors
            ! Count states
            sectors(isec)%n_states = 0
            do i = 1, n_states
                if (state_to_sector(i) == isec) then
                    sectors(isec)%n_states = sectors(isec)%n_states + 1
                end if
            end do
            allocate(sectors(isec)%indices(sectors(isec)%n_states))
            allocate(sectors(isec)%eigvals(sectors(isec)%n_states))
            allocate(sectors(isec)%eigvecs(sectors(isec)%n_states, sectors(isec)%n_states))
             
            ! Fill indices
            k = 0
            do i = 1, n_states
                if (state_to_sector(i) == isec) then
                    k = k + 1
                    sectors(isec)%indices(k) = i
                end if
            end do
        end do
        
        write(*,*) 'Found ', n_sectors, ' symmetry sectors'
        ! Debug print sectors
        do isec = 1, n_sectors
            write(*,'(A,I4,A,I2,A,I3,A,I3,A,I4,A,I4,A)') &
             '  Sector', isec, ': parity=', sectors(isec)%parity, &
             ' jz=', sectors(isec)%jz, ' tz=', sectors(isec)%tz, &
             ' | ', sectors(isec)%n_states, ' states'
        end do
        write(*,*)

        ! Initial Density Guess
        ! In deformed mode: regenerate occupation from energy sorting to allow j mixing
        ! In spherical mode: use provided occ_indices directly
        if (trim(mode) == 'deformed') then
            ! Deformed HF: sort ALL single-particle states by energy
            ! (This breaks j conservation and allows mixing)
            block
                real(8), allocatable :: p_evals_local(:), n_evals_local(:)
                integer, allocatable :: p_idx_local(:), n_idx_local(:)
                integer :: p_cnt, n_cnt, i_sort
                
                allocate(p_evals_local(n_states), n_evals_local(n_states))
                allocate(p_idx_local(n_states), n_idx_local(n_states))
                
                p_cnt = 0
                n_cnt = 0
                do i = 1, n_states
                    if (m_orbits%tz(i) < 0) then ! proton
                        p_cnt = p_cnt + 1
                        p_evals_local(p_cnt) = m_v1b(i, i)
                        p_idx_local(p_cnt) = i
                    else ! neutron
                        n_cnt = n_cnt + 1
                        n_evals_local(n_cnt) = m_v1b(i, i)
                        n_idx_local(n_cnt) = i
                    end if
                end do
                
                ! Sort energies and keep track of indices
                call sort_eigenvalues(p_evals_local(1:p_cnt), p_idx_local(1:p_cnt), p_cnt)
                call sort_eigenvalues(n_evals_local(1:n_cnt), n_idx_local(1:n_cnt), n_cnt)
                
                ! Set occupation based on lowest energies
                rho = 0.0d0
                do i = 1, n_p_val
                    idx = p_idx_local(i)
                    rho(idx, idx) = 1.0d0
                end do
                do i = 1, n_n_val
                    idx = n_idx_local(i)
                    rho(idx, idx) = 1.0d0
                end do
                
                write(*,*) 'Deformed HF: regenerated initial occupation from energy sorting'
                deallocate(p_evals_local, n_evals_local, p_idx_local, n_idx_local)
            end block
        else
            ! Spherical HF: use provided occ_indices directly
            rho = 0.0d0
            do i = 1, n_occ
                idx = occ_indices(i)
                rho(idx, idx) = 1.0d0
            end do
        end if

        ! Initialize fixed sector occupations from the initial density
        do isec = 1, n_sectors
            sectors(isec)%n_occ = 0
            do i = 1, sectors(isec)%n_states
                idx = sectors(isec)%indices(i)
                if (rho(idx, idx) > 0.5d0) then
                    sectors(isec)%n_occ = sectors(isec)%n_occ + 1
                end if
            end do
        end do
        
        old_energy = 0.0d0
        
        write(*,*) '--- Starting Hartree-Fock Iteration (Block Diagonalization) ---'
        write(*,'(A,A,A)') 'Iter', '     Total Energy', '       Delta E'
        write(*,'(A)') '--------------------------------------'
        
        ! HF Iteration Loop
        do iter = 1, max_iter
            ! Build Fock matrix (using Talsh logic equivalent)
            ! F_ab = t_ab + sum_cd <ac|V|bd> rho_dc
            fock = m_v1b ! Start with 1-body part
            
            ! Add 2-body mean field
            ! Assuming fock is initially 1-body
            ! Loop over all basis states
            ! Optimized using m_v2b limits
            do a = 1, n_states
                do b = 1, n_states
                    ! Add sum_cd m_v2b(a,c,b,d) * rho(d,c)
                    ! Note: indices of m_v2b are (a,b,c,d) -> <ab|V|cd>
                    ! We want <ac|V|bd>.
                    ! rho(d,c) is density matrix element
                    
                    ! Only compute if needed (block diagonal eventually)
                    ! But initially we compute all to check symmetry
                    
                    do c = 1, n_states
                        do d = 1, n_states
                            if (abs(rho(d,c)) > 1.0d-10) then
                                ! <ac|V|bd> correspond to m_v2b(a,c,b,d)
                                fock(a,b) = fock(a,b) + rho(d,c) * m_v2b(a,c,b,d)
                            end if
                        end do
                    end do
                end do
            end do
            
            ! Block Diagonalization
            ! Loop over sectors
            
            ! Reset sorting counters
            p_count = 0
            n_count = 0
            
            do isec = 1, n_sectors
                sub_n = sectors(isec)%n_states
                if (sub_n == 0) cycle
                allocate(sub_fock(sub_n, sub_n))
                
                ! Extract sub-block
                do i = 1, sub_n
                    do j = 1, sub_n
                        sub_fock(i,j) = fock(sectors(isec)%indices(i), sectors(isec)%indices(j))
                    end do
                end do
                
                ! Diagonalize
                call diagonalize_symmetric(sub_fock, sub_n, sectors(isec)%eigvals, sectors(isec)%eigvecs)
                
                deallocate(sub_fock)
                
                ! Collect eigenvalues for global sorting
                do i = 1, sub_n
                    if (sectors(isec)%tz < 0) then ! Proton (tz=-1)
                        p_count = p_count + 1
                        p_evals(p_count) = sectors(isec)%eigvals(i)
                        p_map(1, p_count) = isec
                        p_map(2, p_count) = i
                    else ! Neutron (tz=1)
                        n_count = n_count + 1
                        n_evals(n_count) = sectors(isec)%eigvals(i)
                        n_map(1, n_count) = isec
                        n_map(2, n_count) = i
                    end if
                    
                    ! Also store in global array (mapped)
                    idx = sectors(isec)%indices(i)
                    ! Note: Eigenvalues are associated with transformed states, not basis states directly
                    ! But we need sp_energies for output
                    ! We map them later after occupation decision
                end do
            end do
            
            ! --- Update Occupation (Global Sorting) ---
            if (allow_sector_reoccupy) then
                ! Sort Protons
                ! We use a simple index Array to track
                block
                    integer, allocatable :: p_idx(:), n_idx(:)
                    allocate(p_idx(p_count))
                    allocate(n_idx(n_count))
                    
                    do i = 1, p_count
                        p_idx(i) = i
                    end do
                    do i = 1, n_count
                        n_idx(i) = i
                    end do
                    
                    call sort_eigenvalues(p_evals(1:p_count), p_idx, p_count)
                    call sort_eigenvalues(n_evals(1:n_count), n_idx, n_count)
                    
                    ! Determine new occupation counts per sector
                    do isec = 1, n_sectors
                        sectors(isec)%n_occ = 0
                    end do
                    
                    ! Count top n_p_val protons
                    do i = 1, n_p_val
                        ! Original index in unsorted list was p_idx(i)
                        ! p_map(:, p_idx(i)) tells us (sector, index)
                        isec = p_map(1, p_idx(i))
                        sectors(isec)%n_occ = sectors(isec)%n_occ + 1
                    end do
                    
                    ! Count top n_n_val neutrons
                    do i = 1, n_n_val
                        isec = n_map(1, n_idx(i))
                        sectors(isec)%n_occ = sectors(isec)%n_occ + 1
                    end do
                    
                    deallocate(p_idx, n_idx)
                end block
            end if
            
            ! Construct New Density Matrix and global info
            new_rho = 0.0d0
            sp_coeffs = 0.0d0 ! Global transformation matrix
            
            do isec = 1, n_sectors
                sub_n = sectors(isec)%n_states
                
                ! Transform sub_eigvecs to full basis
                ! sp_coeffs column k corresponds to eigenvector k
                ! The k-th eigenvector of sector isec belongs to a specific global index?
                ! We simply fill sp_coeffs block-wise.
                ! Let's assign global indices to the eigenvectors sequentially or based on basis?
                ! It is better to just store them block-wise in sp_coeffs
                ! Map sector internal index to global basis index
                
                do k = 1, sub_n
                    ! Global index of the k-th eigenvector in this sector?
                    ! Ideally we sort them by energy globally?
                    ! But the code expects sp_energies(i) to match sp_coeffs(:, i)
                    ! And usually we keep basis ordering loosely?
                    ! Let's validly map: 
                    ! For density construction:
                    ! If k <= sectors(isec)%n_occ, it is occupied.
                    
                    if (k <= sectors(isec)%n_occ) then
                         ! Add |v><v| to rho
                         do a = 1, sub_n
                             idx_a = sectors(isec)%indices(a)
                             do b = 1, sub_n
                                 idx_b = sectors(isec)%indices(b)
                                 new_rho(idx_a, idx_b) = new_rho(idx_a, idx_b) + &
                                     sectors(isec)%eigvecs(a, k) * sectors(isec)%eigvecs(b, k)
                             end do
                         end do
                    end if
                end do
                 
                 ! Can also fill sp_energies and sp_coeffs for output
                 ! This usually requires mapping back to a 'diagonal' basis.
                 ! We can just leave sp_energies unsorted or sorted per sector.
                 do k = 1, sub_n
                     ! Map to global index matching the basis index 'sectors(isec)%indices(k)'?
                     ! No, that's assuming diagonal.
                     ! Let's just place them into sp_energies at the indices corresponding to the sector
                     ! This is arbitrary but valid storage
                     idx = sectors(isec)%indices(k) 
                     sp_energies(idx) = sectors(isec)%eigvals(k)
                     
                     ! Store eigenvector in sp_coeffs
                     ! Col 'idx' of sp_coeffs = k-th eigenvector
                     ! Row 'r' = coefficient for basis state r
                     do i = 1, sub_n
                         sp_coeffs(sectors(isec)%indices(i), idx) = sectors(isec)%eigvecs(i, k)
                     end do
                 end do
            end do
            
            ! Calculate Energy
            e1b = sum(new_rho * m_v1b)
            e2b = 0.5d0 * sum(new_rho * (fock - m_v1b))
            energy = e1b + e2b
            
            delta_e = abs(energy - old_energy)
            
            write(*,'(I4,F20.8,ES12.4,A,F10.4,A,F10.4,A)') iter, energy, delta_e, &
                 '  (E1B=', e1b, ' E2B=', e2b, ')'
            
            if (delta_e < tol .and. iter > 1) then
                write(*,*) 'Converged!'
                exit
            end if
            
            old_energy = energy
            rho = new_rho
        end do
        
        hf_energy = energy
        write(*,*)
        write(*,'(A,F20.8,A)') ' Final HF Energy: ', hf_energy, ' MeV'
        
        ! --- CRITICAL: Reorder orbitals to match Python convention ---
        ! Python returns: occupied first (sorted by energy), then virtuals (sorted by energy)
        ! IMPORTANT: Use sector occupations, NOT density matrix diagonal (which can be fractional)
        block
            real(8), allocatable :: sp_energies_reord(:), sp_coeffs_reord(:,:)
            real(8), allocatable :: occupied_e(:), virtual_e(:)
            real(8), allocatable :: occupied_c(:,:), virtual_c(:,:)
            real(8), allocatable :: e_tmp(:)
            integer, allocatable :: idx(:)
            integer :: n_occ_found, n_vir_found, i_occ, i_vir, isec, i, k, sub_n
            
            allocate(sp_energies_reord(n_states))
            allocate(sp_coeffs_reord(n_states, n_states))
            allocate(occupied_e(n_states), virtual_e(n_states))
            allocate(occupied_c(n_states, n_states), virtual_c(n_states, n_states))
            allocate(e_tmp(n_states), idx(n_states))
            
            ! Initialize arrays
            occupied_c = 0.0d0
            virtual_c = 0.0d0
            
            ! Separate occupied and virtual based on sector occupations (not density matrix!)
            ! Each sector has sectors(isec)%n_occ eigenvectors marked as occupied
            n_occ_found = 0
            n_vir_found = 0
            
            do isec = 1, n_sectors
                sub_n = sectors(isec)%n_states
                do k = 1, sub_n
                    if (k <= sectors(isec)%n_occ) then
                        ! This eigenvector was marked as occupied in this sector
                        n_occ_found = n_occ_found + 1
                        occupied_e(n_occ_found) = sectors(isec)%eigvals(k)
                        ! Reconstruct full eigenvector from sector basis
                        do i = 1, sub_n
                            occupied_c(sectors(isec)%indices(i), n_occ_found) = sectors(isec)%eigvecs(i, k)
                        end do
                    else
                        ! Virtual
                        n_vir_found = n_vir_found + 1
                        virtual_e(n_vir_found) = sectors(isec)%eigvals(k)
                        do i = 1, sub_n
                            virtual_c(sectors(isec)%indices(i), n_vir_found) = sectors(isec)%eigvecs(i, k)
                        end do
                    end if
                end do
            end do
            
            write(*,'(A,I3,A,I3)') ' DEBUG: Sector-based found ', n_occ_found, ' occupied, ', n_vir_found, ' virtual'
            
            ! Sort occupied by energy
            do i = 1, n_occ_found
                idx(i) = i
            end do
            e_tmp(1:n_occ_found) = occupied_e(1:n_occ_found)
            call sort_eigenvalues(e_tmp(1:n_occ_found), idx(1:n_occ_found), n_occ_found)
            ! Copy sorted occupied orbitals
            do i_occ = 1, n_occ_found
                sp_energies_reord(i_occ) = occupied_e(idx(i_occ))
                sp_coeffs_reord(:, i_occ) = occupied_c(:, idx(i_occ))
            end do
            
            ! Sort virtual by energy
            do i = 1, n_vir_found
                idx(i) = i
            end do
            e_tmp(1:n_vir_found) = virtual_e(1:n_vir_found)
            call sort_eigenvalues(e_tmp(1:n_vir_found), idx(1:n_vir_found), n_vir_found)
            ! Copy sorted virtual orbitals
            do i_vir = 1, n_vir_found
                sp_energies_reord(n_occ_found + i_vir) = virtual_e(idx(i_vir))
                sp_coeffs_reord(:, n_occ_found + i_vir) = virtual_c(:, idx(i_vir))
            end do
            
            ! Replace with reordered versions
            sp_energies = sp_energies_reord
            sp_coeffs = sp_coeffs_reord
            
            deallocate(sp_energies_reord, sp_coeffs_reord, occupied_e, virtual_e, occupied_c, virtual_c, e_tmp, idx)
            
            write(*,'(A)') ' Reordered: occupied first (sorted), then virtuals (sorted by energy)'
        end block
        
        write(*,*)
        
        ! Cleanup
        if (allocated(p_evals)) deallocate(p_evals, n_evals, p_map, n_map)
        
        do isec = 1, n_sectors
            if (allocated(sectors(isec)%indices)) deallocate(sectors(isec)%indices)
            if (allocated(sectors(isec)%eigvals)) deallocate(sectors(isec)%eigvals)
            if (allocated(sectors(isec)%eigvecs)) deallocate(sectors(isec)%eigvecs)
        end do
        if (allocated(sectors)) deallocate(sectors)
        
    end subroutine hartree_fock_talsh
    
    ! Diagonalize symmetric matrix (use LAPACK)
    subroutine diagonalize_symmetric(matrix, n, eigenvalues, eigenvectors)
        integer, intent(in) :: n
        real(8), intent(in) :: matrix(n,n)
        real(8), intent(out) :: eigenvalues(n)
        real(8), intent(out) :: eigenvectors(n,n)
        
        real(8) :: work_array(3*n)
        integer :: info
        
        eigenvectors = matrix
        
        ! Call LAPACK dsyev to diagonalize
        call dsyev('V', 'U', n, eigenvectors, n, eigenvalues, work_array, 3*n, info)
        
        if (info /= 0) then
            write(*,*) 'Error in diagonalization, info = ', info
            stop
        end if
    end subroutine diagonalize_symmetric
    
    ! Normal order Hamiltonian (transform to HF basis)
    subroutine normal_order_talsh(m_orbits, m_v1b, m_v2b, hf_energy, &
                                  sp_energies, rho, sp_coeffs, n_occ_in, no_ham)
        type(single_particle_orbits), intent(in) :: m_orbits
        real(8), intent(in) :: m_v1b(:,:)
        real(8), intent(in) :: m_v2b(:,:,:,:)  ! 4D m-scheme interaction
        real(8), intent(in) :: hf_energy
        real(8), intent(in) :: sp_energies(:)
        real(8), intent(in) :: rho(:,:)
        real(8), intent(in) :: sp_coeffs(:,:)
        integer, intent(in) :: n_occ_in  ! Number of occupied states
        type(normal_ordered_hamiltonian), intent(out) :: no_ham
        
        integer :: n_states, i, j, p, q, r, s, a, b, c, d, nocc, nvir
        integer :: ierr
        complex(8), parameter :: ZERO = (0.0d0, 0.0d0)
        real(8), allocatable :: f_hf_real(:,:)
        real(8), allocatable :: Gamma_hf_real(:,:,:,:)
        real(8), allocatable :: temp1(:,:,:,:), temp2(:,:,:,:), temp3(:,:,:,:)
        
        n_states = m_orbits%total_orbits
        nocc = n_occ_in  ! Use the value passed from the HF routine
        nvir = n_states - nocc
        
        no_ham%E0 = hf_energy
        no_ham%n_states = n_states
        no_ham%nocc = nocc
        no_ham%nvir = nvir
        
        write(*,*) '--- Normal Ordering Hamiltonian ---'
        write(*,'(A,F16.8)') ' E0 (HF Energy): ', no_ham%E0
        write(*,'(A,I4,A,I4,A,I4)') ' States: ', n_states, ' Holes: ', nocc, ' Particles: ', nvir
        write(*,*)
        
        ! 1. Fock matrix in HF basis (should be diagonal with sp_energies)
        allocate(f_hf_real(n_states, n_states))
        f_hf_real = 0.0d0
        do i = 1, n_states
            f_hf_real(i,i) = sp_energies(i)
        end do
        
        ! 2. Transform 2-body interaction to HF basis  
        ! Gamma_abcd = sum_pqrs C_pa * C_qb * V_pqrs * C_rc * C_sd
        ! Use 4-step N^5 algorithm (matching Python normal_order)
        write(*,*) 'Transforming 2-body interaction to HF basis (N^5 optimized)...'
        
        allocate(temp1(n_states, n_states, n_states, n_states))
        allocate(temp2(n_states, n_states, n_states, n_states))
        allocate(temp3(n_states, n_states, n_states, n_states))
        allocate(Gamma_hf_real(n_states, n_states, n_states, n_states))
        
        ! Step 1: temp1(p,q,r,d) = sum_s V(p,q,r,s) * C(s,d)
        write(*,*) '  Step 1/4...'
        temp1 = 0.0d0
        do p = 1, n_states
            do q = 1, n_states
                do r = 1, n_states
                    do d = 1, n_states
                        do s = 1, n_states
                            temp1(p,q,r,d) = temp1(p,q,r,d) + m_v2b(p,q,r,s) * sp_coeffs(s,d)
                        end do
                    end do
                end do
            end do
        end do
        
        ! Step 2: temp2(p,q,c,d) = sum_r temp1(p,q,r,d) * C(r,c)
        write(*,*) '  Step 2/4...'
        temp2 = 0.0d0
        do p = 1, n_states
            do q = 1, n_states
                do c = 1, n_states
                    do d = 1, n_states
                        do r = 1, n_states
                            temp2(p,q,c,d) = temp2(p,q,c,d) + temp1(p,q,r,d) * sp_coeffs(r,c)
                        end do
                    end do
                end do
            end do
        end do
        
        ! Step 3: temp3(p,b,c,d) = sum_q temp2(p,q,c,d) * C(q,b)
        write(*,*) '  Step 3/4...'
        temp3 = 0.0d0
        do p = 1, n_states
            do b = 1, n_states
                do c = 1, n_states
                    do d = 1, n_states
                        do q = 1, n_states
                            temp3(p,b,c,d) = temp3(p,b,c,d) + temp2(p,q,c,d) * sp_coeffs(q,b)
                        end do
                    end do
                end do
            end do
        end do
        
        ! Step 4: Gamma(a,b,c,d) = sum_p temp3(p,b,c,d) * C(p,a)
        write(*,*) '  Step 4/4...'
        Gamma_hf_real = 0.0d0
        do a = 1, n_states
            do b = 1, n_states
                do c = 1, n_states
                    do d = 1, n_states
                        do p = 1, n_states
                            Gamma_hf_real(a,b,c,d) = Gamma_hf_real(a,b,c,d) + temp3(p,b,c,d) * sp_coeffs(p,a)
                        end do
                    end do
                end do
            end do
        end do
        
        write(*,*) 'Transformation complete!'
        write(*,*)
        
        deallocate(temp1, temp2, temp3)
        
        ! Copy to normal-ordered Hamiltonian structure
        allocate(no_ham%f_hf(n_states, n_states))
        allocate(no_ham%Gamma_hf(n_states, n_states, n_states, n_states))
        
        no_ham%f_hf = f_hf_real
        no_ham%Gamma_hf = Gamma_hf_real
        
        deallocate(f_hf_real, Gamma_hf_real)
        
        write(*,*) 'Normal-ordered Hamiltonian ready for CC calculations'
        
    end subroutine normal_order_talsh
    
    ! Deallocate normal-ordered Hamiltonian
    subroutine deallocate_no_ham(no_ham)
        type(normal_ordered_hamiltonian), intent(inout) :: no_ham
        if (allocated(no_ham%f_hf)) deallocate(no_ham%f_hf)
        if (allocated(no_ham%Gamma_hf)) deallocate(no_ham%Gamma_hf)
    end subroutine deallocate_no_ham

end module hartree_fock_module
