module snt_io_module
    implicit none
    
    ! Single particle orbit structure
    type :: single_particle_orbits
        integer :: n_p_core, n_n_core
        integer :: n_p_orbits, n_n_orbits
        integer :: total_orbits
        integer, allocatable :: n(:)           ! Principal quantum number
        integer, allocatable :: l(:)           ! Orbital angular momentum
        integer, allocatable :: j(:)           ! 2*j (total angular momentum)
        integer, allocatable :: jz(:)          ! 2*jz (projection)
        integer, allocatable :: tz(:)          ! isospin projection (-1 proton, +1 neutron)
        integer, allocatable :: coupling_map(:) ! Maps m-scheme to j-scheme index
    end type single_particle_orbits
    
    ! Two-body matrix element entry
    type :: tbme_entry
        integer :: idx_i, idx_j, idx_k, idx_l  ! Orbit indices (0-based internally)
        integer :: J_val                        ! Total angular momentum
        real(8) :: val_me                       ! Matrix element value
    end type tbme_entry
    
    ! Potential structure
    type :: potential
        integer :: n_orbits
        real(8), allocatable :: v1b(:,:)       ! One-body matrix elements
        type(tbme_entry), allocatable :: v2b(:) ! Two-body matrix elements
        integer :: n_tbme                       ! Number of 2-body elements
        real(8) :: hw                          ! Oscillator frequency
    end type potential
    
contains

    ! Initialize single particle orbits
    subroutine init_sp_orbits(orbits, n_p_core, n_n_core, n_p_orbits, n_n_orbits)
        type(single_particle_orbits), intent(inout) :: orbits
        integer, intent(in) :: n_p_core, n_n_core, n_p_orbits, n_n_orbits
        
        orbits%n_p_core = n_p_core
        orbits%n_n_core = n_n_core
        orbits%n_p_orbits = n_p_orbits
        orbits%n_n_orbits = n_n_orbits
        orbits%total_orbits = n_p_orbits + n_n_orbits
        
        allocate(orbits%n(orbits%total_orbits))
        allocate(orbits%l(orbits%total_orbits))
        allocate(orbits%j(orbits%total_orbits))
        allocate(orbits%jz(orbits%total_orbits))
        allocate(orbits%tz(orbits%total_orbits))
        allocate(orbits%coupling_map(orbits%total_orbits))
        
        orbits%n = 0
        orbits%l = 0
        orbits%j = 0
        orbits%jz = 0
        orbits%tz = 0
        orbits%coupling_map = 0
    end subroutine init_sp_orbits
    
    ! Initialize potential
    subroutine init_potential(pot, n_orbits, n_tbme_est)
        type(potential), intent(inout) :: pot
        integer, intent(in) :: n_orbits, n_tbme_est
        
        pot%n_orbits = n_orbits
        pot%n_tbme = 0
        pot%hw = 0.0d0
        
        allocate(pot%v1b(n_orbits, n_orbits))
        allocate(pot%v2b(n_tbme_est))
        pot%v1b = 0.0d0
    end subroutine init_potential
    
    ! Read SNT file
    subroutine read_snt(filename, scale_factor, orbits, pot)
        character(len=*), intent(in) :: filename
        real(8), intent(in) :: scale_factor
        type(single_particle_orbits), intent(out) :: orbits
        type(potential), intent(out) :: pot
        
        integer :: unit_num, ios, i, j
        integer :: n_p_orbits, n_n_orbits, n_p_core, n_n_core
        integer :: total_orbits, n_elements, n_elements_2b
        integer :: idx, i_idx, j_idx, k_idx, l_idx, J_val
        real(8) :: val
        character(len=256) :: line
        
        unit_num = 10
        open(unit=unit_num, file=trim(filename), status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(*,*) 'Error opening file: "', trim(filename), '" IOS=', ios
            stop
        end if
        
        ! Skip comment lines
        do
            read(unit_num, '(A)', iostat=ios) line
            if (ios /= 0) exit
            if (len_trim(line) == 0) cycle
            if (line(1:1) == '!') cycle
            exit
        end do
        
        ! Read model space definition
        read(line, *) n_p_orbits, n_n_orbits, n_p_core, n_n_core
        total_orbits = n_p_orbits + n_n_orbits
        
        call init_sp_orbits(orbits, n_p_core, n_n_core, n_p_orbits, n_n_orbits)
        
        ! Read orbit definitions
        do i = 1, total_orbits
            do
                read(unit_num, '(A)', iostat=ios) line
                if (len_trim(line) == 0) cycle
                if (line(1:1) == '!') cycle
                exit
            end do
            read(line, *) idx, orbits%n(i), orbits%l(i), orbits%j(i), orbits%tz(i)
            orbits%jz(i) = orbits%j(i)  ! For j-scheme, jz = j initially
        end do
        
        ! Read 1-body interaction
        do
            read(unit_num, '(A)', iostat=ios) line
            if (ios /= 0) exit
            if (len_trim(line) == 0) cycle
            if (line(1:1) == '!') cycle
            exit
        end do
        
        read(line, *) n_elements
        call init_potential(pot, total_orbits, 10000)  ! Estimate 10000 TBMEs
        
        do i = 1, n_elements
            do
                read(unit_num, '(A)', iostat=ios) line
                if (len_trim(line) == 0) cycle
                if (line(1:1) == '!') cycle
                exit
            end do
            read(line, *) i_idx, j_idx, val
            pot%v1b(i_idx, j_idx) = val
            if (i_idx /= j_idx) then
                pot%v1b(j_idx, i_idx) = val  ! Hermitian
            end if
        end do
        
        ! Read 2-body interaction
        do
            read(unit_num, '(A)', iostat=ios) line
            if (ios /= 0) exit
            if (len_trim(line) == 0) cycle
            if (line(1:1) == '!') cycle
            exit
        end do
        
        read(line, *) n_elements_2b
        
        pot%n_tbme = n_elements_2b
        do i = 1, n_elements_2b
            do
                read(unit_num, '(A)', iostat=ios) line
                if (len_trim(line) == 0) cycle
                if (line(1:1) == '!') cycle
                exit
            end do
            read(line, *) i_idx, j_idx, k_idx, l_idx, J_val, val
            
            ! Store as 0-based indices internally
            pot%v2b(i)%idx_i = i_idx - 1
            pot%v2b(i)%idx_j = j_idx - 1
            pot%v2b(i)%idx_k = k_idx - 1
            pot%v2b(i)%idx_l = l_idx - 1
            pot%v2b(i)%J_val = J_val
            pot%v2b(i)%val_me = val * scale_factor
        end do
        
        close(unit_num)
        
        write(*,'(A,I4,A,I4)') ' Read ', total_orbits, ' orbits and ', n_elements_2b, ' TBMEs'
    end subroutine read_snt
    
    ! Deallocate structures
    subroutine deallocate_sp_orbits(orbits)
        type(single_particle_orbits), intent(inout) :: orbits
        if (allocated(orbits%n)) deallocate(orbits%n)
        if (allocated(orbits%l)) deallocate(orbits%l)
        if (allocated(orbits%j)) deallocate(orbits%j)
        if (allocated(orbits%jz)) deallocate(orbits%jz)
        if (allocated(orbits%tz)) deallocate(orbits%tz)
        if (allocated(orbits%coupling_map)) deallocate(orbits%coupling_map)
    end subroutine deallocate_sp_orbits
    
    subroutine deallocate_potential(pot)
        type(potential), intent(inout) :: pot
        if (allocated(pot%v1b)) deallocate(pot%v1b)
        if (allocated(pot%v2b)) deallocate(pot%v2b)
    end subroutine deallocate_potential

end module snt_io_module
