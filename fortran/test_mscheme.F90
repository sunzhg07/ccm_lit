program test_mscheme
    use snt_io_module
    use m_scheme_module
    implicit none
    
    type(single_particle_orbits) :: j_orbits, m_orbits
    type(potential) :: j_potential
    real(8), allocatable :: m_v1b(:,:)
    real(8), allocatable :: m_v2b(:,:,:,:)
    character(len=256) :: snt_file
    real(8) :: scale_factor
    integer :: i, n_states
    
    ! Setup parameters
    snt_file = 'test_input.snt'
    scale_factor = (42.0d0/56.0d0)**0.30d0
    
    write(*,*) 'Reading SNT file...'
    call read_snt(snt_file, scale_factor, j_orbits, j_potential)
    
    write(*,*) 'Generating M-scheme basis...'
    call generate_m_scheme(j_orbits, m_orbits)
    n_states = m_orbits%total_orbits
    
    write(*,*) 'M-Scheme Orbits (First 10):'
    write(*,'(A5,A5,A5,A5,A5,A5)') 'Idx', 'n', 'l', '2j', '2jz', '2tz'
    do i = 1, min(10, n_states)
        write(*,'(I5,I5,I5,I5,I5,I5)') i, m_orbits%n(i), m_orbits%l(i), &
                                        m_orbits%j(i), m_orbits%jz(i), m_orbits%tz(i)
    end do
    
    write(*,*) 'Decoupling interactions...'
    call decouple_1b(j_potential, m_orbits, m_v1b)
    call decouple_2b(j_potential, m_orbits, m_v2b)
    
    write(*,*) 'Verification Complete.'
    
end program test_mscheme
