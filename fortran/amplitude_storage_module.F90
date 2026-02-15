module amplitude_storage_module
    implicit none
    
    ! Storage for passing amplitudes between solvers
    real(8), allocatable :: stored_t1(:,:)
    real(8), allocatable :: stored_t2(:,:,:,:)
    real(8), allocatable :: stored_t3(:,:,:,:,:,:)
    real(8), allocatable :: stored_t4(:,:,:,:,:,:,:,:)
    
    logical :: have_t1 = .false.
    logical :: have_t2 = .false.
    logical :: have_t3 = .false.
    logical :: have_t4 = .false.
    
contains
    
    subroutine store_t2(t2)
        real(8), intent(in) :: t2(:,:,:,:)
        if (.not. allocated(stored_t2)) then
            allocate(stored_t2(size(t2, 1), size(t2, 2), size(t2, 3), size(t2, 4)))
        end if
        stored_t2 = t2
        have_t2 = .true.
    end subroutine store_t2
    
    subroutine store_t1_t2(t1, t2)
        real(8), intent(in) :: t1(:,:)
        real(8), intent(in) :: t2(:,:,:,:)
        if (.not. allocated(stored_t1)) then
            allocate(stored_t1(size(t1, 1), size(t1, 2)))
        end if
        if (.not. allocated(stored_t2)) then
            allocate(stored_t2(size(t2, 1), size(t2, 2), size(t2, 3), size(t2, 4)))
        end if
        stored_t1 = t1
        stored_t2 = t2
        have_t1 = .true.
        have_t2 = .true.
    end subroutine store_t1_t2
    
    subroutine store_ccsdt_amplitudes(t1, t2, t3)
        real(8), intent(in) :: t1(:,:)
        real(8), intent(in) :: t2(:,:,:,:)
        real(8), intent(in) :: t3(:,:,:,:,:,:)
        if (.not. allocated(stored_t1)) then
            allocate(stored_t1(size(t1, 1), size(t1, 2)))
        end if
        if (.not. allocated(stored_t2)) then
            allocate(stored_t2(size(t2, 1), size(t2, 2), size(t2, 3), size(t2, 4)))
        end if
        if (.not. allocated(stored_t3)) then
            allocate(stored_t3(size(t3, 1), size(t3, 2), size(t3, 3), size(t3, 4), size(t3, 5), size(t3, 6)))
        end if
        stored_t1 = t1
        stored_t2 = t2
        stored_t3 = t3
        have_t1 = .true.
        have_t2 = .true.
        have_t3 = .true.
    end subroutine store_ccsdt_amplitudes
    
    subroutine clear_storage()
        if (allocated(stored_t1)) deallocate(stored_t1)
        if (allocated(stored_t2)) deallocate(stored_t2)
        if (allocated(stored_t3)) deallocate(stored_t3)
        if (allocated(stored_t4)) deallocate(stored_t4)
        have_t1 = .false.
        have_t2 = .false.
        have_t3 = .false.
        have_t4 = .false.
    end subroutine clear_storage
    
end module amplitude_storage_module
