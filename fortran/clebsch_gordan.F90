module clebsch_gordan
    implicit none
    
    ! Simple CG coefficient cache
    integer, parameter :: MAX_CACHE = 100000
    type :: cg_cache_entry
        integer :: j1, m1, j2, m2, J, M
        real(8) :: value
        logical :: used
    end type cg_cache_entry
    
    type(cg_cache_entry), save :: cg_cache(MAX_CACHE)
    integer, save :: n_cached = 0
    
    real(8), save :: fact(0:50) ! Precomputed factorials
    logical, save :: fact_init = .false.
    
contains

    ! Initialize CG cache and factorials
    subroutine init_cg_cache()
        integer :: i
        do i = 1, MAX_CACHE
            cg_cache(i)%used = .false.
        end do
        n_cached = 0
        
        if (.not. fact_init) then
            fact(0) = 1.0d0
            do i = 1, 50
                fact(i) = fact(i-1) * dble(i)
            end do
            fact_init = .true.
        end if
    end subroutine init_cg_cache
    
    ! Get CG coefficient with caching
    ! All arguments are 2*quantum_number (e.g., j1=3 means j=3/2)
    function get_cg_cached(two_j1, two_m1, two_j2, two_m2, two_J, two_M) result(cg_val)
        integer, intent(in) :: two_j1, two_m1, two_j2, two_m2, two_J, two_M
        real(8) :: cg_val
        integer :: i
        
        ! Selection rules check (fast fail)
        if (two_m1 + two_m2 /= two_M) then
            cg_val = 0.0d0
            return
        end if
        
        ! Search cache
        do i = 1, n_cached
            if (cg_cache(i)%used .and. &
                cg_cache(i)%j1 == two_j1 .and. &
                cg_cache(i)%m1 == two_m1 .and. &
                cg_cache(i)%j2 == two_j2 .and. &
                cg_cache(i)%m2 == two_m2 .and. &
                cg_cache(i)%J == two_J .and. &
                cg_cache(i)%M == two_M) then
                cg_val = cg_cache(i)%value
                return
            end if
        end do
        
        ! Compute CG coefficient
        cg_val = compute_cg(two_j1, two_m1, two_j2, two_m2, two_J, two_M)
        
        ! Cache the result
        if (n_cached < MAX_CACHE) then
            n_cached = n_cached + 1
            cg_cache(n_cached)%j1 = two_j1
            cg_cache(n_cached)%m1 = two_m1
            cg_cache(n_cached)%j2 = two_j2
            cg_cache(n_cached)%m2 = two_m2
            cg_cache(n_cached)%J = two_J
            cg_cache(n_cached)%M = two_M
            cg_cache(n_cached)%value = cg_val
            cg_cache(n_cached)%used = .true.
        end if
    end function get_cg_cached
    
    ! Compute CG coefficient using Racah formula
    ! All inputs are 2*quantum_number
    function compute_cg(tj1, tm1, tj2, tm2, tJ, tM) result(cg)
        integer, intent(in) :: tj1, tm1, tj2, tm2, tJ, tM
        real(8) :: cg
        integer :: k, k_min, k_max
        real(8) :: term, sum_k, delta_factor, sqrt_factor
        
        ! Basic checks
        if (tm1 + tm2 /= tM) then
            cg = 0.0d0
            return
        end if
        
        if (tj1 < abs(tm1) .or. tj2 < abs(tm2) .or. tJ < abs(tM)) then
            cg = 0.0d0
            return
        end if
        
        if (tJ > tj1 + tj2 .or. tJ < abs(tj1 - tj2)) then
            cg = 0.0d0
            return
        end if
        
        ! Check parity (j1+j2-J must be integer)
        if (mod(tj1 + tj2 - tJ, 2) /= 0) then
            cg = 0.0d0
            return
        end if
        
        ! Initialize factorials if needed
        if (.not. fact_init) call init_cg_cache()

        ! Racah formula implementation
        ! Using integer arithmetic for factorials where possible: (2a)/2 -> a
        
        ! Delta factor: Delta(j1, j2, J)
        ! sqrt( (j1+j2-J)! (j1-j2+J)! (-j1+j2+J)! / (j1+j2+J+1)! )
        delta_factor = sqrt(fact((tj1 + tj2 - tJ)/2) * &
                            fact((tj1 - tj2 + tJ)/2) * &
                            fact((-tj1 + tj2 + tJ)/2) / &
                            fact((tj1 + tj2 + tJ)/2 + 1))
                            
        ! Sqrt factor
        ! sqrt( (2J+1) (j1+m1)! (j1-m1)! (j2+m2)! (j2-m2)! (J+M)! (J-M)! )
        sqrt_factor = sqrt(dble(tJ + 1) * &
                           fact((tj1 + tm1)/2) * fact((tj1 - tm1)/2) * &
                           fact((tj2 + tm2)/2) * fact((tj2 - tm2)/2) * &
                           fact((tJ + tM)/2) * fact((tJ - tM)/2))
                           
        ! Sum over k
        ! Conditions for k: factorial arguments must be >= 0
        ! k >= 0
        ! (tj1 + tj2 - tJ)/2 - k >= 0  => k <= (tj1 + tj2 - tJ)/2
        ! (tj1 - tm1)/2 - k >= 0      => k <= (tj1 - tm1)/2
        ! (tj2 + tm2)/2 - k >= 0      => k <= (tj2 + tm2)/2
        ! (tJ - tj2 + tm1)/2 + k >= 0 => k >= -(tJ - tj2 + tm1)/2
        ! (tJ - tj1 - tm2)/2 + k >= 0 => k >= -(tJ - tj1 - tm2)/2
        
        k_min = max(0, max(-(tJ - tj2 + tm1)/2, -(tJ - tj1 - tm2)/2))
        k_max = min((tj1 + tj2 - tJ)/2, min((tj1 - tm1)/2, (tj2 + tm2)/2))
        
        sum_k = 0.0d0
        if (k_min <= k_max) then
            do k = k_min, k_max
                term = ((-1.0d0)**k) / (fact(k) * &
                        fact((tj1 + tj2 - tJ)/2 - k) * &
                        fact((tj1 - tm1)/2 - k) * &
                        fact((tj2 + tm2)/2 - k) * &
                        fact((tJ - tj2 + tm1)/2 + k) * &
                        fact((tJ - tj1 - tm2)/2 + k))
                sum_k = sum_k + term
            end do
        end if
        
        cg = delta_factor * sqrt_factor * sum_k
        
    end function compute_cg

end module clebsch_gordan
