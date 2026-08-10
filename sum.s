# sum.s
# Calculates the sum of 1 to N

.section .rodata
fmt: .string "Sum(1..10) = %d\n"

.section .text
.global main
main:
    # Prologue: save return address
    addi sp, sp, -16
    sd ra, 8(sp)

    li a0, 10       # N = 10
    li a1, 0        # Accumulator (sum)
    li t0, 1        # Loop counter (i)

loop_sum:
    bgt t0, a0, end_sum  # If i > N, exit loop
    add a1, a1, t0       # sum = sum + i
    addi t0, t0, 1       # i++
    j loop_sum           # Repeat

end_sum:
    # Prepare arguments for printf
    la a0, fmt           # a0 = address of format string
    # a1 already contains the sum
    call printf

    # Epilogue: restore return address and return 0
    li a0, 0
    ld ra, 8(sp)
    addi sp, sp, 16
    ret
