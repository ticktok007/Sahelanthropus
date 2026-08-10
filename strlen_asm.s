# strlen_asm.s
# Calculates the length of a null-terminated string

.section .rodata
str: .string "Hello, RISC-V!"
fmt: .string "Length: %d\n"

.section .text
.global main
main:
    # Prologue
    addi sp, sp, -16
    sd ra, 8(sp)

    la t0, str      # t0 = pointer to string
    li a1, 0        # a1 = length counter

loop_strlen:
    lb t1, 0(t0)         # Load a byte from the string
    beqz t1, end_strlen  # If byte is 0 (null terminator), exit
    addi a1, a1, 1       # length++
    addi t0, t0, 1       # pointer++
    j loop_strlen

end_strlen:
    # Prepare arguments for printf
    la a0, fmt
    # a1 already contains the length
    call printf

    # Epilogue
    li a0, 0
    ld ra, 8(sp)
    addi sp, sp, 16
    ret
