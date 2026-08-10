# memcpy_asm.s
# Copies N bytes from source to destination

.section .rodata
src: .string "Copy this!"
fmt: .string "Copied: %s\n"

.section .bss
dest: .space 20     # Allocate 20 bytes of uninitialized memory

.section .text
.global main
main:
    # Prologue
    addi sp, sp, -16
    sd ra, 8(sp)

    la t0, src      # t0 = source pointer
    la t1, dest     # t1 = destination pointer
    li t2, 11       # t2 = number of bytes to copy (10 chars + null terminator)
    li t3, 0        # t3 = counter

loop_memcpy:
    bge t3, t2, end_memcpy # If counter >= bytes to copy, exit
    lb t4, 0(t0)           # Read byte from source
    sb t4, 0(t1)           # Write byte to destination
    addi t0, t0, 1         # Increment source pointer
    addi t1, t1, 1         # Increment destination pointer
    addi t3, t3, 1         # counter++
    j loop_memcpy

end_memcpy:
    # Prepare arguments for printf
    la a0, fmt
    la a1, dest            # Pass the destination address to print
    call printf

    # Epilogue
    li a0, 0
    ld ra, 8(sp)
    addi sp, sp, 16
    ret
