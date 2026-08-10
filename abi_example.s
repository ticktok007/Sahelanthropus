# abi_example.s
# Example: A simple function demonstrating ABI usage and pseudo-instructions
# C equivalent: int add_nums(int a, int b) { return a + b; }

.global add_nums
add_nums:
    # The ABI dictates that 'a' is in a0, and 'b' is in a1.
    # The return value must be placed in a0.
    
    add a0, a0, a1    # a0 = a + b (Uses ABI register names)
    
    ret               # Pseudo-instruction for: jalr x0, 0(ra)
                      # Returns control to the caller