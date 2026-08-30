	.file	"gemm.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.type	init_array, @function
init_array:
.LFB6:
	.cfi_startproc
	addi	sp,sp,-96
	.cfi_def_cfa_offset 96
	sd	ra,88(sp)
	sd	s0,80(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	addi	s0,sp,96
	.cfi_def_cfa 8, 0
	sd	a3,-56(s0)
	sd	a4,-64(s0)
	sd	a5,-72(s0)
	sd	a6,-80(s0)
	sd	a7,-88(s0)
	mv	a5,a0
	sw	a5,-36(s0)
	mv	a5,a1
	sw	a5,-40(s0)
	mv	a5,a2
	sw	a5,-44(s0)
	ld	a5,-56(s0)
	lla	a4,.LC0
	fld	fa5,0(a4)
	fsd	fa5,0(a5)
	ld	a5,-64(s0)
	lla	a4,.LC1
	fld	fa5,0(a4)
	fsd	fa5,0(a5)
	sw	zero,-20(s0)
	j	.L2
.L5:
	sw	zero,-24(s0)
	j	.L3
.L4:
	lw	a5,-20(s0)
	mv	a4,a5
	lw	a5,-24(s0)
	mulw	a5,a4,a5
	sext.w	a5,a5
	addiw	a5,a5,1
	sext.w	a5,a5
	mv	a4,a5
	lw	a5,-36(s0)
	remw	a5,a4,a5
	sext.w	a5,a5
	fcvt.d.w	fa4,a5
	lw	a5,-36(s0)
	fcvt.d.w	fa5,a5
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,608
	mul	a5,a4,a5
	ld	a4,-72(s0)
	add	a4,a4,a5
	fdiv.d	fa5,fa4,fa5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fsd	fa5,0(a5)
	lw	a5,-24(s0)
	addiw	a5,a5,1
	sw	a5,-24(s0)
.L3:
	lw	a5,-24(s0)
	mv	a4,a5
	lw	a5,-40(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L4
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sw	a5,-20(s0)
.L2:
	lw	a5,-20(s0)
	mv	a4,a5
	lw	a5,-36(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L5
	sw	zero,-20(s0)
	j	.L6
.L9:
	sw	zero,-24(s0)
	j	.L7
.L8:
	lw	a5,-24(s0)
	addiw	a5,a5,1
	sext.w	a5,a5
	lw	a4,-20(s0)
	mulw	a5,a4,a5
	sext.w	a5,a5
	mv	a4,a5
	lw	a5,-44(s0)
	remw	a5,a4,a5
	sext.w	a5,a5
	fcvt.d.w	fa4,a5
	lw	a5,-44(s0)
	fcvt.d.w	fa5,a5
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,1408
	mul	a5,a4,a5
	ld	a4,-80(s0)
	add	a4,a4,a5
	fdiv.d	fa5,fa4,fa5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fsd	fa5,0(a5)
	lw	a5,-24(s0)
	addiw	a5,a5,1
	sw	a5,-24(s0)
.L7:
	lw	a5,-24(s0)
	mv	a4,a5
	lw	a5,-44(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L8
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sw	a5,-20(s0)
.L6:
	lw	a5,-20(s0)
	mv	a4,a5
	lw	a5,-36(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L9
	sw	zero,-20(s0)
	j	.L10
.L13:
	sw	zero,-24(s0)
	j	.L11
.L12:
	lw	a5,-24(s0)
	addiw	a5,a5,2
	sext.w	a5,a5
	lw	a4,-20(s0)
	mulw	a5,a4,a5
	sext.w	a5,a5
	mv	a4,a5
	lw	a5,-40(s0)
	remw	a5,a4,a5
	sext.w	a5,a5
	fcvt.d.w	fa4,a5
	lw	a5,-40(s0)
	fcvt.d.w	fa5,a5
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,608
	mul	a5,a4,a5
	ld	a4,-88(s0)
	add	a4,a4,a5
	fdiv.d	fa5,fa4,fa5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fsd	fa5,0(a5)
	lw	a5,-24(s0)
	addiw	a5,a5,1
	sw	a5,-24(s0)
.L11:
	lw	a5,-24(s0)
	mv	a4,a5
	lw	a5,-40(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L12
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sw	a5,-20(s0)
.L10:
	lw	a5,-20(s0)
	mv	a4,a5
	lw	a5,-44(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L13
	nop
	nop
	ld	ra,88(sp)
	.cfi_restore 1
	ld	s0,80(sp)
	.cfi_restore 8
	.cfi_def_cfa 2, 96
	addi	sp,sp,96
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE6:
	.size	init_array, .-init_array
	.section	.rodata
	.align	3
.LC2:
	.string	"==BEGIN DUMP_ARRAYS==\n"
	.align	3
.LC3:
	.string	"C"
	.align	3
.LC4:
	.string	"begin dump: %s"
	.align	3
.LC5:
	.string	"\n"
	.align	3
.LC6:
	.string	"%0.2lf "
	.align	3
.LC7:
	.string	"\nend   dump: %s\n"
	.align	3
.LC8:
	.string	"==END   DUMP_ARRAYS==\n"
	.text
	.align	1
	.type	print_array, @function
print_array:
.LFB7:
	.cfi_startproc
	addi	sp,sp,-48
	.cfi_def_cfa_offset 48
	sd	ra,40(sp)
	sd	s0,32(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	addi	s0,sp,48
	.cfi_def_cfa 8, 0
	mv	a5,a0
	mv	a4,a1
	sd	a2,-48(s0)
	sw	a5,-36(s0)
	mv	a5,a4
	sw	a5,-40(s0)
	la	a5,stderr
	ld	a5,0(a5)
	lla	a1,.LC2
	mv	a0,a5
	call	fprintf@plt
	la	a5,stderr
	ld	a5,0(a5)
	lla	a2,.LC3
	lla	a1,.LC4
	mv	a0,a5
	call	fprintf@plt
	sw	zero,-20(s0)
	j	.L15
.L19:
	sw	zero,-24(s0)
	j	.L16
.L18:
	lw	a5,-20(s0)
	mv	a4,a5
	lw	a5,-36(s0)
	mulw	a5,a4,a5
	sext.w	a5,a5
	lw	a4,-24(s0)
	addw	a5,a4,a5
	sext.w	a5,a5
	mv	a4,a5
	sext.w	a3,a4
	li	a5,1717985280
	addi	a5,a5,1639
	mul	a5,a3,a5
	srli	a5,a5,32
	sraiw	a5,a5,3
	mv	a3,a5
	sraiw	a5,a4,31
	subw	a5,a3,a5
	mv	a3,a5
	mv	a5,a3
	slliw	a5,a5,2
	addw	a5,a5,a3
	slliw	a5,a5,2
	subw	a5,a4,a5
	sext.w	a5,a5
	bne	a5,zero,.L17
	la	a5,stderr
	ld	a5,0(a5)
	lla	a1,.LC5
	mv	a0,a5
	call	fprintf@plt
.L17:
	la	a5,stderr
	ld	a3,0(a5)
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,608
	mul	a5,a4,a5
	ld	a4,-48(s0)
	add	a4,a4,a5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fld	fa5,0(a5)
	fmv.x.d	a2,fa5
	lla	a1,.LC6
	mv	a0,a3
	call	fprintf@plt
	lw	a5,-24(s0)
	addiw	a5,a5,1
	sw	a5,-24(s0)
.L16:
	lw	a5,-24(s0)
	mv	a4,a5
	lw	a5,-40(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L18
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sw	a5,-20(s0)
.L15:
	lw	a5,-20(s0)
	mv	a4,a5
	lw	a5,-36(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L19
	la	a5,stderr
	ld	a5,0(a5)
	lla	a2,.LC3
	lla	a1,.LC7
	mv	a0,a5
	call	fprintf@plt
	la	a5,stderr
	ld	a5,0(a5)
	lla	a1,.LC8
	mv	a0,a5
	call	fprintf@plt
	nop
	ld	ra,40(sp)
	.cfi_restore 1
	ld	s0,32(sp)
	.cfi_restore 8
	.cfi_def_cfa 2, 48
	addi	sp,sp,48
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE7:
	.size	print_array, .-print_array
	.align	1
	.type	kernel_gemm, @function
kernel_gemm:
.LFB8:
	.cfi_startproc
	addi	sp,sp,-96
	.cfi_def_cfa_offset 96
	sd	ra,88(sp)
	sd	s0,80(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	addi	s0,sp,96
	.cfi_def_cfa 8, 0
	fsd	fa0,-56(s0)
	fsd	fa1,-64(s0)
	sd	a3,-72(s0)
	sd	a4,-80(s0)
	sd	a5,-88(s0)
	mv	a5,a0
	sw	a5,-36(s0)
	mv	a5,a1
	sw	a5,-40(s0)
	mv	a5,a2
	sw	a5,-44(s0)
	sw	zero,-20(s0)
	j	.L21
.L28:
	sw	zero,-24(s0)
	j	.L22
.L23:
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,608
	mul	a5,a4,a5
	ld	a4,-72(s0)
	add	a4,a4,a5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fld	fa4,0(a5)
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,608
	mul	a5,a4,a5
	ld	a4,-72(s0)
	add	a4,a4,a5
	fld	fa5,-64(s0)
	fmul.d	fa5,fa4,fa5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fsd	fa5,0(a5)
	lw	a5,-24(s0)
	addiw	a5,a5,1
	sw	a5,-24(s0)
.L22:
	lw	a5,-24(s0)
	mv	a4,a5
	lw	a5,-40(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L23
	sw	zero,-28(s0)
	j	.L24
.L27:
	sw	zero,-24(s0)
	j	.L25
.L26:
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,608
	mul	a5,a4,a5
	ld	a4,-72(s0)
	add	a4,a4,a5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fld	fa4,0(a5)
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,1408
	mul	a5,a4,a5
	ld	a4,-80(s0)
	add	a4,a4,a5
	lw	a5,-28(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fld	fa3,0(a5)
	fld	fa5,-56(s0)
	fmul.d	fa3,fa3,fa5
	lw	a4,-28(s0)
	li	a5,8192
	addi	a5,a5,608
	mul	a5,a4,a5
	ld	a4,-88(s0)
	add	a4,a4,a5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fld	fa5,0(a5)
	fmul.d	fa5,fa3,fa5
	lw	a4,-20(s0)
	li	a5,8192
	addi	a5,a5,608
	mul	a5,a4,a5
	ld	a4,-72(s0)
	add	a4,a4,a5
	fadd.d	fa5,fa4,fa5
	lw	a5,-24(s0)
	slli	a5,a5,3
	add	a5,a4,a5
	fsd	fa5,0(a5)
	lw	a5,-24(s0)
	addiw	a5,a5,1
	sw	a5,-24(s0)
.L25:
	lw	a5,-24(s0)
	mv	a4,a5
	lw	a5,-40(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L26
	lw	a5,-28(s0)
	addiw	a5,a5,1
	sw	a5,-28(s0)
.L24:
	lw	a5,-28(s0)
	mv	a4,a5
	lw	a5,-44(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L27
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sw	a5,-20(s0)
.L21:
	lw	a5,-20(s0)
	mv	a4,a5
	lw	a5,-36(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	blt	a4,a5,.L28
	nop
	nop
	ld	ra,88(sp)
	.cfi_restore 1
	ld	s0,80(sp)
	.cfi_restore 8
	.cfi_def_cfa 2, 96
	addi	sp,sp,96
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE8:
	.size	kernel_gemm, .-kernel_gemm
	.section	.rodata
	.align	3
.LC9:
	.string	""
	.text
	.align	1
	.globl	main
	.type	main, @function
main:
.LFB9:
	.cfi_startproc
	addi	sp,sp,-96
	.cfi_def_cfa_offset 96
	sd	ra,88(sp)
	sd	s0,80(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	addi	s0,sp,96
	.cfi_def_cfa 8, 0
	mv	a5,a0
	sd	a1,-96(s0)
	sw	a5,-84(s0)
	li	a5,1000
	sw	a5,-20(s0)
	li	a5,1100
	sw	a5,-24(s0)
	li	a5,1200
	sw	a5,-28(s0)
	li	a1,8
	li	a5,1101824
	addi	a0,a5,-1824
	call	polybench_alloc_data@plt
	sd	a0,-40(s0)
	li	a1,8
	li	a5,1200128
	addi	a0,a5,-128
	call	polybench_alloc_data@plt
	sd	a0,-48(s0)
	li	a1,8
	li	a5,1318912
	addi	a0,a5,1088
	call	polybench_alloc_data@plt
	sd	a0,-56(s0)
	addi	a4,s0,-72
	addi	a3,s0,-64
	lw	a2,-28(s0)
	lw	a1,-24(s0)
	lw	a0,-20(s0)
	ld	a7,-56(s0)
	ld	a6,-48(s0)
	ld	a5,-40(s0)
	call	init_array
	fld	fa5,-64(s0)
	fld	fa4,-72(s0)
	lw	a2,-28(s0)
	lw	a1,-24(s0)
	lw	a0,-20(s0)
	ld	a5,-56(s0)
	ld	a4,-48(s0)
	ld	a3,-40(s0)
	fmv.d	fa1,fa4
	fmv.d	fa0,fa5
	call	kernel_gemm
	lw	a5,-84(s0)
	sext.w	a4,a5
	li	a5,42
	ble	a4,a5,.L30
	ld	a5,-96(s0)
	ld	a5,0(a5)
	lla	a1,.LC9
	mv	a0,a5
	call	strcmp@plt
	mv	a5,a0
	bne	a5,zero,.L30
	lw	a4,-24(s0)
	lw	a5,-20(s0)
	ld	a2,-40(s0)
	mv	a1,a4
	mv	a0,a5
	call	print_array
.L30:
	ld	a0,-40(s0)
	call	free@plt
	ld	a0,-48(s0)
	call	free@plt
	ld	a0,-56(s0)
	call	free@plt
	li	a5,0
	mv	a0,a5
	ld	ra,88(sp)
	.cfi_restore 1
	ld	s0,80(sp)
	.cfi_restore 8
	.cfi_def_cfa 2, 96
	addi	sp,sp,96
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.section	.rodata
	.align	3
.LC0:
	.word	0
	.word	1073217536
	.align	3
.LC1:
	.word	858993459
	.word	1072902963
	.ident	"GCC: (GNU) 15.1.0"
	.section	.note.GNU-stack,"",@progbits
