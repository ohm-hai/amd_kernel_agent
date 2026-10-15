	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	kda_decode_step_kern_0
	.p2align	8
	.type	kda_decode_step_kern_0,@function
kda_decode_step_kern_0:
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 30
	s_add_i32 s3, s2, s3
	s_ashr_i32 s10, s3, 2
	s_and_b32 s3, s3, -4
	s_sub_i32 s24, s2, s3
	s_cmp_lg_u32 s2, s3
	s_cselect_b64 s[6:7], -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b64 s[2:3], -1, 0
	s_load_dwordx2 s[8:9], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x18
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dwordx2 s[4:5], s[0:1], 0x48
	s_and_b64 s[2:3], s[2:3], s[6:7]
	s_subb_u32 s3, s10, 0
	v_and_b32_e32 v3, 63, v0
	s_lshl_b32 s2, s3, 8
	s_waitcnt lgkmcnt(0)
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, -1
	v_lshl_or_b32 v1, v3, 2, s2
	s_and_b32 s17, s17, 0xffff
	s_mov_b32 s18, s10
	s_mov_b32 s19, s11
	s_and_b32 s13, s13, 0xffff
	s_mov_b32 s14, s10
	s_mov_b32 s15, s11
	buffer_load_ushort v2, v1, s[8:11], 0 offen
	buffer_load_ushort v6, v1, s[8:11], 0 offen offset:2
	buffer_load_ushort v7, v1, s[16:19], 0 offen
	buffer_load_ushort v8, v1, s[16:19], 0 offen offset:2
	buffer_load_ushort v9, v1, s[12:15], 0 offen
	buffer_load_ushort v18, v1, s[12:15], 0 offen offset:2
	s_load_dwordx2 s[20:21], s[0:1], 0x60
	s_load_dwordx2 s[12:13], s[0:1], 0x70
	s_load_dwordx2 s[16:17], s[0:1], 0x100
	s_lshl_b32 s6, s3, 1
	s_mov_b32 s22, s10
	s_mov_b32 s23, s11
	s_waitcnt lgkmcnt(0)
	s_and_b32 s21, s21, 0xffff
	v_mov_b32_e32 v1, s6
	buffer_load_ushort v20, v1, s[20:23], 0 offen
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_b32_e32 v0, 24, v0
	v_lshl_or_b32 v5, s24, 5, v0
	v_lshl_add_u32 v0, v3, 8, v5
	s_lshl_b32 s3, s3, 14
	v_add_lshl_u32 v48, v0, s3, 2
	s_and_b32 s13, s13, 0xffff
	buffer_load_dwordx4 v[10:13], v48, s[12:15], 0 offen
	buffer_load_dwordx4 v[14:17], v48, s[12:15], 0 offen offset:512
	s_mov_b32 s3, 0x3fb8aa3b
	s_and_b32 s17, s17, 0xffff
	buffer_load_dword v4, off, s[16:19], 0
	s_mov_b32 s8, 0xc2ce8ed0
	s_mov_b32 s9, 0x42b17218
	s_mov_b32 s6, s10
	s_mov_b32 s7, s11
	s_and_b32 s5, s5, 0xffff
	v_lshl_add_u32 v50, v5, 1, s2
	buffer_load_dwordx4 v[30:33], v48, s[12:15], 0 offen offset:528
	s_waitcnt vmcnt(10)
	v_lshlrev_b32_e32 v22, 16, v2
	s_waitcnt vmcnt(9)
	v_lshlrev_b32_e32 v27, 16, v6
	s_waitcnt vmcnt(8)
	v_lshlrev_b32_e32 v0, 16, v7
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v2, 16, v8
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v21, 16, v9
	v_mul_f32_e32 v7, v22, v22
	v_mul_f32_e32 v9, v27, v27
	v_mul_f32_e32 v23, v0, v0
	v_mul_f32_e32 v1, v2, v2
	v_add_f32_e32 v25, v7, v9
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v24, 16, v18
	v_pk_add_f32 v[18:19], v[22:23], v[0:1]
	v_mov_b32_e32 v1, v25
	v_mov_b32_e32 v7, v25
	v_mov_b32_e32 v18, v19
	v_mov_b32_e32 v23, v19
	v_permlane32_swap_b32_e32 v1, v7
	s_nop 0
	v_permlane32_swap_b32_e32 v18, v23
	v_cmp_eq_u32_e32 vcc, v1, v25
	v_mul_f32_e32 v6, v27, v2
	v_mul_f32_e32 v8, v22, v0
	v_cndmask_b32_e32 v1, v1, v7, vcc
	v_cmp_eq_u32_e32 vcc, v18, v19
	v_mov_b32_e32 v9, v19
	v_add_f32_e32 v1, v25, v1
	v_cndmask_b32_e32 v7, v18, v23, vcc
	v_pk_add_f32 v[8:9], v[8:9], v[6:7]
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v7, v1
	s_nop 1
	v_permlane16_swap_b32_e32 v6, v7
	v_cmp_eq_u32_e32 vcc, v6, v1
	v_mov_b32_e32 v18, v9
	v_mov_b32_e32 v19, v9
	v_cndmask_b32_e32 v6, v6, v7, vcc
	v_add_f32_e32 v1, v1, v6
	ds_swizzle_b32 v6, v1 offset:swizzle(SWAP,8)
	v_permlane16_swap_b32_e32 v18, v19
	v_cmp_eq_u32_e32 vcc, v18, v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v6
	v_cndmask_b32_e32 v7, v18, v19, vcc
	v_add_f32_e32 v7, v9, v7
	ds_swizzle_b32 v9, v7 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v18, v1 offset:swizzle(SWAP,4)
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v6, 16, v20
	v_mov_b32_e32 v19, v8
	v_mov_b32_e32 v20, v8
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v7, v7, v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v18
	ds_swizzle_b32 v9, v7 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v18, v1 offset:swizzle(SWAP,2)
	v_permlane32_swap_b32_e32 v19, v20
	v_cmp_eq_u32_e32 vcc, v19, v8
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v9, v7, v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v1, v18
	v_cndmask_b32_e32 v18, v19, v20, vcc
	v_add_f32_e32 v8, v8, v18
	v_mov_b32_e32 v18, v8
	v_mov_b32_e32 v19, v8
	s_nop 1
	v_permlane16_swap_b32_e32 v18, v19
	v_cmp_eq_u32_e32 vcc, v18, v8
	ds_swizzle_b32 v1, v9 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v49, v7 offset:swizzle(SWAP,1)
	v_cndmask_b32_e32 v18, v18, v19, vcc
	v_mul_f32_e32 v19, 0x3fb8aa3b, v21
	v_fma_f32 v20, v21, s3, -v19
	v_rndne_f32_e32 v23, v19
	v_add_f32_e32 v8, v8, v18
	v_fmac_f32_e32 v20, 0x32a5705f, v21
	v_sub_f32_e32 v19, v19, v23
	ds_swizzle_b32 v18, v8 offset:swizzle(SWAP,8)
	v_add_f32_e32 v19, v19, v20
	v_exp_f32_e32 v19, v19
	v_cvt_i32_f32_e32 v20, v23
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v1, v9, v1
	v_mul_f32_e32 v9, 0x3fb8aa3b, v24
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v51, v8, v18
	v_ldexp_f32 v8, v19, v20
	v_fma_f32 v18, v24, s3, -v9
	v_rndne_f32_e32 v19, v9
	v_fmac_f32_e32 v18, 0x32a5705f, v24
	v_sub_f32_e32 v9, v9, v19
	v_add_f32_e32 v9, v9, v18
	v_exp_f32_e32 v9, v9
	v_cvt_i32_f32_e32 v18, v19
	v_cmp_ngt_f32_e32 vcc, s8, v21
	v_mov_b32_e32 v19, 0x7f800000
	s_nop 0
	v_cndmask_b32_e32 v8, 0, v8, vcc
	v_cmp_nlt_f32_e32 vcc, s9, v21
	s_nop 1
	v_cndmask_b32_e32 v26, v19, v8, vcc
	v_ldexp_f32 v8, v9, v18
	v_cmp_ngt_f32_e32 vcc, s8, v24
	s_nop 1
	v_cndmask_b32_e32 v8, 0, v8, vcc
	v_cmp_nlt_f32_e32 vcc, s9, v24
	s_nop 1
	v_cndmask_b32_e32 v34, v19, v8, vcc
	buffer_load_dwordx4 v[18:21], v48, s[12:15], 0 offen offset:16
	s_waitcnt vmcnt(4)
	v_pk_mul_f32 v[8:9], v[26:27], v[10:11] op_sel_hi:[0,1]
	s_waitcnt vmcnt(3)
	v_pk_mul_f32 v[10:11], v[34:35], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[8:9], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[10:11], v[2:3] op_sel_hi:[1,0]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[20:21], v[26:27], v[20:21] op_sel_hi:[0,1]
	v_pk_add_f32 v[14:15], v[14:15], v[24:25]
	s_nop 0
	v_mov_b32_e32 v25, v15
	v_mov_b32_e32 v28, v15
	v_mov_b32_e32 v23, v14
	v_mov_b32_e32 v24, v14
	v_permlane32_swap_b32_e32 v25, v28
	s_nop 0
	v_permlane32_swap_b32_e32 v23, v24
	v_cmp_eq_u32_e32 vcc, v25, v15
	s_nop 1
	v_cndmask_b32_e32 v25, v25, v28, vcc
	v_cmp_eq_u32_e32 vcc, v23, v14
	s_nop 1
	v_cndmask_b32_e32 v24, v23, v24, vcc
	v_pk_add_f32 v[14:15], v[14:15], v[24:25]
	s_nop 0
	v_mov_b32_e32 v25, v15
	v_mov_b32_e32 v28, v15
	v_mov_b32_e32 v23, v14
	v_mov_b32_e32 v24, v14
	v_permlane16_swap_b32_e32 v25, v28
	s_nop 0
	v_permlane16_swap_b32_e32 v23, v24
	v_cmp_eq_u32_e32 vcc, v25, v15
	s_nop 1
	v_cndmask_b32_e32 v25, v25, v28, vcc
	v_cmp_eq_u32_e32 vcc, v23, v14
	s_nop 1
	v_cndmask_b32_e32 v24, v23, v24, vcc
	v_pk_add_f32 v[28:29], v[14:15], v[24:25]
	ds_swizzle_b32 v36, v28 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v37, v29 offset:swizzle(SWAP,8)
	v_pk_mul_f32 v[14:15], v[26:27], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[34:35], v[16:17] op_sel_hi:[0,1]
	v_mul_f32_e32 v24, v4, v22
	ds_swizzle_b32 v25, v51 offset:swizzle(SWAP,4)
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[16:17], v[28:29], v[36:37]
	v_pk_mul_f32 v[28:29], v[14:15], v[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[12:13], v[2:3] op_sel_hi:[1,0]
	ds_swizzle_b32 v22, v16 offset:swizzle(SWAP,4)
	v_pk_add_f32 v[28:29], v[28:29], v[36:37]
	ds_swizzle_b32 v23, v17 offset:swizzle(SWAP,4)
	v_mov_b32_e32 v37, v29
	v_mov_b32_e32 v38, v29
	v_mov_b32_e32 v35, v28
	v_mov_b32_e32 v36, v28
	v_permlane32_swap_b32_e32 v37, v38
	s_nop 0
	v_permlane32_swap_b32_e32 v35, v36
	v_cmp_eq_u32_e32 vcc, v37, v29
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[16:17], v[16:17], v[22:23]
	ds_swizzle_b32 v22, v16 offset:swizzle(SWAP,2)
	v_cndmask_b32_e32 v37, v37, v38, vcc
	v_cmp_eq_u32_e32 vcc, v35, v28
	ds_swizzle_b32 v23, v17 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[40:41], v[16:17], v[22:23]
	v_cndmask_b32_e32 v36, v35, v36, vcc
	v_pk_add_f32 v[28:29], v[28:29], v[36:37]
	v_pk_mul_f32 v[22:23], v[26:27], v[18:19] op_sel_hi:[0,1]
	v_mov_b32_e32 v37, v29
	v_mov_b32_e32 v38, v29
	v_mov_b32_e32 v35, v28
	v_mov_b32_e32 v36, v28
	v_permlane16_swap_b32_e32 v37, v38
	s_nop 0
	v_permlane16_swap_b32_e32 v35, v36
	v_cmp_eq_u32_e32 vcc, v37, v29
	v_pk_mul_f32 v[18:19], v[34:35], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[34:35], v[32:33] op_sel_hi:[0,1]
	v_cndmask_b32_e32 v37, v37, v38, vcc
	v_cmp_eq_u32_e32 vcc, v35, v28
	v_pk_mul_f32 v[32:33], v[22:23], v[0:1] op_sel_hi:[1,0]
	ds_swizzle_b32 v42, v40 offset:swizzle(SWAP,1)
	v_cndmask_b32_e32 v36, v35, v36, vcc
	v_pk_add_f32 v[36:37], v[28:29], v[36:37]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,8)
	v_pk_mul_f32 v[34:35], v[18:19], v[2:3] op_sel_hi:[1,0]
	ds_swizzle_b32 v43, v41 offset:swizzle(SWAP,1)
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	v_mul_f32_e32 v28, v4, v27
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,4)
	v_mov_b32_e32 v35, v33
	v_mov_b32_e32 v29, v32
	v_mov_b32_e32 v34, v32
	s_nop 1
	v_permlane32_swap_b32_e32 v29, v34
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,2)
	v_pk_add_f32 v[26:27], v[40:41], v[42:43]
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[30:31], v[36:37], v[38:39]
	v_mov_b32_e32 v36, v33
	s_nop 1
	v_permlane32_swap_b32_e32 v35, v36
	v_cmp_eq_u32_e32 vcc, v35, v33
	v_pk_mul_f32 v[38:39], v[16:17], v[2:3] op_sel_hi:[1,0]
	ds_swizzle_b32 v41, v31 offset:swizzle(SWAP,1)
	v_cndmask_b32_e32 v35, v35, v36, vcc
	v_cmp_eq_u32_e32 vcc, v29, v32
	s_nop 1
	v_cndmask_b32_e32 v34, v29, v34, vcc
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	s_nop 0
	v_mov_b32_e32 v35, v33
	v_mov_b32_e32 v36, v33
	s_nop 1
	v_permlane16_swap_b32_e32 v35, v36
	v_cmp_eq_u32_e32 vcc, v35, v33
	v_mov_b32_e32 v29, v32
	v_mov_b32_e32 v34, v32
	v_cndmask_b32_e32 v35, v35, v36, vcc
	v_pk_mul_f32 v[36:37], v[20:21], v[0:1] op_sel_hi:[1,0]
	v_permlane16_swap_b32_e32 v29, v34
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	v_cmp_eq_u32_e32 vcc, v29, v32
	v_mov_b32_e32 v39, v37
	v_mov_b32_e32 v40, v37
	v_cndmask_b32_e32 v34, v29, v34, vcc
	v_mov_b32_e32 v29, v36
	v_mov_b32_e32 v38, v36
	v_permlane32_swap_b32_e32 v39, v40
	s_nop 0
	v_permlane32_swap_b32_e32 v29, v38
	v_cmp_eq_u32_e32 vcc, v39, v37
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,8)
	v_cndmask_b32_e32 v39, v39, v40, vcc
	v_cmp_eq_u32_e32 vcc, v29, v36
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,8)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	v_cndmask_b32_e32 v38, v29, v38, vcc
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,4)
	v_mov_b32_e32 v39, v37
	v_mov_b32_e32 v40, v37
	v_mov_b32_e32 v29, v36
	v_mov_b32_e32 v38, v36
	v_permlane16_swap_b32_e32 v39, v40
	s_nop 0
	v_permlane16_swap_b32_e32 v29, v38
	v_cmp_eq_u32_e32 vcc, v39, v37
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,4)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	v_cndmask_b32_e32 v39, v39, v40, vcc
	v_cmp_eq_u32_e32 vcc, v29, v36
	ds_swizzle_b32 v40, v30 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,2)
	v_cndmask_b32_e32 v38, v29, v38, vcc
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,8)
	s_waitcnt lgkmcnt(3)
	v_pk_add_f32 v[30:31], v[30:31], v[40:41]
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,4)
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[56:57], v[32:33], v[34:35]
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[40:41], v[36:37], v[38:39]
	v_pk_mul_f32 v[36:37], v[24:25], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[28:29], v[10:11] op_sel_hi:[0,1]
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v54, v40 offset:swizzle(SWAP,1)
	v_mov_b32_e32 v39, v37
	v_mov_b32_e32 v42, v37
	v_mov_b32_e32 v29, v36
	v_mov_b32_e32 v38, v36
	v_permlane32_swap_b32_e32 v39, v42
	s_nop 0
	v_permlane32_swap_b32_e32 v29, v38
	v_cmp_eq_u32_e32 vcc, v39, v37
	ds_swizzle_b32 v55, v41 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[58:59], v[40:41], v[54:55]
	v_cndmask_b32_e32 v39, v39, v42, vcc
	v_cmp_eq_u32_e32 vcc, v29, v36
	v_pk_mul_f32 v[40:41], v[24:25], v[22:23] op_sel_hi:[0,1]
	s_nop 0
	v_cndmask_b32_e32 v38, v29, v38, vcc
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	s_nop 0
	v_mov_b32_e32 v39, v37
	v_mov_b32_e32 v42, v37
	v_mov_b32_e32 v29, v36
	v_mov_b32_e32 v38, v36
	v_permlane16_swap_b32_e32 v39, v42
	s_nop 0
	v_permlane16_swap_b32_e32 v29, v38
	v_cmp_eq_u32_e32 vcc, v39, v37
	s_nop 1
	v_cndmask_b32_e32 v39, v39, v42, vcc
	v_cmp_eq_u32_e32 vcc, v29, v36
	s_nop 1
	v_cndmask_b32_e32 v38, v29, v38, vcc
	v_pk_add_f32 v[42:43], v[36:37], v[38:39]
	v_pk_mul_f32 v[36:37], v[24:25], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[28:29], v[12:13] op_sel_hi:[0,1]
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v44, v42 offset:swizzle(SWAP,8)
	v_mov_b32_e32 v39, v37
	v_mov_b32_e32 v46, v37
	v_mov_b32_e32 v29, v36
	v_mov_b32_e32 v38, v36
	v_permlane32_swap_b32_e32 v39, v46
	s_nop 0
	v_permlane32_swap_b32_e32 v29, v38
	v_cmp_eq_u32_e32 vcc, v39, v37
	ds_swizzle_b32 v45, v43 offset:swizzle(SWAP,8)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[42:43], v[42:43], v[44:45]
	v_cndmask_b32_e32 v39, v39, v46, vcc
	v_cmp_eq_u32_e32 vcc, v29, v36
	ds_swizzle_b32 v44, v42 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v45, v43 offset:swizzle(SWAP,4)
	v_cndmask_b32_e32 v38, v29, v38, vcc
	v_pk_add_f32 v[46:47], v[36:37], v[38:39]
	buffer_load_dwordx4 v[36:39], v50, s[4:7], 0 offen
	v_mov_b32_e32 v29, v46
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[42:43], v[42:43], v[44:45]
	ds_swizzle_b32 v44, v42 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v45, v43 offset:swizzle(SWAP,2)
	v_mov_b32_e32 v5, v46
	s_nop 1
	v_permlane16_swap_b32_e32 v29, v5
	v_mov_b32_e32 v52, v47
	v_mov_b32_e32 v53, v47
	s_nop 1
	v_permlane16_swap_b32_e32 v52, v53
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[32:33], v[42:43], v[44:45]
	v_pk_mul_f32 v[42:43], v[28:29], v[18:19] op_sel_hi:[0,1]
	v_cmp_eq_u32_e32 vcc, v52, v47
	v_pk_add_f32 v[40:41], v[40:41], v[42:43]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v45, 0xffff0000, v37
	v_cndmask_b32_e32 v53, v52, v53, vcc
	v_cmp_eq_u32_e32 vcc, v29, v46
	v_mov_b32_e32 v42, v41
	v_mov_b32_e32 v43, v41
	v_cndmask_b32_e32 v52, v29, v5, vcc
	v_add_f32_e32 v5, v51, v25
	v_mov_b32_e32 v25, v40
	v_mov_b32_e32 v29, v40
	v_permlane32_swap_b32_e32 v42, v43
	s_nop 0
	v_permlane32_swap_b32_e32 v25, v29
	v_cmp_eq_u32_e32 vcc, v42, v41
	v_pk_add_f32 v[46:47], v[46:47], v[52:53]
	ds_swizzle_b32 v52, v46 offset:swizzle(SWAP,8)
	v_cndmask_b32_e32 v43, v42, v43, vcc
	v_cmp_eq_u32_e32 vcc, v25, v40
	ds_swizzle_b32 v53, v47 offset:swizzle(SWAP,8)
	v_lshlrev_b32_e32 v44, 16, v37
	v_cndmask_b32_e32 v42, v25, v29, vcc
	v_pk_add_f32 v[40:41], v[40:41], v[42:43]
	v_and_b32_e32 v61, 0xffff0000, v38
	v_mov_b32_e32 v42, v41
	v_mov_b32_e32 v43, v41
	v_mov_b32_e32 v25, v40
	v_mov_b32_e32 v29, v40
	v_permlane16_swap_b32_e32 v42, v43
	s_nop 0
	v_permlane16_swap_b32_e32 v25, v29
	v_cmp_eq_u32_e32 vcc, v42, v41
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[46:47], v[46:47], v[52:53]
	ds_swizzle_b32 v52, v46 offset:swizzle(SWAP,4)
	v_cndmask_b32_e32 v43, v42, v43, vcc
	v_cmp_eq_u32_e32 vcc, v25, v40
	ds_swizzle_b32 v53, v47 offset:swizzle(SWAP,4)
	v_lshlrev_b32_e32 v60, 16, v38
	v_cndmask_b32_e32 v42, v25, v29, vcc
	v_pk_mul_f32 v[24:25], v[24:25], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[28:29], v[16:17] op_sel_hi:[0,1]
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_pk_add_f32 v[40:41], v[40:41], v[42:43]
	v_mov_b32_e32 v29, v25
	v_mov_b32_e32 v43, v25
	v_mov_b32_e32 v28, v24
	v_mov_b32_e32 v42, v24
	v_permlane32_swap_b32_e32 v29, v43
	s_nop 0
	v_permlane32_swap_b32_e32 v28, v42
	v_cmp_eq_u32_e32 vcc, v29, v25
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[46:47], v[46:47], v[52:53]
	ds_swizzle_b32 v52, v46 offset:swizzle(SWAP,2)
	v_cndmask_b32_e32 v29, v29, v43, vcc
	v_cmp_eq_u32_e32 vcc, v28, v24
	ds_swizzle_b32 v53, v47 offset:swizzle(SWAP,2)
	v_pk_add_f32 v[30:31], v[44:45], v[30:31] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e32 v28, v28, v42, vcc
	v_pk_add_f32 v[24:25], v[24:25], v[28:29]
	v_and_b32_e32 v29, 0xffff0000, v36
	v_mov_b32_e32 v37, v25
	v_mov_b32_e32 v38, v25
	v_mov_b32_e32 v42, v24
	v_lshlrev_b32_e32 v28, 16, v36
	v_mov_b32_e32 v36, v24
	v_permlane16_swap_b32_e32 v37, v38
	s_nop 0
	v_permlane16_swap_b32_e32 v42, v36
	v_cmp_eq_u32_e32 vcc, v37, v25
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[34:35], v[46:47], v[52:53]
	v_and_b32_e32 v53, 0xffff0000, v39
	v_cndmask_b32_e32 v37, v37, v38, vcc
	v_cmp_eq_u32_e32 vcc, v42, v24
	v_lshlrev_b32_e32 v52, 16, v39
	ds_swizzle_b32 v38, v40 offset:swizzle(SWAP,8)
	v_cndmask_b32_e32 v36, v42, v36, vcc
	v_pk_add_f32 v[24:25], v[24:25], v[36:37]
	ds_swizzle_b32 v39, v41 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v36, v24 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v37, v25 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v42, v5 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v46, v34 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(4)
	v_pk_add_f32 v[38:39], v[40:41], v[38:39]
	ds_swizzle_b32 v40, v38 offset:swizzle(SWAP,4)
	s_waitcnt lgkmcnt(3)
	v_pk_add_f32 v[24:25], v[24:25], v[36:37]
	ds_swizzle_b32 v41, v39 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v36, v24 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v37, v25 offset:swizzle(SWAP,4)
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v54, v5, v42
	ds_swizzle_b32 v5, v1 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(3)
	v_pk_add_f32 v[38:39], v[38:39], v[40:41]
	ds_swizzle_b32 v40, v38 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[24:25], v[24:25], v[36:37]
	ds_swizzle_b32 v41, v39 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v42, v24 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v43, v25 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v55, v54 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v47, v35 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(4)
	v_pk_add_f32 v[36:37], v[38:39], v[40:41]
	ds_swizzle_b32 v40, v36 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(3)
	v_pk_add_f32 v[38:39], v[24:25], v[42:43]
	ds_swizzle_b32 v42, v32 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v43, v33 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v41, v37 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v44, v38 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v45, v39 offset:swizzle(SWAP,1)
	v_add_u32_e32 v51, 0x200, v48
	v_pk_add_f32 v[28:29], v[28:29], v[26:27] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25], v[60:61], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[26:27], v[52:53], v[58:59] neg_lo:[0,1] neg_hi:[0,1]
	v_or_b32_e32 v53, 16, v48
	v_add_u32_e32 v52, 0x210, v48
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_2
	v_add_f32_e32 v3, v7, v49
	s_mov_b32 s4, 0xf800000
	v_mul_f32_e32 v7, 0x4f800000, v3
	v_cmp_gt_f32_e32 vcc, s4, v3
	v_add_f32_e32 v1, v1, v5
	v_mul_f32_e32 v5, 0x4f800000, v1
	v_cndmask_b32_e32 v3, v3, v7, vcc
	v_sqrt_f32_e32 v7, v3
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v54, v54, v55
	s_load_dwordx2 s[8:9], s[0:1], 0xd0
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[34:35], v[34:35], v[46:47]
	v_add_u32_e32 v49, -1, v7
	v_fma_f32 v56, -v49, v7, v3
	v_cmp_ge_f32_e64 s[2:3], 0, v56
	v_add_u32_e32 v56, 1, v7
	v_pk_add_f32 v[32:33], v[32:33], v[42:43]
	v_cndmask_b32_e64 v49, v7, v49, s[2:3]
	v_fma_f32 v7, -v56, v7, v3
	v_cmp_lt_f32_e64 s[2:3], 0, v7
	s_and_b32 s9, s9, 0xffff
	v_or_b32_e32 v57, 8, v50
	v_cndmask_b32_e64 v7, v49, v56, s[2:3]
	v_mul_f32_e32 v49, 0x37800000, v7
	v_cndmask_b32_e32 v7, v7, v49, vcc
	v_mov_b32_e32 v49, 0x260
	v_cmp_class_f32_e32 vcc, v3, v49
	s_nop 1
	v_cndmask_b32_e32 v3, v7, v3, vcc
	v_div_scale_f32 v7, s[2:3], v3, v3, 1.0
	v_rcp_f32_e32 v56, v7
	v_cmp_gt_f32_e64 s[2:3], s4, v1
	v_fma_f32 v55, -v7, v56, 1.0
	s_nop 0
	v_cndmask_b32_e64 v1, v1, v5, s[2:3]
	v_fmac_f32_e32 v56, v55, v56
	v_div_scale_f32 v55, vcc, 1.0, v3, 1.0
	v_sqrt_f32_e32 v5, v1
	v_mul_f32_e32 v58, v55, v56
	v_fma_f32 v59, -v7, v58, v55
	v_fmac_f32_e32 v58, v59, v56
	v_fma_f32 v7, -v7, v58, v55
	v_add_u32_e32 v55, -1, v5
	v_fma_f32 v59, -v55, v5, v1
	v_cmp_ge_f32_e64 s[4:5], 0, v59
	v_add_u32_e32 v59, 1, v5
	v_div_fmas_f32 v7, v7, v56, v58
	v_cndmask_b32_e64 v55, v5, v55, s[4:5]
	v_fma_f32 v5, -v59, v5, v1
	v_cmp_lt_f32_e64 s[4:5], 0, v5
	v_div_fixup_f32 v3, v7, v3, 1.0
	v_mul_f32_e32 v7, v3, v54
	v_cndmask_b32_e64 v5, v55, v59, s[4:5]
	v_mul_f32_e32 v55, 0x37800000, v5
	v_cndmask_b32_e64 v5, v5, v55, s[2:3]
	v_cmp_class_f32_e64 s[2:3], v1, v49
	s_nop 1
	v_cndmask_b32_e64 v1, v5, v1, s[2:3]
	v_div_scale_f32 v5, s[2:3], v1, v1, 1.0
	v_rcp_f32_e32 v49, v5
	s_nop 0
	v_fma_f32 v3, -v5, v49, 1.0
	v_fmac_f32_e32 v49, v3, v49
	v_div_scale_f32 v3, vcc, 1.0, v1, 1.0
	v_mul_f32_e32 v54, v3, v49
	v_fma_f32 v55, -v5, v54, v3
	v_fmac_f32_e32 v54, v55, v49
	v_fma_f32 v3, -v5, v54, v3
	v_div_fmas_f32 v3, v3, v49, v54
	v_div_fixup_f32 v5, v3, v1, 1.0
	v_pk_mul_f32 v[4:5], v[4:5], v[6:7]
	s_nop 0
	v_mul_f32_e32 v4, v4, v5
	v_pk_mul_f32 v[42:43], v[4:5], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[4:5], v[28:29] op_sel_hi:[0,1]
	v_pk_add_f32 v[32:33], v[32:33], v[46:47]
	v_pk_add_f32 v[34:35], v[34:35], v[42:43]
	s_nop 0
	v_cvt_pk_bf16_f32 v35, v34, v35
	v_cvt_pk_bf16_f32 v34, v32, v33
	buffer_store_dwordx2 v[34:35], v50, s[8:11], 0 offen
	v_pk_add_f32 v[32:33], v[38:39], v[44:45]
	v_pk_add_f32 v[34:35], v[36:37], v[40:41]
	v_pk_mul_f32 v[36:37], v[4:5], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[4:5], v[24:25] op_sel_hi:[0,1]
	v_pk_add_f32 v[4:5], v[34:35], v[4:5]
	v_pk_add_f32 v[32:33], v[32:33], v[36:37]
	s_nop 0
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v4, v5
	buffer_store_dwordx2 v[32:33], v57, s[8:11], 0 offen
.LBB0_2:
	s_or_b64 exec, exec, s[6:7]
	s_load_dwordx2 s[8:9], s[0:1], 0xa0
	v_mul_f32_e32 v4, v2, v6
	v_mul_f32_e32 v6, v0, v6
	v_pk_mul_f32 v[0:1], v[6:7], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[6:7], v[30:31] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	s_and_b32 s9, s9, 0xffff
	v_pk_add_f32 v[2:3], v[14:15], v[2:3]
	v_pk_add_f32 v[0:1], v[8:9], v[0:1]
	buffer_store_dwordx4 v[0:3], v48, s[8:11], 0 offen
	s_nop 1
	v_pk_mul_f32 v[0:1], v[4:5], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[4:5], v[30:31] op_sel_hi:[0,1]
	v_pk_add_f32 v[2:3], v[12:13], v[2:3]
	v_pk_add_f32 v[0:1], v[10:11], v[0:1]
	buffer_store_dwordx4 v[0:3], v51, s[8:11], 0 offen
	s_nop 1
	v_pk_mul_f32 v[0:1], v[6:7], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[6:7], v[26:27] op_sel_hi:[0,1]
	v_pk_add_f32 v[2:3], v[20:21], v[2:3]
	v_pk_add_f32 v[0:1], v[22:23], v[0:1]
	buffer_store_dwordx4 v[0:3], v53, s[8:11], 0 offen
	s_nop 1
	v_pk_mul_f32 v[0:1], v[4:5], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[4:5], v[26:27] op_sel_hi:[0,1]
	v_pk_add_f32 v[2:3], v[16:17], v[2:3]
	v_pk_add_f32 v[0:1], v[18:19], v[0:1]
	buffer_store_dwordx4 v[0:3], v52, s[8:11], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kda_decode_step_kern_0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 268
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 62
		.amdhsa_next_free_sgpr 25
		.amdhsa_accum_offset 64
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	kda_decode_step_kern_0, .Lfunc_end0-kda_decode_step_kern_0

	.set kda_decode_step_kern_0.num_vgpr, 62
	.set kda_decode_step_kern_0.num_agpr, 0
	.set kda_decode_step_kern_0.numbered_sgpr, 25
	.set kda_decode_step_kern_0.num_named_barrier, 0
	.set kda_decode_step_kern_0.private_seg_size, 0
	.set kda_decode_step_kern_0.uses_vcc, 1
	.set kda_decode_step_kern_0.uses_flat_scratch, 0
	.set kda_decode_step_kern_0.has_dyn_sized_stack, 0
	.set kda_decode_step_kern_0.has_recursion, 0
	.set kda_decode_step_kern_0.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .offset:         80
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .offset:         120
        .size:           40
        .value_kind:     by_value
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     global_buffer
      - .offset:         168
        .size:           40
        .value_kind:     by_value
      - .address_space:  global
        .offset:         208
        .size:           8
        .value_kind:     global_buffer
      - .offset:         216
        .size:           40
        .value_kind:     by_value
      - .address_space:  global
        .offset:         256
        .size:           8
        .value_kind:     global_buffer
      - .offset:         264
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 268
    .max_flat_workgroup_size: 256
    .name:           kda_decode_step_kern_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         kda_decode_step_kern_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     62
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
