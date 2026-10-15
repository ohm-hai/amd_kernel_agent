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
	s_sub_i32 s21, s2, s3
	s_cmp_lg_u32 s2, s3
	s_cselect_b64 s[6:7], -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b64 s[2:3], -1, 0
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x18
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	s_and_b64 s[2:3], s[2:3], s[6:7]
	s_subb_u32 s2, s10, 0
	v_and_b32_e32 v3, 63, v0
	s_lshl_b32 s20, s2, 8
	s_waitcnt lgkmcnt(0)
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	v_lshl_or_b32 v1, v3, 2, s20
	s_and_b32 s17, s17, 0xffff
	s_mov_b32 s18, s6
	s_mov_b32 s19, s7
	s_and_b32 s13, s13, 0xffff
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	buffer_load_ushort v2, v1, s[4:7], 0 offen
	buffer_load_ushort v4, v1, s[4:7], 0 offen offset:2
	buffer_load_ushort v5, v1, s[16:19], 0 offen
	buffer_load_ushort v6, v1, s[16:19], 0 offen offset:2
	buffer_load_ushort v7, v1, s[12:15], 0 offen
	buffer_load_ushort v9, v1, s[12:15], 0 offen offset:2
	s_load_dwordx2 s[24:25], s[0:1], 0x60
	s_load_dwordx2 s[12:13], s[0:1], 0x70
	s_load_dwordx2 s[16:17], s[0:1], 0x100
	s_lshl_b32 s3, s2, 1
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	s_waitcnt lgkmcnt(0)
	s_and_b32 s25, s25, 0xffff
	v_mov_b32_e32 v1, s3
	buffer_load_ushort v12, v1, s[24:27], 0 offen
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_b32_e32 v0, 24, v0
	v_lshl_or_b32 v21, s21, 5, v0
	s_mov_b32 s5, 0xf800000
	s_lshl_b32 s4, s2, 14
	s_and_b32 s13, s13, 0xffff
	v_mov_b32_e32 v24, 0x260
	s_and_b32 s17, s17, 0xffff
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_and_b32 s9, s9, 0xffff
	v_lshl_add_u32 v51, v21, 1, s20
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v8, 16, v2
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v18, 16, v4
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v0, 16, v5
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v19, 16, v6
	v_mul_f32_e32 v2, v8, v8
	v_mul_f32_e32 v5, v18, v18
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v22, 16, v9
	v_mul_f32_e32 v9, v0, v0
	v_mul_f32_e32 v1, v19, v19
	v_add_f32_e32 v2, v2, v5
	v_pk_add_f32 v[10:11], v[8:9], v[0:1]
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v9, v11
	v_mov_b32_e32 v10, v11
	v_permlane32_swap_b32_e32 v1, v5
	s_nop 0
	v_permlane32_swap_b32_e32 v9, v10
	v_cmp_eq_u32_e32 vcc, v1, v2
	v_lshlrev_b32_e32 v20, 16, v7
	v_mul_f32_e32 v4, v18, v19
	v_cndmask_b32_e32 v1, v1, v5, vcc
	v_cmp_eq_u32_e32 vcc, v9, v11
	v_mul_f32_e32 v6, v8, v0
	v_mov_b32_e32 v7, v11
	v_cndmask_b32_e32 v5, v9, v10, vcc
	v_add_f32_e32 v1, v2, v1
	v_pk_add_f32 v[4:5], v[6:7], v[4:5]
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v6, v1
	s_nop 1
	v_permlane16_swap_b32_e32 v2, v6
	v_cmp_eq_u32_e32 vcc, v2, v1
	v_mov_b32_e32 v7, v5
	v_mov_b32_e32 v9, v5
	v_cndmask_b32_e32 v2, v2, v6, vcc
	v_add_f32_e32 v1, v1, v2
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,8)
	v_permlane16_swap_b32_e32 v7, v9
	v_cmp_eq_u32_e32 vcc, v7, v5
	v_mov_b32_e32 v10, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v2
	v_cndmask_b32_e32 v6, v7, v9, vcc
	v_add_f32_e32 v5, v5, v6
	ds_swizzle_b32 v6, v5 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,4)
	v_mov_b32_e32 v11, v4
	s_nop 1
	v_permlane32_swap_b32_e32 v10, v11
	v_cmp_eq_u32_e32 vcc, v10, v4
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v5, v5, v6
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v2
	v_cndmask_b32_e32 v7, v10, v11, vcc
	ds_swizzle_b32 v6, v5 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,2)
	v_add_f32_e32 v4, v4, v7
	v_mov_b32_e32 v7, v4
	v_mov_b32_e32 v9, v4
	s_nop 1
	v_permlane16_swap_b32_e32 v7, v9
	v_cmp_eq_u32_e32 vcc, v7, v4
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v5, v6
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v2
	v_cndmask_b32_e32 v5, v7, v9, vcc
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,1)
	v_add_f32_e32 v4, v4, v5
	ds_swizzle_b32 v7, v4 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v10, v6 offset:swizzle(SWAP,2)
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v5, 16, v12
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v1, v1, v2
	v_cmp_gt_f32_e32 vcc, s5, v1
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v4, v4, v7
	v_mul_f32_e32 v7, 0x4f800000, v1
	v_cndmask_b32_e32 v1, v1, v7, vcc
	v_sqrt_f32_e32 v7, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v6, v10
	ds_swizzle_b32 v6, v4 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v23, v2 offset:swizzle(SWAP,1)
	v_add_u32_e32 v10, -1, v7
	v_fma_f32 v11, -v10, v7, v1
	v_cmp_ge_f32_e64 s[2:3], 0, v11
	v_add_u32_e32 v11, 1, v7
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v4, v4, v6
	v_cndmask_b32_e64 v10, v7, v10, s[2:3]
	v_fma_f32 v7, -v11, v7, v1
	v_cmp_lt_f32_e64 s[2:3], 0, v7
	ds_swizzle_b32 v6, v4 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v2, v2, v23
	v_cndmask_b32_e64 v7, v10, v11, s[2:3]
	v_mul_f32_e32 v10, 0x37800000, v7
	v_cndmask_b32_e32 v7, v7, v10, vcc
	v_lshl_add_u32 v10, v3, 8, v21
	v_add_lshl_u32 v44, v10, s4, 2
	buffer_load_dwordx4 v[10:13], v44, s[12:15], 0 offen
	buffer_load_dwordx4 v[14:17], v44, s[12:15], 0 offen offset:512
	v_cmp_class_f32_e32 vcc, v1, v24
	v_mul_f32_e32 v23, 0x4f800000, v2
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v45, v4, v6
	v_cndmask_b32_e32 v1, v7, v1, vcc
	v_div_scale_f32 v7, s[2:3], v1, v1, 1.0
	v_rcp_f32_e32 v25, v7
	v_cmp_gt_f32_e64 s[2:3], s5, v2
	buffer_load_dword v9, off, s[16:19], 0
	buffer_load_dwordx4 v[28:31], v44, s[12:15], 0 offen offset:528
	v_fma_f32 v4, -v7, v25, 1.0
	v_cndmask_b32_e64 v2, v2, v23, s[2:3]
	v_fmac_f32_e32 v25, v4, v25
	v_div_scale_f32 v4, vcc, 1.0, v1, 1.0
	v_sqrt_f32_e32 v23, v2
	v_mul_f32_e32 v6, v4, v25
	v_fma_f32 v26, -v7, v6, v4
	v_fmac_f32_e32 v6, v26, v25
	v_fma_f32 v4, -v7, v6, v4
	v_add_u32_e32 v7, -1, v23
	v_fma_f32 v26, -v7, v23, v2
	v_cmp_ge_f32_e64 s[4:5], 0, v26
	v_add_u32_e32 v26, 1, v23
	v_div_fmas_f32 v4, v4, v25, v6
	v_cndmask_b32_e64 v7, v23, v7, s[4:5]
	v_fma_f32 v23, -v26, v23, v2
	v_cmp_lt_f32_e64 s[4:5], 0, v23
	v_div_fixup_f32 v47, v4, v1, 1.0
	v_mul_f32_e32 v8, v47, v8
	v_cndmask_b32_e64 v7, v7, v26, s[4:5]
	v_mul_f32_e32 v23, 0x37800000, v7
	v_cndmask_b32_e64 v7, v7, v23, s[2:3]
	v_cmp_class_f32_e64 s[2:3], v2, v24
	buffer_load_dwordx4 v[24:27], v44, s[12:15], 0 offen offset:16
	s_mov_b32 s4, 0x42b17218
	v_cndmask_b32_e64 v2, v7, v2, s[2:3]
	v_div_scale_f32 v7, s[2:3], v2, v2, 1.0
	v_rcp_f32_e32 v23, v7
	s_mov_b32 s2, 0x3fb8aa3b
	s_mov_b32 s3, 0xc2ce8ed0
	v_mul_f32_e32 v18, v47, v18
	v_fma_f32 v1, -v7, v23, 1.0
	v_fmac_f32_e32 v23, v1, v23
	v_div_scale_f32 v1, vcc, 1.0, v2, 1.0
	v_mul_f32_e32 v4, v1, v23
	v_fma_f32 v6, -v7, v4, v1
	v_fmac_f32_e32 v4, v6, v23
	v_fma_f32 v1, -v7, v4, v1
	v_div_fmas_f32 v1, v1, v23, v4
	v_mul_f32_e32 v4, 0x3fb8aa3b, v20
	v_fma_f32 v6, v20, s2, -v4
	v_rndne_f32_e32 v7, v4
	v_fmac_f32_e32 v6, 0x32a5705f, v20
	v_sub_f32_e32 v4, v4, v7
	v_add_f32_e32 v4, v4, v6
	v_exp_f32_e32 v6, v4
	v_cvt_i32_f32_e32 v7, v7
	v_div_fixup_f32 v49, v1, v2, 1.0
	v_mul_f32_e32 v1, 0x3fb8aa3b, v22
	v_mul_f32_e32 v2, v49, v0
	v_ldexp_f32 v0, v6, v7
	v_fma_f32 v6, v22, s2, -v1
	v_rndne_f32_e32 v7, v1
	v_fmac_f32_e32 v6, 0x32a5705f, v22
	v_sub_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v6
	v_exp_f32_e32 v1, v1
	v_cvt_i32_f32_e32 v6, v7
	v_cmp_ngt_f32_e32 vcc, s3, v20
	v_mov_b32_e32 v7, 0x7f800000
	v_mul_f32_e32 v4, v49, v19
	v_cndmask_b32_e32 v0, 0, v0, vcc
	v_cmp_nlt_f32_e32 vcc, s4, v20
	ds_swizzle_b32 v46, v45 offset:swizzle(SWAP,1)
	v_add_u32_e32 v50, 0x200, v44
	v_cndmask_b32_e32 v32, v7, v0, vcc
	v_ldexp_f32 v0, v1, v6
	v_cmp_ngt_f32_e32 vcc, s3, v22
	v_or_b32_e32 v48, 16, v44
	s_nop 0
	v_cndmask_b32_e32 v0, 0, v0, vcc
	v_cmp_nlt_f32_e32 vcc, s4, v22
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v22, v9, v18
	v_cndmask_b32_e32 v34, v7, v0, vcc
	v_pk_mul_f32 v[6:7], v[32:33], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[34:35], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[6:7], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[0:1], v[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[34:35], v[16:17] op_sel_hi:[0,1]
	v_pk_add_f32 v[10:11], v[10:11], v[14:15]
	s_nop 0
	v_mov_b32_e32 v15, v11
	v_mov_b32_e32 v20, v11
	v_mov_b32_e32 v14, v10
	v_mov_b32_e32 v19, v10
	v_permlane32_swap_b32_e32 v15, v20
	s_nop 0
	v_permlane32_swap_b32_e32 v14, v19
	v_cmp_eq_u32_e32 vcc, v15, v11
	s_nop 1
	v_cndmask_b32_e32 v15, v15, v20, vcc
	v_cmp_eq_u32_e32 vcc, v14, v10
	s_nop 1
	v_cndmask_b32_e32 v14, v14, v19, vcc
	v_pk_add_f32 v[10:11], v[10:11], v[14:15]
	s_nop 0
	v_mov_b32_e32 v15, v11
	v_mov_b32_e32 v20, v11
	v_mov_b32_e32 v14, v10
	v_mov_b32_e32 v19, v10
	v_permlane16_swap_b32_e32 v15, v20
	s_nop 0
	v_permlane16_swap_b32_e32 v14, v19
	v_cmp_eq_u32_e32 vcc, v15, v11
	s_nop 1
	v_cndmask_b32_e32 v15, v15, v20, vcc
	v_cmp_eq_u32_e32 vcc, v14, v10
	v_mul_f32_e32 v20, v9, v8
	s_nop 0
	v_cndmask_b32_e32 v14, v14, v19, vcc
	v_pk_add_f32 v[10:11], v[10:11], v[14:15]
	ds_swizzle_b32 v14, v10 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v15, v11 offset:swizzle(SWAP,8)
	v_pk_mul_f32 v[18:19], v[32:33], v[12:13] op_sel_hi:[0,1]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[12:13], v[32:33], v[26:27] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[8:9], v[10:11], v[14:15]
	ds_swizzle_b32 v10, v8 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v11, v9 offset:swizzle(SWAP,4)
	v_pk_mul_f32 v[14:15], v[32:33], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[14:15], v[2:3] op_sel_hi:[1,0]
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[36:37], v[8:9], v[10:11]
	v_pk_mul_f32 v[8:9], v[34:35], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[34:35], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[18:19], v[2:3] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[16:17], v[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[4:5], v[10:11] op_sel_hi:[0,1]
	v_pk_add_f32 v[28:29], v[28:29], v[30:31]
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	v_mov_b32_e32 v30, v29
	v_mov_b32_e32 v31, v29
	v_mov_b32_e32 v23, v28
	v_mov_b32_e32 v27, v28
	v_permlane32_swap_b32_e32 v30, v31
	s_nop 0
	v_permlane32_swap_b32_e32 v23, v27
	v_cmp_eq_u32_e32 vcc, v30, v29
	v_mov_b32_e32 v34, v33
	v_mov_b32_e32 v35, v33
	v_cndmask_b32_e32 v31, v30, v31, vcc
	v_cmp_eq_u32_e32 vcc, v23, v28
	v_permlane32_swap_b32_e32 v34, v35
	s_nop 0
	v_cndmask_b32_e32 v30, v23, v27, vcc
	v_pk_add_f32 v[28:29], v[28:29], v[30:31]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,2)
	v_mov_b32_e32 v30, v29
	v_mov_b32_e32 v31, v29
	v_mov_b32_e32 v23, v28
	v_mov_b32_e32 v27, v28
	v_permlane16_swap_b32_e32 v30, v31
	s_nop 0
	v_permlane16_swap_b32_e32 v23, v27
	v_cmp_eq_u32_e32 vcc, v30, v29
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[24:25], v[36:37], v[38:39]
	v_cndmask_b32_e32 v31, v30, v31, vcc
	v_cmp_eq_u32_e32 vcc, v23, v28
	ds_swizzle_b32 v26, v24 offset:swizzle(SWAP,1)
	v_pk_mul_f32 v[36:37], v[4:5], v[8:9] op_sel_hi:[0,1]
	v_cndmask_b32_e32 v30, v23, v27, vcc
	v_mov_b32_e32 v23, v32
	v_mov_b32_e32 v27, v32
	s_nop 1
	v_permlane32_swap_b32_e32 v23, v27
	v_cmp_eq_u32_e32 vcc, v34, v33
	v_pk_add_f32 v[28:29], v[28:29], v[30:31]
	ds_swizzle_b32 v30, v28 offset:swizzle(SWAP,8)
	v_cndmask_b32_e32 v35, v34, v35, vcc
	v_cmp_eq_u32_e32 vcc, v23, v32
	ds_swizzle_b32 v31, v29 offset:swizzle(SWAP,8)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[28:29], v[28:29], v[30:31]
	v_cndmask_b32_e32 v34, v23, v27, vcc
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	ds_swizzle_b32 v30, v28 offset:swizzle(SWAP,4)
	v_mov_b32_e32 v34, v33
	v_mov_b32_e32 v35, v33
	v_mov_b32_e32 v23, v32
	v_mov_b32_e32 v27, v32
	v_permlane16_swap_b32_e32 v34, v35
	s_nop 0
	v_permlane16_swap_b32_e32 v23, v27
	v_cmp_eq_u32_e32 vcc, v34, v33
	ds_swizzle_b32 v31, v29 offset:swizzle(SWAP,4)
	s_nop 0
	v_cndmask_b32_e32 v35, v34, v35, vcc
	v_cmp_eq_u32_e32 vcc, v23, v32
	s_nop 1
	v_cndmask_b32_e32 v34, v23, v27, vcc
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v27, v25 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,4)
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[24:25], v[24:25], v[26:27]
	v_pk_add_f32 v[26:27], v[28:29], v[30:31]
	ds_swizzle_b32 v28, v26 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v29, v27 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[30:31], v[32:33], v[34:35]
	v_pk_mul_f32 v[34:35], v[12:13], v[2:3] op_sel_hi:[1,0]
	ds_swizzle_b32 v32, v30 offset:swizzle(SWAP,2)
	v_pk_add_f32 v[34:35], v[34:35], v[36:37]
	ds_swizzle_b32 v33, v31 offset:swizzle(SWAP,2)
	v_mov_b32_e32 v37, v35
	v_mov_b32_e32 v38, v35
	v_mov_b32_e32 v23, v34
	v_mov_b32_e32 v36, v34
	v_permlane32_swap_b32_e32 v37, v38
	s_nop 0
	v_permlane32_swap_b32_e32 v23, v36
	v_cmp_eq_u32_e32 vcc, v37, v35
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[30:31], v[30:31], v[32:33]
	ds_swizzle_b32 v41, v31 offset:swizzle(SWAP,1)
	v_cndmask_b32_e32 v37, v37, v38, vcc
	v_cmp_eq_u32_e32 vcc, v23, v34
	v_pk_add_f32 v[26:27], v[26:27], v[28:29]
	ds_swizzle_b32 v28, v26 offset:swizzle(SWAP,1)
	v_cndmask_b32_e32 v36, v23, v36, vcc
	v_pk_add_f32 v[34:35], v[34:35], v[36:37]
	ds_swizzle_b32 v29, v27 offset:swizzle(SWAP,1)
	v_mov_b32_e32 v37, v35
	v_mov_b32_e32 v38, v35
	v_mov_b32_e32 v23, v34
	v_mov_b32_e32 v36, v34
	v_permlane16_swap_b32_e32 v37, v38
	s_nop 0
	v_permlane16_swap_b32_e32 v23, v36
	v_cmp_eq_u32_e32 vcc, v37, v35
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[28:29], v[26:27], v[28:29]
	v_cndmask_b32_e32 v37, v37, v38, vcc
	v_cmp_eq_u32_e32 vcc, v23, v34
	v_pk_mul_f32 v[38:39], v[22:23], v[0:1] op_sel_hi:[0,1]
	s_nop 0
	v_cndmask_b32_e32 v36, v23, v36, vcc
	v_pk_add_f32 v[34:35], v[34:35], v[36:37]
	ds_swizzle_b32 v36, v34 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v37, v35 offset:swizzle(SWAP,8)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[32:33], v[34:35], v[36:37]
	v_pk_mul_f32 v[36:37], v[20:21], v[6:7] op_sel_hi:[0,1]
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,4)
	v_mov_b32_e32 v39, v37
	v_mov_b32_e32 v40, v37
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v38, v36
	v_permlane32_swap_b32_e32 v39, v40
	s_nop 0
	v_permlane32_swap_b32_e32 v23, v38
	v_cmp_eq_u32_e32 vcc, v39, v37
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,4)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	v_cndmask_b32_e32 v39, v39, v40, vcc
	v_cmp_eq_u32_e32 vcc, v23, v36
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,2)
	v_cndmask_b32_e32 v38, v23, v38, vcc
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[42:43], v[32:33], v[34:35]
	v_mov_b32_e32 v39, v37
	v_mov_b32_e32 v40, v37
	v_mov_b32_e32 v23, v36
	v_mov_b32_e32 v38, v36
	v_permlane16_swap_b32_e32 v39, v40
	s_nop 0
	v_permlane16_swap_b32_e32 v23, v38
	v_cmp_eq_u32_e32 vcc, v39, v37
	v_pk_mul_f32 v[32:33], v[20:21], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[22:23], v[16:17] op_sel_hi:[0,1]
	v_cndmask_b32_e32 v39, v39, v40, vcc
	v_cmp_eq_u32_e32 vcc, v23, v36
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	ds_swizzle_b32 v40, v30 offset:swizzle(SWAP,1)
	v_cndmask_b32_e32 v38, v23, v38, vcc
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,8)
	v_mov_b32_e32 v35, v33
	v_mov_b32_e32 v23, v32
	v_mov_b32_e32 v34, v32
	s_nop 1
	v_permlane32_swap_b32_e32 v23, v34
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	ds_swizzle_b32 v38, v36 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v39, v37 offset:swizzle(SWAP,4)
	v_pk_add_f32 v[58:59], v[30:31], v[40:41]
	ds_swizzle_b32 v52, v42 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v53, v43 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[36:37], v[36:37], v[38:39]
	v_mov_b32_e32 v38, v33
	s_nop 1
	v_permlane32_swap_b32_e32 v35, v38
	v_cmp_eq_u32_e32 vcc, v35, v33
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[60:61], v[42:43], v[52:53]
	ds_swizzle_b32 v56, v36 offset:swizzle(SWAP,2)
	v_cndmask_b32_e32 v35, v35, v38, vcc
	v_cmp_eq_u32_e32 vcc, v23, v32
	ds_swizzle_b32 v57, v37 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[26:27], v[36:37], v[56:57]
	v_cndmask_b32_e32 v34, v23, v34, vcc
	v_pk_add_f32 v[32:33], v[32:33], v[34:35]
	s_nop 0
	v_mov_b32_e32 v35, v33
	v_mov_b32_e32 v38, v33
	v_mov_b32_e32 v23, v32
	v_mov_b32_e32 v34, v32
	v_permlane16_swap_b32_e32 v35, v38
	s_nop 0
	v_permlane16_swap_b32_e32 v23, v34
	v_cmp_eq_u32_e32 vcc, v35, v33
	v_pk_mul_f32 v[40:41], v[22:23], v[10:11] op_sel_hi:[0,1]
	s_nop 0
	v_cndmask_b32_e32 v35, v35, v38, vcc
	v_cmp_eq_u32_e32 vcc, v23, v32
	s_nop 1
	v_cndmask_b32_e32 v34, v23, v34, vcc
	v_pk_add_f32 v[38:39], v[32:33], v[34:35]
	buffer_load_dwordx4 v[32:35], v51, s[8:11], 0 offen
	ds_swizzle_b32 v54, v38 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v55, v39 offset:swizzle(SWAP,8)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[38:39], v[38:39], v[54:55]
	ds_swizzle_b32 v54, v38 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v55, v39 offset:swizzle(SWAP,4)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[30:31], v[38:39], v[54:55]
	v_pk_mul_f32 v[38:39], v[20:21], v[14:15] op_sel_hi:[0,1]
	v_pk_add_f32 v[38:39], v[38:39], v[40:41]
	ds_swizzle_b32 v36, v30 offset:swizzle(SWAP,2)
	v_mov_b32_e32 v40, v39
	v_mov_b32_e32 v41, v39
	v_mov_b32_e32 v21, v38
	v_mov_b32_e32 v23, v38
	v_permlane32_swap_b32_e32 v40, v41
	s_nop 0
	v_permlane32_swap_b32_e32 v21, v23
	v_cmp_eq_u32_e32 vcc, v40, v39
	ds_swizzle_b32 v37, v31 offset:swizzle(SWAP,2)
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[30:31], v[30:31], v[36:37]
	v_cndmask_b32_e32 v41, v40, v41, vcc
	v_cmp_eq_u32_e32 vcc, v21, v38
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v53, 0xffff0000, v32
	v_cndmask_b32_e32 v40, v21, v23, vcc
	v_pk_add_f32 v[38:39], v[38:39], v[40:41]
	v_lshlrev_b32_e32 v52, 16, v32
	v_mov_b32_e32 v40, v39
	v_mov_b32_e32 v41, v39
	v_mov_b32_e32 v21, v38
	v_mov_b32_e32 v23, v38
	v_permlane16_swap_b32_e32 v40, v41
	s_nop 0
	v_permlane16_swap_b32_e32 v21, v23
	v_cmp_eq_u32_e32 vcc, v40, v39
	v_and_b32_e32 v55, 0xffff0000, v35
	v_lshlrev_b32_e32 v54, 16, v35
	v_cndmask_b32_e32 v41, v40, v41, vcc
	v_cmp_eq_u32_e32 vcc, v21, v38
	v_and_b32_e32 v57, 0xffff0000, v34
	v_lshlrev_b32_e32 v56, 16, v34
	v_cndmask_b32_e32 v40, v21, v23, vcc
	v_pk_mul_f32 v[20:21], v[20:21], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[8:9] op_sel_hi:[0,1]
	v_pk_add_f32 v[20:21], v[20:21], v[22:23]
	v_pk_add_f32 v[38:39], v[38:39], v[40:41]
	v_mov_b32_e32 v23, v21
	v_mov_b32_e32 v43, v21
	v_mov_b32_e32 v22, v20
	v_mov_b32_e32 v42, v20
	v_permlane32_swap_b32_e32 v23, v43
	s_nop 0
	v_permlane32_swap_b32_e32 v22, v42
	v_cmp_eq_u32_e32 vcc, v23, v21
	ds_swizzle_b32 v40, v38 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v41, v39 offset:swizzle(SWAP,8)
	v_cndmask_b32_e32 v23, v23, v43, vcc
	v_cmp_eq_u32_e32 vcc, v22, v20
	v_pk_add_f32 v[24:25], v[52:53], v[24:25] neg_lo:[0,1] neg_hi:[0,1]
	v_add_u32_e32 v52, 0x210, v44
	v_cndmask_b32_e32 v22, v22, v42, vcc
	v_pk_add_f32 v[20:21], v[20:21], v[22:23]
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[36:37], v[38:39], v[40:41]
	v_mov_b32_e32 v23, v21
	v_mov_b32_e32 v43, v21
	v_mov_b32_e32 v22, v20
	v_mov_b32_e32 v42, v20
	v_permlane16_swap_b32_e32 v23, v43
	s_nop 0
	v_permlane16_swap_b32_e32 v22, v42
	v_cmp_eq_u32_e32 vcc, v23, v21
	ds_swizzle_b32 v40, v36 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v41, v37 offset:swizzle(SWAP,4)
	v_cndmask_b32_e32 v23, v23, v43, vcc
	v_cmp_eq_u32_e32 vcc, v22, v20
	ds_swizzle_b32 v38, v26 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v39, v27 offset:swizzle(SWAP,1)
	v_cndmask_b32_e32 v22, v22, v42, vcc
	v_pk_add_f32 v[20:21], v[20:21], v[22:23]
	ds_swizzle_b32 v22, v20 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v23, v21 offset:swizzle(SWAP,8)
	s_waitcnt lgkmcnt(4)
	v_pk_add_f32 v[34:35], v[36:37], v[40:41]
	ds_swizzle_b32 v36, v34 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v37, v35 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v42, v30 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(3)
	v_pk_add_f32 v[20:21], v[20:21], v[22:23]
	v_and_b32_e32 v23, 0xffff0000, v33
	v_lshlrev_b32_e32 v22, 16, v33
	ds_swizzle_b32 v32, v20 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v33, v21 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v43, v31 offset:swizzle(SWAP,1)
	v_pk_add_f32 v[28:29], v[22:23], v[28:29] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[22:23], v[54:55], v[60:61] neg_lo:[0,1] neg_hi:[0,1]
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_waitcnt lgkmcnt(1)
	v_pk_add_f32 v[20:21], v[20:21], v[32:33]
	ds_swizzle_b32 v40, v20 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v41, v21 offset:swizzle(SWAP,2)
	v_pk_add_f32 v[32:33], v[34:35], v[36:37]
	ds_swizzle_b32 v34, v32 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v35, v33 offset:swizzle(SWAP,1)
	s_waitcnt lgkmcnt(2)
	v_pk_add_f32 v[36:37], v[20:21], v[40:41]
	ds_swizzle_b32 v40, v36 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v41, v37 offset:swizzle(SWAP,1)
	v_pk_add_f32 v[20:21], v[56:57], v[58:59] neg_lo:[0,1] neg_hi:[0,1]
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
	v_add_f32_e32 v45, v45, v46
	s_load_dwordx2 s[4:5], s[0:1], 0xd0
	v_mul_f32_e32 v45, v47, v45
	v_mul_f32_e32 v45, v49, v45
	v_mul_f32_e32 v46, v45, v5
	v_pk_add_f32 v[30:31], v[30:31], v[42:43]
	v_pk_add_f32 v[26:27], v[26:27], v[38:39]
	v_pk_mul_f32 v[38:39], v[46:47], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[46:47], v[24:25] op_sel_hi:[0,1]
	v_pk_add_f32 v[26:27], v[26:27], v[42:43]
	v_pk_add_f32 v[30:31], v[30:31], v[38:39]
	s_waitcnt lgkmcnt(0)
	s_and_b32 s5, s5, 0xffff
	v_cvt_pk_bf16_f32 v31, v30, v31
	v_cvt_pk_bf16_f32 v30, v26, v27
	buffer_store_dwordx2 v[30:31], v51, s[4:7], 0 offen
	v_pk_add_f32 v[26:27], v[36:37], v[40:41]
	v_pk_add_f32 v[30:31], v[32:33], v[34:35]
	v_pk_mul_f32 v[32:33], v[46:47], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[46:47], v[20:21] op_sel_hi:[0,1]
	v_pk_add_f32 v[30:31], v[30:31], v[34:35]
	v_pk_add_f32 v[26:27], v[26:27], v[32:33]
	v_or_b32_e32 v3, 8, v51
	v_cvt_pk_bf16_f32 v27, v26, v27
	v_cvt_pk_bf16_f32 v26, v30, v31
	buffer_store_dwordx2 v[26:27], v3, s[4:7], 0 offen
.LBB0_2:
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx2 s[4:5], s[0:1], 0xa0
	v_mul_f32_e32 v30, v2, v5
	v_mul_f32_e32 v26, v4, v5
	v_pk_mul_f32 v[2:3], v[30:31], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[30:31], v[28:29] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	s_and_b32 s5, s5, 0xffff
	v_pk_add_f32 v[4:5], v[18:19], v[4:5]
	v_pk_add_f32 v[2:3], v[6:7], v[2:3]
	buffer_store_dwordx4 v[2:5], v44, s[4:7], 0 offen
	s_nop 1
	v_pk_mul_f32 v[4:5], v[26:27], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[26:27], v[28:29] op_sel_hi:[0,1]
	v_pk_add_f32 v[2:3], v[16:17], v[2:3]
	v_pk_add_f32 v[0:1], v[0:1], v[4:5]
	buffer_store_dwordx4 v[0:3], v50, s[4:7], 0 offen
	s_nop 1
	v_pk_mul_f32 v[0:1], v[30:31], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[30:31], v[22:23] op_sel_hi:[0,1]
	v_pk_add_f32 v[2:3], v[12:13], v[2:3]
	v_pk_add_f32 v[0:1], v[14:15], v[0:1]
	buffer_store_dwordx4 v[0:3], v48, s[4:7], 0 offen
	s_nop 1
	v_pk_mul_f32 v[0:1], v[26:27], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[26:27], v[22:23] op_sel_hi:[0,1]
	v_pk_add_f32 v[2:3], v[8:9], v[2:3]
	v_pk_add_f32 v[0:1], v[10:11], v[0:1]
	buffer_store_dwordx4 v[0:3], v52, s[4:7], 0 offen
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
		.amdhsa_next_free_sgpr 28
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
	.set kda_decode_step_kern_0.numbered_sgpr, 28
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
    .sgpr_count:     34
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
