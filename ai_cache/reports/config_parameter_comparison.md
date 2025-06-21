# DataflowBuildConfig Parameter Comparison

## bert_direct (WORKS) vs bert_new (FAILS)

### Parameters in bert_direct but NOT in bert_new output:
1. **folding_config_file** - bert_direct passes this, bert_new might not be
2. **split_large_fifos** - bert_direct explicitly sets to True, bert_new missing
3. **fifosim_n_inferences** - bert_direct sets to 2, bert_new missing
4. **verification_atol** - bert_direct sets to 1e-1, bert_new missing  
5. **verify_input_npy** - bert_direct sets path, bert_new missing
6. **verify_expected_output_npy** - bert_direct sets path, bert_new missing
7. **verify_save_full_context** - bert_direct sets this, bert_new missing
8. **save_intermediate_models** - bert_direct sets this via args
9. **generate_outputs** - format might be different
10. **stitched_ip_gen_dcp** - bert_direct sets based on args.dcp

### Parameters in bert_new but NOT in bert_direct:
1. **mvau_wwidth_max=36** - This could be significant!
2. **auto_fifo_strategy="largefifo_rtlsim"** - This could affect FIFO sizing!
3. **enable_hw_debug=false**
4. **verbose=false**

### Critical Differences:
1. **split_large_fifos** - Missing in bert_new, could cause FIFO issues!
2. **mvau_wwidth_max=36** - Only in bert_new, could affect folding
3. **auto_fifo_strategy** - Only in bert_new, could affect FIFO behavior

These differences could definitely cause FIFO shape mismatches!