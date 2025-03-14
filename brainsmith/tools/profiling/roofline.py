from model_profiling import RooflineModel

DEBUG = True


def auto_format_with_units(value, unit_type):
    # Scales and units for each type  
    scales = {
        'time': [(1, 'sec'), (1e-3, 'ms'), (1e-9, 'ns'), (1e-6, 'us')],
        'memory': [(8e12, 'TB'), (8e9, 'GB'), (8e6, 'MB'), (8e3, 'KB')],
        'compute': [(1e12, 'TOPS'), (1e9, 'GOPS'), (1e6, 'MOPS'), (1e3, 'KOPS'), (1, 'OPs')]
    }

    # Choose the most appropriate scale
    for scale, unit in scales[unit_type]:
        scaled_value = value / scale
        if scaled_value >= 1:
            break

    # Format the number with no more than two decimal places
    if scaled_value > 9999:
        formatted_number = f"{scaled_value:.0f} {unit}"
    elif scaled_value > 999:
        formatted_number = f"{scaled_value:.1f} {unit}"
    else:
        formatted_number = f"{scaled_value:.2f} {unit}"

    return formatted_number


def print_pipeline(compute, weights, activations, hbm, dtype):
    # Apply datatype
    weights = [[x*dtype for x in segment] for segment in weights]
    activations = [[x*dtype for x in segment] for segment in activations]
    print(f'Pipeline ({dtype}bit)')
    # Format
    compute = [[auto_format_with_units(x, 'compute') for x in segment] for segment in compute]
    weights = [[auto_format_with_units(x, 'memory') for x in segment] for segment in weights]
    activations = [[auto_format_with_units(x, 'memory') for x in segment] for segment in activations]
    hbm = [[auto_format_with_units(x, 'memory') for x in segment] for segment in hbm]
    segment_map = ['\nQKV ------', 'Score ----', 'Attention ', 'MHA Out --', 'MLP Up ---', 'MLP Down -']
    # Printing dimensions
    val_len = 13
    pipeline_len = len(compute)
    pipeline_width = max(len(segment) for segment in compute)
    # Print pipeline
    for i in range(pipeline_len):
        print(f'{segment_map[i%len(segment_map)]}' + '-'*(val_len*pipeline_width+12))
        pipe_strs = ['            Compute |', 
                     '            Weights |',
                     '        Activations |',
                     '         HBM (/sec) |']
        for j, segment in enumerate([compute[i], weights[i], activations[i], hbm[i]]):
            for k in range(pipeline_width):
                val = segment[k] if k < len(segment) else ' '*val_len
                val = ' '*(val_len-len(val)) + val
                pipe_strs[j] += val
            pipe_strs[j] += '|'
            print(pipe_strs[j])
    print('-'*(val_len*pipeline_width+22))


def roofline_analysis(model, hw_params, dtypes):
    # Profile model
    roofline = RooflineModel()
    if model['arch'] == 'twin_bert':
        print('Analyzing Twin-BERT model...')
        roofline.profile_bert(model=model['model_1'])
        profile_1 = roofline.get_profile()
        roofline.reset_pipeline()
        roofline.profile_bert(model=model['model_2'])
        profile_2 = roofline.get_profile()
        profile = {'act'  : profile_1['act']+profile_2['act'],
                   'macs' : profile_1['macs']+profile_2['macs'],
                   'w'    : profile_1['w']+profile_2['w'],
                   'vw'   : profile_1['vw']+profile_2['vw'],
                   'hbm'  : profile_1['hbm']+profile_2['hbm'],
        }
        profile['cycles'] = 1 # Offload not supported for twinbert currently
    else:
        if model['arch'] == 'bert':
            print('Analyzing BERT model...')
            roofline.profile_bert(model=model)
        elif model['arch'] == 'slm_pp':
            print('Analyzing SLM model in Prompt Processing mode...')
            roofline.profile_slm_pp(model=model)
        if model['arch'] == 'slm_tg':
            print('Analyzing SLM model in Token Generation mode...')
            roofline.profile_slm_tg(model=model)
        profile = roofline.get_profile()
        # Find iterations
        profile['cycles'] = model['num_layers'] if  model['offload'] else 1
        # Iterate if input is chunked
        if 'chunk_size' in model.keys():
            if DEBUG:
                print(f"Increasing cycles by {model['seq_len']/model['chunk_size']}x ({model['seq_len']}/{model['chunk_size']})")
            profile['cycles'] *= model['seq_len']/model['chunk_size']

    if DEBUG:
        print_pipeline(profile['macs'], profile['w'], profile['act'], profile['hbm'], 4)

    for dtype in dtypes:
        define_hardware(model, profile, hw_params, dtype)


def define_hardware(model, profile, hw_params, dtype):
    # HW Params
    dsps = int(hw_params['dsps'] * hw_params['dsp_util'])
    luts = int(hw_params['luts'] * hw_params['lut_util'])
    sram = int(hw_params['sram'] * hw_params['sram_util'])
    hbm = int(hw_params['hbm_bw'] * hw_params['hbm_util'])
    # Datatype params
    macs_per_dsp = {
        4 : hw_params['dsp_4bit'], # MACs per DSP
        5 : hw_params['dsp_8bit'],
        8 : hw_params['dsp_8bit'],
    }
    luts_per_mac = {
        4 : 247/16*1.2, # Int4 (+20% buffer)
        5 : 385/16*1.2, # Int5 (+20% buffer)
        8 : 76, # Int8
    }
    vec_dtype = 16
    # Sum metrics
    len_pipeline = len(profile['macs'])
    activations = sum(sum(segment) for segment in profile['act'])
    compute = sum(sum(segment) for segment in profile['macs'])
    weights = sum(sum(segment) for segment in profile['w'])
    vec_weights = sum(sum(segment) for segment in profile['vw'])
    hbm_spill = sum(sum(segment) for segment in profile['hbm'])

    # Perform roofline analysis for each datatype
    print(f'\n{dtype}-bit implementation:')
    # Calculate "x" factor
    x_compute = min(min(segment) for segment in profile['macs'])
    x_coef = compute/x_compute
    # Calculate DSPs & LUTs for "x" compute
    x_dsps = int(dsps/x_coef)
    x_luts = int(luts/x_coef)
    # Find pipeline segment latency per factor "x"
    x_macs_per_sec = x_dsps*hw_params['dsp_hz']*macs_per_dsp[dtype] + x_luts*hw_params['lut_hz']/luts_per_mac[dtype]
    x_latency = x_compute/x_macs_per_sec

    # Calculate IPS, latency, and HBM bandwidth
    ips = model['batch']/(profile['cycles']*x_latency)
    # Calculate latency
    num_tokens = model['chunk_size'] if 'chunk_size' in model.keys() else model['seq_len']
    overhang_latency = model['tile_size']/num_tokens*x_latency
    e2e_latency = profile['cycles']*(4*overhang_latency + 2*x_latency) + (x_latency-overhang_latency)
    # Calculate HBM bandwidth utilization
    hbm_bandwidth_cycles = (dtype*weights+vec_dtype*vec_weights)/(x_latency*len_pipeline)
    hbm_bandwidth_spill = dtype*hbm_spill/x_latency
    # Calculate sram usage in MB, check for capacity
    sram_act = activations*dtype
    sram_weights = weights*dtype + vec_weights*vec_dtype
    if model['offload']:
        sram_weights *= 2 # Double buffer if offloading
    sram_used = sram_act + sram_weights
    if sram_used > sram:
        {'-- OVER!' if (sram_used > sram) else ''}

    # Check for memory bottleneck
    hbm_bandwidth = hbm_bandwidth_cycles + hbm_bandwidth_spill
    hbm_ratio = hbm_bandwidth/hbm
    if hbm_ratio > 1:
        theoretical_ips = ips
        ips /= hbm_ratio
        e2e_latency *= hbm_ratio
        print(f'HBM Bottleneck! Throttling IPS & Latency by {hbm_ratio:.6f}x ({theoretical_ips:.6f} --> {ips:.6f})')
    
    # Cleanup & print
    sram_weights /= 8e6
    sram_act /= 8e6
    sram_used /= 8e6
    # Change metrics to per-token for PP
    if model['arch'] == 'slm_pp':
        ips *= model['seq_len']
    print(f'    Throughput: {ips:.2f} IPS')
    print(f'    End-to-End Latency: {e2e_latency/1e-3:.4f} ms')
    print(f'    Per-Token Latency: {x_latency/1e-3:.4f} ms')
    print(f'    HBM: {hbm_bandwidth_cycles/8e9:.2f} + {hbm_bandwidth_spill/8e9:.2f} = {hbm_bandwidth/8e9:.2f}/{int(hbm/8e9)} GB/s = {hbm_bandwidth/hbm:.2f}x')
    print(f"    sram: {sram_weights:.2f} weights + {sram_act:.2f} activations = {sram_used:.2f}/{sram/8e6:.2f} MB")
