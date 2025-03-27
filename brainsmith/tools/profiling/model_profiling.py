import numpy as np

PIPELINE_LENGTH = 6


class RooflineModel():
    def __init__(self):
        # Set model parameters
        self.reset_pipeline()

    def update_model(self, model):
        # Hidden dimensions
        self.q_heads = model['num_heads']
        self.kv_heads = model['num_kv_heads'] if 'num_kv_head' in model.keys() else model['num_heads']
        self.head_size = model['head_size']
        self.hidden_dim = model['num_heads']*model['head_size']
        self.intermediate = model['intermediate']
        # Input tokens
        self.seq_len = model['chunk_size'] if 'chunk_size' in model.keys() else model['seq_len']
        self.out_len = model['out_len'] if 'out_len' in model.keys() else 0
        self.window_size = model['window_size'] if 'window_size' in model.keys() else 0
        # Memory optimizations
        self.min_kernel_dim = model['spill_size'] if 'spill_size' in model.keys() else 0
        self.spill_map = model['spill_map'] if 'spill_map' in model.keys() else PIPELINE_LENGTH*[False]
        self.residuals_in_hbm = model['residuals_in_hbm'] if 'residuals_in_hbm' in model.keys() else False
        self.attn_kernel_fusion = model['attn_kernel_fusion'] if 'attn_kernel_fusion' in model.keys() else False
        # Iterations
        self.hard_batch = model['hard_batch'] if 'hard_batch' in model.keys() else 1
        self.iterations = 1 if model['offload'] else model['num_layers']
        # Initialize

    def reset_pipeline(self):
        self.compute = []
        self.weights = []
        self.activations = []
        self.vector_weights = [[]] # Always empty for now
        self.hbm_bw = []

    def update_pipeline(self, activations, compute, weights, hbm_bw):
        activations = [self.hard_batch * x for x in activations]
        compute = [self.hard_batch * x for x in compute]
        self.activations.append(activations)
        self.compute.append(compute)
        self.weights.append(weights)
        self.hbm_bw.append(hbm_bw)

    def get_profile(self):
        profile = {'act'  : self.iterations*self.activations, 
                   'macs' : self.iterations*self.compute, 
                   'w'    : self.iterations*self.weights,
                   'vw'   : self.iterations*self.vector_weights, 
                   'hbm'  : self.iterations*self.hbm_bw}
        return profile

    # Calculate HBM bandwidth for spilled weights
    def profile_hbm_bw(self, weights, pos):
        hbm_bw = []
        ocm = []
        if (self.spill_map[pos]):
            print(f'Spilling segment {pos} with dim {self.min_kernel_dim}...')
            print(f'Wi: {weights}')
        for weight in weights:
            if self.min_kernel_dim > 0 and self.spill_map[pos]:
                hbm_bw += [np.prod(weight)]
                weight = weight[:-1] + [self.min_kernel_dim]
                print(f'WO: {weight}')
            ocm.append(np.prod(weight))
        return ocm, hbm_bw

    def profile_qkv(self, seq_len, embed_dim, q_heads, k_heads, v_heads, head_size, pos=0):
        # Profile values
        input_act = seq_len*embed_dim
        residual_act = seq_len*embed_dim
        q_compute = seq_len*embed_dim*(q_heads*head_size) # [seq, embed] x [embed, q_heads*head_dim] -> [seq, q_heads*head_dim]
        k_compute = seq_len*embed_dim*(k_heads*head_size) # [seq, embed] x [embed, k_heads*head_dim] -> [seq, k_heads*head_dim]
        v_compute = seq_len*embed_dim*(v_heads*head_size) # [seq, embed] x [embed, v_heads*head_dim] -> [seq, v_heads*head_dim]
        q_weights = [embed_dim, q_heads*head_size]
        k_weights = [embed_dim, k_heads*head_size]
        v_weights = [embed_dim, v_heads*head_size]
        # Aggregate
        activations = [input_act]
        compute = [q_compute, k_compute, v_compute]
        weights, hbm_bw = self.profile_hbm_bw([q_weights, k_weights, v_weights], pos)
        if not self.residuals_in_hbm:
            activations += [residual_act] # Represents the residuals in Score and Attn
        # Add to model
        self.update_pipeline(activations, compute, weights, hbm_bw)

    def profile_mha_score(self, seq_len, embed_dim, inner_dim, q_heads, k_heads, v_heads, head_size, kv_cache, pos=1):
        # Profile values
        shuffle_q_act = seq_len*q_heads*head_size
        shuffle_k_act = seq_len*k_heads*head_size
        shuffle_v_act = seq_len*v_heads*head_size
        score_compute = q_heads*(seq_len*head_size*inner_dim)
        activations = [shuffle_q_act, shuffle_k_act, shuffle_v_act]
        compute = [score_compute]
        # Aggregate
        if kv_cache:
            weights = [[k_heads, head_size, inner_dim]]
            weights, hbm_bw = self.profile_hbm_bw(weights, pos)
        else:
            weights = []
            hbm_bw = []
        if not self.residuals_in_hbm:
            activations += [seq_len*embed_dim]
        # Add to model
        self.update_pipeline(activations, compute, weights, hbm_bw)

    def profile_mha_attn(self, seq_len, embed_dim, inner_dim, q_heads, v_heads, head_size, kv_cache, pos=2):
        score_act = q_heads*seq_len if self.attn_kernel_fusion else q_heads*seq_len*inner_dim
        attn_compute = q_heads*(seq_len*inner_dim*head_size)
        # Aggregate
        activations = [score_act]
        compute = [attn_compute]
        if kv_cache:
            weights = [[v_heads, inner_dim, head_size]]
            weights, hbm_bw = self.profile_hbm_bw(weights, pos)
        else:
            weights = []
            hbm_bw = []
        if self.residuals_in_hbm:
            hbm_bw += seq_len*embed_dim # Stream in from HBM for next segment
        else:
            activations += [seq_len*embed_dim]
        # Add to model
        self.update_pipeline(activations, compute, weights, hbm_bw)

    def profile_mha_out(self, seq_len, hidden_dim, pos=3):
        # Profile values
        input_act = seq_len*hidden_dim
        mha_out_compute = seq_len*hidden_dim*hidden_dim
        mha_out_weights = [hidden_dim, hidden_dim]
        # Aggregate
        activations = [input_act]
        compute = [mha_out_compute]
        weights = [mha_out_weights]
        weights, hbm_bw = self.profile_hbm_bw(weights, pos)
        # Add to model
        self.update_pipeline(activations, compute, weights, hbm_bw)
        
    def profile_mlp_up(self, seq_len, hidden_dim, intermediate, silu, pos=4):
        # Profile values
        input_act = seq_len*hidden_dim
        mlp_in_compute = seq_len*hidden_dim*intermediate
        mlp_gate_compute = seq_len*hidden_dim*intermediate
        mlp_in_weights = [hidden_dim, intermediate]
        mlp_gate_weights = [hidden_dim, intermediate]
        # Aggregate
        activations = [input_act]
        compute = [mlp_in_compute, mlp_gate_compute] if silu else [mlp_in_compute]
        weights = [mlp_in_weights, mlp_gate_weights] if silu else [mlp_in_weights]
        weights, hbm_bw = self.profile_hbm_bw(weights, pos)
        # Add to model
        self.update_pipeline(activations, compute, weights, hbm_bw)
        
    def profile_mlp_down(self, seq_len, hidden_dim, intermediate, pos=5):
        # Profile values
        residual_act = seq_len*hidden_dim
        output_act = seq_len*hidden_dim
        mlp_out_compute = seq_len*hidden_dim*intermediate
        mlp_out_weights = [hidden_dim, intermediate]
        # Aggregate
        activations = [output_act, residual_act]
        compute = [mlp_out_compute]
        weights = [mlp_out_weights]
        weights, hbm_bw = self.profile_hbm_bw(weights, pos)
        # Add to model
        self.update_pipeline(activations, compute, weights, hbm_bw)
        
    def profile_slm(self, input_len, inner_dim, kv_cache):
        if inner_dim > self.window_size > 0:
            inner_dim = self.window_size
        print(f'Inner dim: {inner_dim}')
        self.profile_qkv(input_len, self.hidden_dim, self.q_heads, self.kv_heads, self.kv_heads, self.head_size)
        self.profile_mha_score(input_len, self.hidden_dim, inner_dim, self.q_heads, self.kv_heads, self.kv_heads, self.head_size, kv_cache)
        self.profile_mha_attn(input_len, self.hidden_dim, inner_dim, self.q_heads, self.kv_heads, self.head_size, kv_cache)
        self.profile_mha_out(input_len, self.hidden_dim)
        self.profile_mlp_up(input_len, self.hidden_dim, self.intermediate, True)
        self.profile_mlp_down(input_len, self.hidden_dim, self.intermediate)
        return self.activations, self.compute, self.weights, self.vector_weights, self.hbm_bw

    # Profile compute & memory for SLMs in the "Prompt Processing" phase
    def profile_slm_pp(self, model=None):
        if model != None:
            self.update_model(model)
        return self.profile_slm(self.seq_len, self.seq_len, False)

    # Profile compute & memory for SLMs in the "Token Generation" phase
    def profile_slm_tg(self, model=None):
        if model != None:
            self.update_model(model)
        return self.profile_slm(1, self.seq_len+self.out_len, True)

    # Profile compute & memory for BERT
    def profile_bert(self, model=None):
        if model != None:
            self.update_model(model)
        self.profile_qkv(self.seq_len, self.hidden_dim, self.q_heads, self.kv_heads, self.kv_heads, self.head_size)
        self.profile_mha_score(self.seq_len, self.hidden_dim, self.seq_len, self.q_heads, self.kv_heads, self.kv_heads, self.head_size, False)
        self.profile_mha_attn(self.seq_len, self.hidden_dim, self.seq_len, self.q_heads, self.kv_heads, self.head_size, False)
        self.profile_mha_out(self.seq_len, self.hidden_dim)
        self.profile_mlp_up(self.seq_len, self.hidden_dim, self.intermediate, False)
        self.profile_mlp_down(self.seq_len, self.hidden_dim, self.intermediate)
       



############################## DLRMv2 ##############################

    def profile_mlp_bottom(self, seq_len, hidden_dim, intermediate, pos=5):
        pass


    # Profile compute & memory for DLRMv2
    def profile_dlrm(self, model=None):
        if model is not None:
            self.update_model(model)
        pass
        

############################## HSTU ##############################
    def profile_hstu(self, seq_len, hidden_dim, intermediate, silu, pos=4):
        pass