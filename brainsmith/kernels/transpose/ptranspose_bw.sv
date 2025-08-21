/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief       A streaming 2D parallel transpose unit. (I,J) -> (J,I) with SDIM
 * 		parallelism
 * @author      Shane T. Fleming <shane.fleming@amd.com>
 *
 * @description 
 *
 * This unit can perform a streaming transpose (I,J) -> (J,I) with SDIM
 * parallelism.
 * It achieves this by using SDIM banks of memory and rotating write and reads
 * to the banks such that collisions are avoided and maximum throughput can be
 * maintained (II=1).
 *
 * Decisions about when to rotate writes and reads to the different banks are
 * made by a WR_ROT_PERIOD param, for writes, and a RD_PATTERN param matrix, for reads.
 * These two are computed at elaboration time and are constants at runtime.
 * 
 * After WR_ROT_PERIOD writes to the banks the write bank allocation is shifted to
 * the right by one position.
 * The WR_ROT_PERIOD is determined by considering the prime factors of SDIM
 * along with the inner input dimension I. A possible rotation that will result in
 * a conflict-free bank allocation is when the WR_ROT_PERIOD is set to the inner 
 * dimension divided by the largest prime factor of SDIM.
 *
 * The RD_PATTERN for the read side is a SDIMxSDIM matrix of banks that is a 
 * periodic pattern of banks across the input matrix. This is computed by
 * evaluating what a SDIMxSDIM block of bank allocations will look like with
 * the current WR_ROT_PERIOD.
 *
 * On the write path of the hardware data is written into the banks according
 * to the initial write banks. A counter tracks how many writes have happened
 * and then after WR_ROT_PERIOD counts the banks are rotated. The write
 * address is incremented by one every write for every bank.
 *
 * The Read path has logic to generate the addresses for SDIM reads based on
 * the current index of the output loop:
 *
 *        	j : [0,J)
 *        	   i : [0,I)
 *        	     emit(i*J + j)
 *
 * SDIM address are generated and each is sent to the appropriate SDIM banks
 * based on the schedule in the relevant column of the RD_PATTERN matrix.
 * This column of the RD_PATTERN matrix is then forwarded to the output of the
 * banks, where a clock cycle later the relevant outputs appear at each bank
 * output. The output data is then rearranged again using the forwarded RD_PATTERN 
 * column to assign the appropriate output signals. 
 * Logic is used to track what column of the the RD_PATTERN to use based 
 * on where the circuit current is in the output iteration space.
 *
 * Control flow for writing and reading the banks are managed by job
 * scheduling logic. This means that while a job is being
 * outputted on the read side, the next job can be written on the write side
 * enabling both the write path and the read path to be active simultaneously.  
****************************************************************************/

// A SkidBuffer module. This is a 2-depth FIFO constructed from registers
// that can be used to decouple rdy/vld handshake signals to improve timing.
module skid_buffer #(
	int unsigned WIDTH = 8	
)(
    input  logic              clk,
    input  logic              rst,

    input  logic              input_TVALID,
    input  logic [WIDTH-1:0]  input_TDATA,
    output logic              input_TREADY,

    output logic              output_TVALID,
    input  logic              output_TREADY,
    output logic [WIDTH-1:0]  output_TDATA
);

    // Internal signals
    logic [WIDTH-1:0] buffer_data;
    logic             buffer_valid;

    // Output logic
    assign output_TDATA  =  buffer_valid ? buffer_data : input_TDATA;
    assign output_TVALID  =  buffer_valid || input_TVALID;
    assign input_TREADY  = !buffer_valid && (output_TREADY || !input_TVALID);

    // Skid buffer behavior
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            buffer_valid <= 1'b0;
            buffer_data  <= {WIDTH{1'b0}};
        end else 
            if (buffer_valid && output_TREADY) 
                buffer_valid <= 1'b0; // Buffer emptied
            else if (input_TVALID && !output_TREADY) begin
                buffer_data  <= input_TDATA;
                buffer_valid <= 1'b1; // Buffer filled
            end
    end
endmodule


// A memory bank in the ptranspose design. Pattern was kept as simple
// as possible to help with Vivado BRAM inference. 
module mem_bank #(
	int unsigned WIDTH = 8,
	int unsigned DEPTH = 128
)(
	input logic clk,
	input logic rst,

	input logic [WIDTH-1:0] d_in,
	input logic [$clog2(DEPTH)-1:0] wr_addr,
	input logic wr_en,

	output logic [WIDTH-1:0] d_out,
	input  logic [$clog2(DEPTH)-1:0] rd_addr,
	input  logic rd_hold
);

	(* ram_style="block" *) logic [WIDTH-1:0] mem [DEPTH-1:0]; // The Mem for this bank

	// Write channel
	always_ff @(posedge clk) 
		if (wr_en) mem[wr_addr] <= d_in;
	
	// Read channel
	always_ff @(posedge clk) 
		if (rst) 
			d_out <= 'd0;
		else 
			if(!rd_hold) 
				d_out <= mem[rd_addr];
endmodule

// @brainsmith BITS


// @brainsmith BDIM s_axis [C] [PE]
// @brainsmith BDIM m_axis [C] [PE]
// @brainsmith DATATYPE s_axis FIXED WI WI
// @brainsmith DATATYPE m_axis FIXED O_BITS O_BITS
// @brainsmith DATATYPE_PARAM s_axis width WI
// @brainsmith DATATYPE_PARAM s_axis signed SIGNED
// @brainsmith DATATYPE_PARAM s_axis format FPARG
// @brainsmith DATATYPE_PARAM m_axis width O_BITS
// @brainsmith DATATYPE_PARAM m_axis bias BIAS


	output	logic  input_tready,
	input	logic  input_tvalid,
	input	logic [((input_SDIM*input_WIDTH+7)/8)*8-1:0]  input_tdata,

    
// @brainsmith BDIM input [I, J]
// @brainsmith SDIM input SIMD
// @brainsmith DATATYPE_PARAM input width BITS

// ----------------------------------------
// Parallel Transpose Unit (PTranspose)
// ----------------------------------------
module ptranspose #(
	int unsigned BITS  = 8,  // Bitwidth of each element
	int unsigned I     = 128, // Input dimension I
	int unsigned J     = 384, // Input dimension J 
	int unsigned SDIM  = 4    // SDIM parallelism
)(
	input logic                       clk, // global control 
	input logic                       rst,

	output logic                      input_TREADY, // Input stream
	input  logic                      input_TVALID,
	input  logic [SDIM-1:0][BITS-1:0] input_TDATA,

	input  logic                      output_TREADY, // Output stream
	output logic                      output_TVALID,
	output logic [SDIM-1:0][BITS-1:0] output_TDATA
); 

	// elaboration time compute for generating the WR_ROT_PERIOD
	// This is used to determine how often the write banks should be
	// rotated at runtime, i.e. after how many SDIM writes into the banks
	// do we need to swap the allocation.
	function automatic logic [$clog2(I*J)-1: 0] calculate_WR_ROT_PERIOD();
		int unsigned factors[10];
		int unsigned num_factors;
		int unsigned max = 0;
	    	int unsigned number = SDIM;

		if ((J % SDIM) == 0) return J/SDIM;
	    	
	    	if (SDIM % 2 == 0) begin  // Check for factor of 2  
	    	    factors[num_factors++] = 2;  
	    	    while (number % 2 == 0)   
	    	        number /= 2;  
	    	end  
	
	    	for (int i = 3; i * i <= number; i += 2)  // Check for odd factors starting from 3  
	    	    if (number % i == 0) begin  
	    	        factors[num_factors++] = i;  
	    	        while (number % i == 0)  
	    	            number /= i;  
	    	    end  
	
	    	
	    	if (number > 2) // If number is still greater than 2, it is a prime number  
	        	factors[num_factors++] = number;  
          
        	for (int i = 0; i < num_factors; i++)   
			if ((J % factors[i]) == 0)
				if ((J/factors[i]) > max)
					max = J/factors[i];

		return max;
	endfunction : calculate_WR_ROT_PERIOD

	localparam logic [$clog2(I*J)-1: 0] WR_ROT_PERIOD = calculate_WR_ROT_PERIOD(); 
	localparam logic [$clog2(I*J)-1: 0] RD_ROT_PERIOD = I/SDIM; // (I % SDIM == 0) is a constraint 

	typedef logic [$clog2(SDIM)-1:0] rd_pattern_t     [SDIM-1:0][SDIM-1:0];
	typedef logic [$clog2(SDIM)-1:0] rd_pattern_col_t [SDIM-1:0];

	// --------------------------------------------------------------------------
	// RD_PATTERN generation: 
	// --------------------------------------------------------------------------
	// Generate the SDIMxSDIM RD_BANK pattern at compile time. 
	// This will then be read at runtime to reconstruct the bank
	// access pattern.
	function rd_pattern_t generate_rd_pattern();
		rd_pattern_t pattern;
		int unsigned row=0;

		logic[$clog2(SDIM)-1:0] ct_wr_banks[SDIM]; // Bank allocation for generating compile time RD_PATTERN
		for (int unsigned i=0; i<SDIM; i++) ct_wr_banks[i] = i;

		// Generate the RD_PATTERN
		for(int unsigned c=0; c<(SDIM*J); c++) begin
			if ((c%J == 0) && (c != 0)) begin // Track what row we are on
				row = row + 1;
				row = row % SDIM; 
			end	
			
			if ((c%(WR_ROT_PERIOD*SDIM) == 0) && (c != 0)) begin // Do we rotate?
	    			logic[$clog2(SDIM)-1:0] first_element;  
	    			first_element = ct_wr_banks[0]; 
	    			for (int unsigned i = 0; i <SDIM-1; i++)   
	    			    ct_wr_banks[i] = ct_wr_banks[i + 1]; 
	    			ct_wr_banks[SDIM-1] = first_element; 
			end	

			if ((c % J) < SDIM) // Are we in the assignment region?				
				pattern[row][c%J] = ct_wr_banks[c%SDIM];
		end
		return pattern;
	endfunction : generate_rd_pattern	

	localparam rd_pattern_t RD_PATTERN = generate_rd_pattern();

	// --------------------------------------------------------------------------
	// RD_PATTERN Phase shift calculation 
	// --------------------------------------------------------------------------
	function automatic int unsigned gcd(input int a, input int b);  
		if (b == 0)   
			return a;  
		else   
			return gcd(b, a % b);  
	endfunction  
        
	function automatic int unsigned compute_RD_PHASE_SHIFT(input int x, input int y);  
		const int gcd_value = gcd(x, y);  
		if (gcd_value == 1 || gcd_value == SDIM)   
			return 0; // Numbers are coprime  
		else   
			return gcd_value; // Return the GCD  
	endfunction  
        
        localparam logic[$clog2(SDIM)-1:0] RD_PHASE_SHIFT = compute_RD_PHASE_SHIFT(J%SDIM, SDIM); 

	// --------------------------------------------------------------------------
	//   Memory Banks
	// --------------------------------------------------------------------------
	logic osb_vld; // output skidbuffer valid signal
	logic osb_rdy; // output skid buffer ready signal

	localparam int unsigned BANK_DEPTH  = 2*(I*J/SDIM);
	localparam int unsigned PAGE_OFFSET =   (I*J)/SDIM; 

	// Instantiate separate banks
	logic                           mem_banks_wr_en   [SDIM-1:0];
	logic [BITS-1:0]                mem_banks_in      [SDIM-1:0];
	logic [BITS-1:0]                mem_banks_out     [SDIM-1:0];
	logic [$clog2(BANK_DEPTH)-1:0]  mem_banks_rd_addr [SDIM-1:0];
	logic [$clog2(BANK_DEPTH)-1:0]  wr_addr; 

	// Generates the SDIM dual port memory banks
	for(genvar i =0; i<SDIM; i++) begin : gen_mem_banks
		mem_bank #(
			.WIDTH(BITS),
			.DEPTH(BANK_DEPTH)
		) mem_bank_inst (
			.clk(clk),
			.rst(rst),
			.d_in(mem_banks_in[i]),
			.wr_addr(wr_addr),
			.wr_en(input_TREADY && input_TVALID),
			.d_out(mem_banks_out[i]),
			.rd_addr(mem_banks_rd_addr[i]),
			.rd_hold(!osb_rdy)
		);
	end : gen_mem_banks

	// Write bank schedule	
	logic[$clog2(SDIM)-1:0] wr_bank_schedule      [SDIM-1:0];
	logic[$clog2(SDIM)-1:0] next_wr_bank_schedule [SDIM-1:0];

	// Rotate the next write schedule (only registered every WR_ROT_PERIOD)
	always_comb begin : writeBankScheduleRotation
		// At the page boundary we need to reset the write schedule
		if ((wr_addr == PAGE_OFFSET - 1) || (wr_addr == 2*PAGE_OFFSET - 1))
			for(int unsigned i=0; i<SDIM; i++) next_wr_bank_schedule[i] = i;
		else begin
			next_wr_bank_schedule [SDIM-1] = wr_bank_schedule[0];
			for(int unsigned i=0; i<SDIM-1; i++) 
				next_wr_bank_schedule[i] = wr_bank_schedule[i+1];
		end

	end : writeBankScheduleRotation

	// Remap the input based on the current write bank rotation
	always_comb begin
		for(int unsigned i=0; i<SDIM; i++)  mem_banks_in[i] = 'd0;  // default values to avoid latch inference
		for(int unsigned i=0; i<SDIM; i++)  mem_banks_in[wr_bank_schedule[i]] = input_TDATA[i];
	end

	// Write bank schedule rotation logic
	logic[$clog2(WR_ROT_PERIOD)-1:0] wr_rot_counter;
	logic[$clog2(I*J/SDIM)-1:0]      wr_counter;

	always_ff @(posedge clk) begin
		if (rst) begin
			for(int unsigned i=0; i<SDIM; i++) wr_bank_schedule[i] <= i;
			wr_rot_counter <= 'd0;
			wr_counter <= 'd0;
		end
		else 
			if (input_TVALID && input_TREADY) begin // Detect once we need to rotate and perform right rotation
				if (wr_rot_counter == WR_ROT_PERIOD - 1) begin
					wr_rot_counter <= 'd0;
					if (wr_counter == (I*J/SDIM - 1))  
						wr_counter <= 'd0;
					for (int unsigned i = 0; i < SDIM; i++) wr_bank_schedule[i] <= next_wr_bank_schedule[i];
				end	
				else begin 
					wr_rot_counter <= wr_rot_counter + 'd1;
					wr_counter <= wr_counter + 'd1;
				end
			end
	end

	// Job tracking and bank page locking
	logic [1:0] wr_jobs_done; // Bit vector tracking when writes have been completed to pages
	logic rd_page_in_progress; // 0 - reading from PAGE A, 1 - reading from PAGE B  
	logic [$clog2(BANK_DEPTH)-1:0] page_rd_offset;

	always_ff @(posedge clk) begin
		if (rst) begin
			wr_jobs_done <= 2'b00;
			rd_page_in_progress <= 1'b0;
		end

		// Track if we have completed a job
		if (wr_addr == PAGE_OFFSET   - 1) wr_jobs_done[0] <= 1'b1;
		if (wr_addr == 2*PAGE_OFFSET - 1) wr_jobs_done[1] <= 1'b1;

		// Clear the relevant job once it is read
		if ((rd_j_cnt == J-1) && (rd_i_cnt+SDIM == I) && (osb_rdy && osb_vld)) begin	
		       wr_jobs_done[rd_page_in_progress] <= 1'b0;	
		       rd_page_in_progress <= !rd_page_in_progress;
		end
	end

	assign page_rd_offset = rd_page_in_progress ? PAGE_OFFSET : 'd0;
        assign input_TREADY = !wr_jobs_done[0] || !wr_jobs_done[1]; 	

	// Write address incrementer (resets to the start once the second page is written)
	always_ff @(posedge clk) begin
		if (rst) wr_addr <= 'd0;
		else 
			if (input_TVALID && input_TREADY) 
				if (wr_addr < (2*PAGE_OFFSET - 1))
					wr_addr <= wr_addr + 'd1;
				else
					wr_addr <= 'd0;
	end
	
	// --------------------------------------------------------------------------
	//    Read Address generation 
	// --------------------------------------------------------------------------
	logic[$clog2(I)-1 : 0] rd_i_cnt;
	logic[$clog2(J)-1 : 0] rd_j_cnt;
	logic rd_guard;
	assign rd_guard = !rd_page_in_progress && !wr_jobs_done[0] && !wr_jobs_done[1];

	// Logic to track which iteration we are on for the read side
	always_ff @(posedge clk) begin : readIndexLoopTracking
		if (rst) begin
			rd_i_cnt <= 'd0;
			rd_j_cnt <= 'd0;
		end
		else
			if(osb_rdy && !rd_guard) 
				if((rd_i_cnt+SDIM) >= I) begin
					rd_i_cnt <= 'd0;
					if( rd_j_cnt < J-1) 
						rd_j_cnt <= rd_j_cnt + 'd1;
					else 
						rd_j_cnt <= 'd0;
				end
				else
					rd_i_cnt <= rd_i_cnt + SDIM; 
	end : readIndexLoopTracking

	// Combinatorial generation of the current set of Read addresses
	always_comb begin : bankRdAddrGen
		for(int unsigned i=0; i<SDIM; i++) mem_banks_rd_addr[i] = 'd0; // default to avoid latch inference	
		for(int unsigned i=0; i < SDIM; i++) 
			mem_banks_rd_addr[RD_PATTERN[i][rd_pattern_phase_adj_idx]] = ((rd_i_cnt + i)*J + rd_j_cnt)/SDIM + page_rd_offset; 
	end : bankRdAddrGen
	// --------------------------------------------------------------------------
	
	// --------------------------------------------------------------------------
        logic [SDIM-1:0][BITS-1:0]     data_reg; // remapped output   
	logic [$clog2(I*J/SDIM)-1:0]   rd_pattern_idx;
	logic [$clog2(I*J/SDIM)-1:0]   rd_pattern_phase_adj_idx; // Phase adjusted rd_pattern_idx
	rd_pattern_col_t               rd_pattern_col_ff; // The fowarded rotation pattern 

	// Forward the current RD_PATTERN row onto the next pipeline stage
	always_ff @(posedge clk) begin : rdPatternColForwarding
		if (rst) osb_vld <= 0;
		else
			osb_vld <= !rd_guard && osb_rdy;
			if (osb_rdy) 
				for(int unsigned i=0; i<SDIM; i++) 
					rd_pattern_col_ff[i] <= RD_PATTERN[i][rd_pattern_phase_adj_idx];
	end : rdPatternColForwarding

	// Structural remapping using the output of the memory banks
	// and the Read rotation from the previous clock cycle that was
	// used to generate the read addresses.
	for(genvar i=0; i<SDIM; i++) 
		assign data_reg[i] = mem_banks_out[rd_pattern_col_ff[i]];
	// --------------------------------------------------------------------------

	// --------------------------------------------------------------------------
	logic [$clog2(I*J/SDIM)-1:0] rd_counter;

	// Track the read count for determining when rotations should occur.
	always_ff @(posedge clk) begin : readTrackingForRotationDecisions
		if (rst) begin
			rd_pattern_idx <= 'd0;
			rd_counter <= 'd0;
		end
		else begin
			if (osb_rdy && !rd_guard) begin
				rd_counter <= rd_counter + 'd1;
				if (rd_counter == RD_ROT_PERIOD-1) begin
					rd_counter <= 'd0;
      					if (rd_pattern_idx == (SDIM-1)) rd_pattern_idx <= 'd0;
					else begin
						rd_counter <= 'd0;
						rd_pattern_idx <= rd_pattern_idx + 'd1;
					end
				end 
					

				// At the page boundary reset our RD_PATTERN lookup
				if ((rd_j_cnt == J-1) && (rd_i_cnt+SDIM == I )) begin 
					rd_pattern_idx <= 'd0;
					rd_counter <= 'd0;
				end
			end

		end	
	end : readTrackingForRotationDecisions

	// --------------------------------------------------------------------------
	// Logic for tracking the phase adjust
	// --------------------------------------------------------------------------
	logic [$clog2(I*J/SDIM)-1:0] rd_pattern_phase_adj_sum;
	logic [$clog2(SDIM)-1:0]     phase_pattern_counter;

	always_ff @(posedge clk) begin
		if (rst) begin
			phase_pattern_counter <= 'd0;
		end else begin
			if (osb_rdy && !rd_guard) begin
				if(rd_counter == RD_ROT_PERIOD-1) 
					phase_pattern_counter <= 'd0;
				else
					if (phase_pattern_counter >= (SDIM-1))
						phase_pattern_counter <= phase_pattern_counter - (SDIM-1);
					else
						phase_pattern_counter <= phase_pattern_counter + RD_PHASE_SHIFT;

				if ((rd_j_cnt == J-1) && (rd_i_cnt+SDIM == I ))  
					phase_pattern_counter <= 'd0;
			end
		end
	end

	assign rd_pattern_phase_adj_sum =  phase_pattern_counter + rd_pattern_idx;
	assign rd_pattern_phase_adj_idx = (rd_pattern_phase_adj_sum > (SDIM-1)) ?  rd_pattern_phase_adj_sum - (SDIM) : rd_pattern_phase_adj_sum;	
	// --------------------------------------------------------------------------

	// Output SkidBuffer -- Used to decouple control signals for timing
	// improvements
	skid_buffer #( 
		.WIDTH(SDIM*BITS) 
	) 
	oskidbf_inst (
		.clk(clk),
		.rst(rst),

		.input_TVALID(osb_vld),
		.input_TREADY(osb_rdy),
		.input_TDATA(data_reg),

		.output_TVALID(output_TVALID),
		.output_TREADY(output_TREADY),
		.output_TDATA(output_TDATA)
	);
	
endmodule : ptranspose