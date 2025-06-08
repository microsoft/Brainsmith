/**
 * Vector Dot Product Accelerator
 * 
 * This module implements a configurable vector dot product operation optimized for
 * neural network inference. It demonstrates key HKG features including:
 * - AXI-Stream interfaces with backpressure
 * - Dataflow pragmas for automatic optimization
 * - Configurable parallelism and precision
 * - Performance optimization annotations
 */

module vector_dot_product #(
    // Configuration parameters
    parameter VECTOR_SIZE = 768,           // Vector dimension (typical BERT hidden size)
    parameter DATA_WIDTH = 8,              // Input data precision 
    parameter RESULT_WIDTH = 32,           // Accumulator width
    parameter PARALLELISM = 8              // Number of parallel multiply-accumulate units
)(
    input wire clk,
    input wire rst_n,
    
    // AXI-Stream Input A (Query Vector)
    (* dataflow interface_type="INPUT" qDim=768 tDim=96 sDim=8 dtype="INT8" 
       protocol="AXI_STREAM" role="primary_input" *)
    input wire [DATA_WIDTH*PARALLELISM-1:0] s_axis_a_tdata,
    input wire s_axis_a_tvalid,
    output reg s_axis_a_tready,
    input wire s_axis_a_tlast,
    
    // AXI-Stream Input B (Key Vector)  
    (* dataflow interface_type="INPUT" qDim=768 tDim=96 sDim=8 dtype="INT8"
       protocol="AXI_STREAM" role="secondary_input" *)
    input wire [DATA_WIDTH*PARALLELISM-1:0] s_axis_b_tdata,
    input wire s_axis_b_tvalid,
    output reg s_axis_b_tready,
    input wire s_axis_b_tlast,
    
    // AXI-Stream Output (Dot Product Result)
    (* dataflow interface_type="OUTPUT" qDim=1 tDim=1 sDim=1 dtype="INT32"
       protocol="AXI_STREAM" role="result_output" *)
    output reg [RESULT_WIDTH-1:0] m_axis_result_tdata,
    output reg m_axis_result_tvalid,
    input wire m_axis_result_tready,
    output reg m_axis_result_tlast,
    
    // Configuration Interface (AXI-Lite)
    (* dataflow interface_type="CONFIG" protocol="AXI_LITE" *)
    input wire [31:0] config_vector_size,
    input wire [31:0] config_scale_factor,
    input wire config_enable,
    
    // Status and Control
    (* dataflow interface_type="CONTROL" *)
    output reg computation_done,
    output reg [15:0] cycle_count,
    input wire reset_counters
);

// Performance annotation pragmas
(* performance target="latency" value=96 unit="cycles" *)
(* performance target="throughput" value=1 unit="samples_per_cycle" *)
(* resource usage="conservative" *)

// Internal signals
reg [DATA_WIDTH-1:0] vector_a [0:PARALLELISM-1];
reg [DATA_WIDTH-1:0] vector_b [0:PARALLELISM-1];
reg [RESULT_WIDTH-1:0] partial_products [0:PARALLELISM-1];
reg [RESULT_WIDTH-1:0] accumulator;
reg [$clog2(VECTOR_SIZE/PARALLELISM)-1:0] element_counter;
reg computation_active;

// State machine
typedef enum logic [2:0] {
    IDLE       = 3'b000,
    RECEIVING  = 3'b001,
    COMPUTING  = 3'b010,
    OUTPUTTING = 3'b011,
    DONE       = 3'b100
} state_t;

state_t current_state, next_state;

// Input data unpacking
genvar i;
generate
    for (i = 0; i < PARALLELISM; i = i + 1) begin : unpack_inputs
        always_ff @(posedge clk) begin
            if (s_axis_a_tvalid && s_axis_a_tready) begin
                vector_a[i] <= s_axis_a_tdata[i*DATA_WIDTH +: DATA_WIDTH];
            end
            if (s_axis_b_tvalid && s_axis_b_tready) begin
                vector_b[i] <= s_axis_b_tdata[i*DATA_WIDTH +: DATA_WIDTH];
            end
        end
    end
endgenerate

// Parallel multiply-accumulate units
generate
    for (i = 0; i < PARALLELISM; i = i + 1) begin : mac_units
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                partial_products[i] <= 0;
            end else if (computation_active) begin
                // Signed multiplication with accumulation
                partial_products[i] <= $signed(vector_a[i]) * $signed(vector_b[i]);
            end
        end
    end
endgenerate

// Accumulator tree
always_ff @(posedge clk) begin
    if (!rst_n) begin
        accumulator <= 0;
    end else if (computation_active) begin
        accumulator <= accumulator + 
                      partial_products[0] + partial_products[1] + 
                      partial_products[2] + partial_products[3] +
                      partial_products[4] + partial_products[5] + 
                      partial_products[6] + partial_products[7];
    end else if (current_state == IDLE) begin
        accumulator <= 0;
    end
end

// State machine
always_ff @(posedge clk) begin
    if (!rst_n) begin
        current_state <= IDLE;
    end else begin
        current_state <= next_state;
    end
end

always_comb begin
    next_state = current_state;
    
    case (current_state)
        IDLE: begin
            if (config_enable && s_axis_a_tvalid && s_axis_b_tvalid) begin
                next_state = RECEIVING;
            end
        end
        
        RECEIVING: begin
            if (s_axis_a_tvalid && s_axis_a_tready && 
                s_axis_b_tvalid && s_axis_b_tready) begin
                if (element_counter == (config_vector_size / PARALLELISM) - 1) begin
                    next_state = COMPUTING;
                end
            end
        end
        
        COMPUTING: begin
            // Single cycle computation due to pipelining
            next_state = OUTPUTTING;
        end
        
        OUTPUTTING: begin
            if (m_axis_result_tvalid && m_axis_result_tready) begin
                next_state = DONE;
            end
        end
        
        DONE: begin
            next_state = IDLE;
        end
        
        default: next_state = IDLE;
    endcase
end

// Control logic
always_ff @(posedge clk) begin
    if (!rst_n) begin
        element_counter <= 0;
        computation_active <= 0;
        s_axis_a_tready <= 0;
        s_axis_b_tready <= 0;
        m_axis_result_tvalid <= 0;
        m_axis_result_tlast <= 0;
        computation_done <= 0;
        cycle_count <= 0;
    end else begin
        
        // Default assignments
        s_axis_a_tready <= 0;
        s_axis_b_tready <= 0;
        m_axis_result_tvalid <= 0;
        computation_active <= 0;
        
        case (current_state)
            IDLE: begin
                element_counter <= 0;
                computation_done <= 0;
                if (!reset_counters) cycle_count <= cycle_count + 1;
                else cycle_count <= 0;
            end
            
            RECEIVING: begin
                s_axis_a_tready <= 1;
                s_axis_b_tready <= 1;
                computation_active <= 1;
                
                if (s_axis_a_tvalid && s_axis_a_tready && 
                    s_axis_b_tvalid && s_axis_b_tready) begin
                    element_counter <= element_counter + 1;
                end
                cycle_count <= cycle_count + 1;
            end
            
            COMPUTING: begin
                computation_active <= 1;
                cycle_count <= cycle_count + 1;
            end
            
            OUTPUTTING: begin
                m_axis_result_tvalid <= 1;
                m_axis_result_tlast <= 1;
                m_axis_result_tdata <= accumulator;
                cycle_count <= cycle_count + 1;
            end
            
            DONE: begin
                computation_done <= 1;
            end
        endcase
    end
end

// Assertions for verification
`ifdef SIMULATION
    // Verify dimensional constraints
    initial begin
        assert (PARALLELISM <= VECTOR_SIZE) 
            else $error("PARALLELISM cannot exceed VECTOR_SIZE");
        assert (VECTOR_SIZE % PARALLELISM == 0) 
            else $error("VECTOR_SIZE must be divisible by PARALLELISM");
    end
    
    // Performance monitoring
    always @(posedge clk) begin
        if (current_state == DONE) begin
            $display("Dot product completed in %d cycles", cycle_count);
        end
    end
`endif

endmodule