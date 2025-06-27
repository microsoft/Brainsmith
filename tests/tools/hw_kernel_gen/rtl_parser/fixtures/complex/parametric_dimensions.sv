////////////////////////////////////////////////////////////////////////////
// Module with complex parameter relationships and computed dimensions
//
// This fixture demonstrates:
// - Parameters computed from other parameters
// - DERIVED_PARAMETER pragmas
// - Complex dimension calculations
// - Conditional parameter usage
// - Parameter aliasing
////////////////////////////////////////////////////////////////////////////

module parametric_dimensions #(
    // Base configuration
    parameter integer BASE_SIZE = 16,       // @brainsmith ALIAS BASE_SIZE BaseChannels
    parameter integer SCALE_FACTOR = 4,     // @brainsmith ALIAS SCALE_FACTOR ParallelismLevel
    parameter integer NUM_LAYERS = 3,
    
    // Computed dimensions
    parameter integer LAYER1_SIZE = BASE_SIZE * SCALE_FACTOR,      // @brainsmith DERIVED_PARAMETER LAYER1_SIZE BASE_SIZE * SCALE_FACTOR
    parameter integer LAYER2_SIZE = LAYER1_SIZE * 2,               // @brainsmith DERIVED_PARAMETER LAYER2_SIZE LAYER1_SIZE * 2
    parameter integer LAYER3_SIZE = LAYER2_SIZE * 2,               // @brainsmith DERIVED_PARAMETER LAYER3_SIZE LAYER2_SIZE * 2
    
    // Input dimensions (computed)
    parameter integer INPUT_HEIGHT = 224,
    parameter integer INPUT_WIDTH = 224,
    parameter integer INPUT_PIXELS = 50176,  // @brainsmith DERIVED_PARAMETER INPUT_PIXELS INPUT_HEIGHT * INPUT_WIDTH
    
    // Tiling parameters
    parameter integer TILE_SIZE = 16,
    parameter integer TILES_H = 14,         // @brainsmith DERIVED_PARAMETER TILES_H INPUT_HEIGHT / TILE_SIZE
    parameter integer TILES_W = 14,         // @brainsmith DERIVED_PARAMETER TILES_W INPUT_WIDTH / TILE_SIZE
    parameter integer TOTAL_TILES = 196,    // @brainsmith DERIVED_PARAMETER TOTAL_TILES TILES_H * TILES_W
    
    // Buffer sizes (complex calculations)
    parameter integer L1_BUFFER_SIZE = 4096,    // @brainsmith DERIVED_PARAMETER L1_BUFFER_SIZE TILE_SIZE * TILE_SIZE * BASE_SIZE
    parameter integer L2_BUFFER_SIZE = 16384,   // @brainsmith DERIVED_PARAMETER L2_BUFFER_SIZE L1_BUFFER_SIZE * SCALE_FACTOR
    parameter integer L3_BUFFER_SIZE = 65536,   // @brainsmith DERIVED_PARAMETER L3_BUFFER_SIZE L2_BUFFER_SIZE * SCALE_FACTOR
    
    // Memory parameters
    parameter integer ADDR_WIDTH = 16,
    parameter integer MEM_DEPTH = 65536,        // @brainsmith DERIVED_PARAMETER MEM_DEPTH 1 << ADDR_WIDTH
    parameter integer CACHE_LINES = 1024,
    parameter integer CACHE_SIZE = 32768,       // @brainsmith DERIVED_PARAMETER CACHE_SIZE CACHE_LINES * 32
    
    // Interface widths (conditional)
    parameter integer USE_WIDE_BUS = 1,
    parameter integer NARROW_WIDTH = 32,
    parameter integer WIDE_WIDTH = 128,
    parameter integer DATA_WIDTH = 32,         // @brainsmith DERIVED_PARAMETER DATA_WIDTH USE_WIDE_BUS ? WIDE_WIDTH : NARROW_WIDTH
    
    // Processing elements
    parameter integer MAX_PE = 32,
    parameter integer ACTIVE_PE = 16,          // @brainsmith DERIVED_PARAMETER ACTIVE_PE min(MAX_PE, SCALE_FACTOR * 4)
    
    // Interface BDIM/SDIM parameters
    parameter integer s_axis_layer1_BDIM = LAYER1_SIZE,
    parameter integer s_axis_layer1_SDIM = INPUT_PIXELS,
    parameter integer s_axis_layer2_BDIM = LAYER2_SIZE,
    parameter integer s_axis_layer2_SDIM = TOTAL_TILES,
    parameter integer s_axis_layer3_BDIM = LAYER3_SIZE,
    parameter integer s_axis_layer3_SDIM = 1,
    parameter integer m_axis_output_BDIM = LAYER3_SIZE
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Layer 1 input - uses computed dimensions
    input wire [DATA_WIDTH-1:0] s_axis_layer1_tdata,
    input wire s_axis_layer1_tvalid,
    output wire s_axis_layer1_tready,
    
    // Layer 2 input - uses derived dimensions
    input wire [DATA_WIDTH-1:0] s_axis_layer2_tdata,
    input wire s_axis_layer2_tvalid,
    output wire s_axis_layer2_tready,
    
    // Layer 3 input - uses complex calculations
    input wire [DATA_WIDTH-1:0] s_axis_layer3_tdata,
    input wire s_axis_layer3_tvalid,
    output wire s_axis_layer3_tready,
    
    // Tiled weights - dimensions based on tiling
    // @brainsmith WEIGHT s_axis_tiled_w
    // @brainsmith BDIM s_axis_tiled_w [TILE_SIZE, TILE_SIZE, BASE_SIZE]
    // @brainsmith SDIM s_axis_tiled_w TOTAL_TILES
    input wire [15:0] s_axis_tiled_w_tdata,
    input wire s_axis_tiled_w_tvalid,
    output wire s_axis_tiled_w_tready,
    
    // Output - final layer size
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    
    // Memory interface - uses computed address width
    output wire [ADDR_WIDTH-1:0] mem_addr,
    output wire mem_we,
    output wire [DATA_WIDTH-1:0] mem_wdata,
    input wire [DATA_WIDTH-1:0] mem_rdata
);

    // Internal buffers sized by parameters
    reg [DATA_WIDTH-1:0] l1_buffer [0:L1_BUFFER_SIZE/4-1];
    reg [DATA_WIDTH-1:0] l2_buffer [0:L2_BUFFER_SIZE/4-1];
    reg [DATA_WIDTH-1:0] l3_buffer [0:L3_BUFFER_SIZE/4-1];
    
    // Processing element array
    reg [DATA_WIDTH-1:0] pe_results [0:ACTIVE_PE-1];
    
    // Address generation using computed parameters
    reg [ADDR_WIDTH-1:0] addr_counter;
    reg [$clog2(TOTAL_TILES)-1:0] tile_counter;
    reg [$clog2(NUM_LAYERS)-1:0] layer_counter;
    
    // State machine
    localparam IDLE = 0, L1_PROC = 1, L2_PROC = 2, L3_PROC = 3;
    reg [1:0] state;
    
    // Processing logic
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            state <= IDLE;
            addr_counter <= '0;
            tile_counter <= '0;
            layer_counter <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (s_axis_layer1_tvalid) state <= L1_PROC;
                end
                
                L1_PROC: begin
                    if (s_axis_layer1_tvalid && s_axis_layer1_tready) begin
                        // Process with LAYER1_SIZE channels
                        addr_counter <= addr_counter + 1;
                        if (addr_counter == L1_BUFFER_SIZE/4 - 1) begin
                            addr_counter <= '0;
                            state <= L2_PROC;
                        end
                    end
                end
                
                L2_PROC: begin
                    if (s_axis_layer2_tvalid && s_axis_layer2_tready) begin
                        // Process with LAYER2_SIZE channels
                        tile_counter <= tile_counter + 1;
                        if (tile_counter == TOTAL_TILES - 1) begin
                            tile_counter <= '0;
                            state <= L3_PROC;
                        end
                    end
                end
                
                L3_PROC: begin
                    if (s_axis_layer3_tvalid && s_axis_layer3_tready) begin
                        // Process with LAYER3_SIZE channels
                        layer_counter <= layer_counter + 1;
                        if (layer_counter == NUM_LAYERS - 1) begin
                            layer_counter <= '0;
                            state <= IDLE;
                        end
                    end
                end
            endcase
        end
    end
    
    // Memory address generation using parameters
    assign mem_addr = (state == L1_PROC) ? addr_counter :
                     (state == L2_PROC) ? {tile_counter, addr_counter[7:0]} :
                     {layer_counter, addr_counter[13:0]};
    
    assign mem_we = (state != IDLE) && (addr_counter[1:0] == 2'b00);
    assign mem_wdata = (state == L1_PROC) ? s_axis_layer1_tdata :
                      (state == L2_PROC) ? s_axis_layer2_tdata :
                      s_axis_layer3_tdata;
    
    // Output based on final layer size
    assign m_axis_output_tdata = pe_results[0];
    assign m_axis_output_tvalid = (state == L3_PROC);
    
    // Ready signals based on state
    assign s_axis_layer1_tready = (state == L1_PROC) && m_axis_output_tready;
    assign s_axis_layer2_tready = (state == L2_PROC) && m_axis_output_tready;
    assign s_axis_layer3_tready = (state == L3_PROC) && m_axis_output_tready;
    assign s_axis_tiled_w_tready = (state != IDLE) && m_axis_output_tready;
    
    // Initialize PE results
    integer i;
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            for (i = 0; i < ACTIVE_PE; i = i + 1) begin
                pe_results[i] <= '0;
            end
        end else if (state != IDLE) begin
            // Simple accumulation using computed PE count
            for (i = 0; i < ACTIVE_PE; i = i + 1) begin
                if (i < SCALE_FACTOR * 4) begin
                    pe_results[i] <= pe_results[i] + mem_rdata;
                end
            end
        end
    end

endmodule