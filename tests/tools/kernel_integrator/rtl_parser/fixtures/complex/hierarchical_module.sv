////////////////////////////////////////////////////////////////////////////
// Hierarchical module with nested instantiations
//
// This fixture demonstrates:
// - Multiple module definitions in one file
// - Module instantiation and hierarchy
// - TOP_MODULE pragma to select main module
// - Parameter passing through hierarchy
// - Interface connections between modules
////////////////////////////////////////////////////////////////////////////

// @brainsmith TOP_MODULE hierarchical_top
module hierarchical_top #(
    // Top-level parameters
    parameter integer NUM_ENGINES = 4,
    parameter integer ENGINE_WIDTH = 32,
    parameter integer FIFO_DEPTH = 64,
    
    // Interface parameters
    parameter integer s_axis_input_BDIM = 128,
    parameter integer s_axis_input_SDIM = 1024,
    parameter integer m_axis_output_BDIM = 128,
    
    // Derived parameters for submodules
    parameter integer TOTAL_WIDTH = NUM_ENGINES * ENGINE_WIDTH,
    parameter integer ADDR_WIDTH = $clog2(FIFO_DEPTH)
) (
    // Global control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Main data input
    input wire [ENGINE_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    // Main data output
    output wire [ENGINE_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready,
    
    // Control interface
    input wire [31:0] ctrl_reg,
    output wire [31:0] status_reg
);

    // Internal signals
    wire [ENGINE_WIDTH-1:0] engine_in_data [0:NUM_ENGINES-1];
    wire [NUM_ENGINES-1:0] engine_in_valid;
    wire [NUM_ENGINES-1:0] engine_in_ready;
    
    wire [ENGINE_WIDTH-1:0] engine_out_data [0:NUM_ENGINES-1];
    wire [NUM_ENGINES-1:0] engine_out_valid;
    wire [NUM_ENGINES-1:0] engine_out_ready;
    
    wire [ENGINE_WIDTH-1:0] fifo_din;
    wire fifo_wr_en;
    wire fifo_full;
    wire [ENGINE_WIDTH-1:0] fifo_dout;
    wire fifo_rd_en;
    wire fifo_empty;
    
    // Input distributor instance
    input_distributor #(
        .NUM_OUTPUTS(NUM_ENGINES),
        .DATA_WIDTH(ENGINE_WIDTH)
    ) i_distributor (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .s_data(s_axis_input_tdata),
        .s_valid(s_axis_input_tvalid),
        .s_ready(s_axis_input_tready),
        .m_data(engine_in_data),
        .m_valid(engine_in_valid),
        .m_ready(engine_in_ready),
        .config(ctrl_reg[NUM_ENGINES-1:0])
    );
    
    // Processing engine instances
    genvar i;
    generate
        for (i = 0; i < NUM_ENGINES; i = i + 1) begin : gen_engines
            processing_engine #(
                .DATA_WIDTH(ENGINE_WIDTH),
                .ENGINE_ID(i)
            ) i_engine (
                .clk(ap_clk),
                .rst_n(ap_rst_n),
                .s_data(engine_in_data[i]),
                .s_valid(engine_in_valid[i]),
                .s_ready(engine_in_ready[i]),
                .m_data(engine_out_data[i]),
                .m_valid(engine_out_valid[i]),
                .m_ready(engine_out_ready[i]),
                .enable(ctrl_reg[16 + i])
            );
        end
    endgenerate
    
    // Output arbiter instance
    output_arbiter #(
        .NUM_INPUTS(NUM_ENGINES),
        .DATA_WIDTH(ENGINE_WIDTH)
    ) i_arbiter (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .s_data(engine_out_data),
        .s_valid(engine_out_valid),
        .s_ready(engine_out_ready),
        .m_data(fifo_din),
        .m_valid(fifo_wr_en),
        .m_ready(!fifo_full)
    );
    
    // Output FIFO instance
    sync_fifo #(
        .DATA_WIDTH(ENGINE_WIDTH),
        .DEPTH(FIFO_DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) i_out_fifo (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .din(fifo_din),
        .wr_en(fifo_wr_en && !fifo_full),
        .full(fifo_full),
        .dout(fifo_dout),
        .rd_en(fifo_rd_en),
        .empty(fifo_empty)
    );
    
    // Connect FIFO to output
    assign m_axis_output_tdata = fifo_dout;
    assign m_axis_output_tvalid = !fifo_empty;
    assign fifo_rd_en = m_axis_output_tready && !fifo_empty;
    
    // Status register
    assign status_reg = {
        16'h0,
        fifo_full,
        fifo_empty,
        2'b0,
        engine_out_valid,
        engine_in_valid,
        2'b0,
        s_axis_input_tvalid,
        m_axis_output_tvalid
    };

endmodule

// Input distributor submodule
module input_distributor #(
    parameter integer NUM_OUTPUTS = 4,
    parameter integer DATA_WIDTH = 32
) (
    input wire clk,
    input wire rst_n,
    
    // Slave interface
    input wire [DATA_WIDTH-1:0] s_data,
    input wire s_valid,
    output wire s_ready,
    
    // Master interfaces
    output wire [DATA_WIDTH-1:0] m_data [0:NUM_OUTPUTS-1],
    output wire [NUM_OUTPUTS-1:0] m_valid,
    input wire [NUM_OUTPUTS-1:0] m_ready,
    
    // Configuration
    input wire [NUM_OUTPUTS-1:0] config
);

    reg [$clog2(NUM_OUTPUTS)-1:0] current_output;
    wire selected_ready = m_ready[current_output];
    
    // Distribute data to selected output
    genvar i;
    generate
        for (i = 0; i < NUM_OUTPUTS; i = i + 1) begin : gen_outputs
            assign m_data[i] = s_data;
            assign m_valid[i] = s_valid && (current_output == i) && config[i];
        end
    endgenerate
    
    assign s_ready = selected_ready && config[current_output];
    
    // Round-robin arbitration
    always @(posedge clk) begin
        if (!rst_n) begin
            current_output <= 0;
        end else if (s_valid && s_ready) begin
            if (current_output == NUM_OUTPUTS - 1)
                current_output <= 0;
            else
                current_output <= current_output + 1;
        end
    end

endmodule

// Processing engine submodule
module processing_engine #(
    parameter integer DATA_WIDTH = 32,
    parameter integer ENGINE_ID = 0
) (
    input wire clk,
    input wire rst_n,
    
    // Slave interface
    input wire [DATA_WIDTH-1:0] s_data,
    input wire s_valid,
    output wire s_ready,
    
    // Master interface
    output reg [DATA_WIDTH-1:0] m_data,
    output reg m_valid,
    input wire m_ready,
    
    // Control
    input wire enable
);

    // Simple processing - add engine ID to data
    always @(posedge clk) begin
        if (!rst_n) begin
            m_data <= 0;
            m_valid <= 0;
        end else if (enable && s_valid && s_ready) begin
            m_data <= s_data + ENGINE_ID;
            m_valid <= 1;
        end else if (m_valid && m_ready) begin
            m_valid <= 0;
        end
    end
    
    assign s_ready = enable && (!m_valid || m_ready);

endmodule

// Output arbiter submodule
module output_arbiter #(
    parameter integer NUM_INPUTS = 4,
    parameter integer DATA_WIDTH = 32
) (
    input wire clk,
    input wire rst_n,
    
    // Slave interfaces
    input wire [DATA_WIDTH-1:0] s_data [0:NUM_INPUTS-1],
    input wire [NUM_INPUTS-1:0] s_valid,
    output reg [NUM_INPUTS-1:0] s_ready,
    
    // Master interface
    output reg [DATA_WIDTH-1:0] m_data,
    output reg m_valid,
    input wire m_ready
);

    // Priority arbiter - lower index has higher priority
    integer i;
    reg [$clog2(NUM_INPUTS)-1:0] selected;
    
    always @(*) begin
        selected = 0;
        for (i = NUM_INPUTS - 1; i >= 0; i = i - 1) begin
            if (s_valid[i]) selected = i;
        end
    end
    
    always @(posedge clk) begin
        if (!rst_n) begin
            m_data <= 0;
            m_valid <= 0;
            s_ready <= 0;
        end else begin
            if (|s_valid && (!m_valid || m_ready)) begin
                m_data <= s_data[selected];
                m_valid <= 1;
                s_ready <= 0;
                s_ready[selected] <= 1;
            end else if (m_valid && m_ready) begin
                m_valid <= 0;
                s_ready <= 0;
            end else begin
                s_ready <= 0;
            end
        end
    end

endmodule

// FIFO submodule
module sync_fifo #(
    parameter integer DATA_WIDTH = 32,
    parameter integer DEPTH = 64,
    parameter integer ADDR_WIDTH = 6
) (
    input wire clk,
    input wire rst_n,
    
    // Write interface
    input wire [DATA_WIDTH-1:0] din,
    input wire wr_en,
    output wire full,
    
    // Read interface
    output reg [DATA_WIDTH-1:0] dout,
    input wire rd_en,
    output wire empty
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_WIDTH:0] wr_ptr, rd_ptr;
    wire [ADDR_WIDTH:0] wr_ptr_next, rd_ptr_next;
    
    assign wr_ptr_next = wr_ptr + 1;
    assign rd_ptr_next = rd_ptr + 1;
    
    assign full = (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]) && 
                  (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]);
    assign empty = (wr_ptr == rd_ptr);
    
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            dout <= 0;
        end else begin
            if (wr_en && !full) begin
                mem[wr_ptr[ADDR_WIDTH-1:0]] <= din;
                wr_ptr <= wr_ptr_next;
            end
            
            if (rd_en && !empty) begin
                dout <= mem[rd_ptr[ADDR_WIDTH-1:0]];
                rd_ptr <= rd_ptr_next;
            end
        end
    end

endmodule