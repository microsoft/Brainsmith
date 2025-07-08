############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""RTL patterns for AST parser testing.

This module provides pre-defined RTL patterns focused on testing the
tree-sitter AST parser, including syntax edge cases, malformed modules,
and complex language constructs.
"""

from typing import List, Dict, Optional
from .rtl_builder import RTLBuilder


class ASTPatterns:
    """RTL patterns for AST parser testing."""
    
    @staticmethod
    def syntax_edge_case(case_type: str) -> str:
        """Generate SystemVerilog syntax edge cases.
        
        Args:
            case_type: Type of edge case
                - "escaped_identifiers": \\escaped names
                - "real_numbers": Real number literals
                - "attributes": (* synthesis *) attributes
                - "assertions": SVA constructs
                - "interfaces": SV interfaces
                - "packages": Package imports
        """
        if case_type == "escaped_identifiers":
            return """
            module \\escaped-module-name! #(
                parameter \\strange+param = 32
            ) (
                input wire clk,
                input wire \\data.in[31:0] ,
                output wire \\result@out 
            );
                wire \\internal*signal ;
                assign \\result@out = \\data.in[31:0] ;
            endmodule
            """
        
        elif case_type == "real_numbers":
            return """
            module real_numbers #(
                parameter real GAIN = 1.5,
                parameter real OFFSET = -0.25e-3,
                parameter realtime DELAY = 100.0ns
            ) (
                input wire clk,
                input real data_in,
                output real data_out
            );
                real internal = 3.14159;
                assign data_out = data_in * GAIN + OFFSET;
            endmodule
            """
        
        elif case_type == "attributes":
            return """
            (* keep = "true" *)
            module attributed_module #(
                (* DONT_TOUCH = "yes" *) parameter WIDTH = 32
            ) (
                (* mark_debug = "true" *) input wire clk,
                (* async_reg = "true" *) input wire [WIDTH-1:0] data,
                output wire [WIDTH-1:0] result
            );
                (* ram_style = "block" *) reg [WIDTH-1:0] memory [0:255];
                (* full_case, parallel_case *) 
                always @(posedge clk) begin
                    result <= data;
                end
            endmodule
            """
        
        elif case_type == "assertions":
            return """
            module with_assertions (
                input wire clk,
                input wire req,
                output wire ack
            );
                // Immediate assertion
                always @(posedge clk) begin
                    assert (req |-> ##[1:3] ack) 
                    else $error("Ack not received within 3 cycles");
                end
                
                // Property definition
                property req_ack_protocol;
                    @(posedge clk) req |-> ##[1:3] ack;
                endproperty
                
                // Assert property
                assert property (req_ack_protocol);
            endmodule
            """
        
        elif case_type == "interfaces":
            return """
            interface axi_if #(parameter WIDTH = 32);
                logic [WIDTH-1:0] data;
                logic valid;
                logic ready;
                
                modport master (output data, valid, input ready);
                modport slave (input data, valid, output ready);
            endinterface
            
            module using_interface (
                input wire clk,
                axi_if.slave s_axi,
                axi_if.master m_axi
            );
                assign m_axi.data = s_axi.data;
                assign m_axi.valid = s_axi.valid;
                assign s_axi.ready = m_axi.ready;
            endmodule
            """
        
        elif case_type == "packages":
            return """
            package my_pkg;
                typedef logic [7:0] byte_t;
                typedef enum logic [1:0] {
                    IDLE = 2'b00,
                    BUSY = 2'b01,
                    DONE = 2'b10
                } state_t;
                
                function automatic int add(int a, int b);
                    return a + b;
                endfunction
            endpackage
            
            module using_package 
                import my_pkg::*;
            (
                input wire clk,
                input byte_t data_in,
                output state_t state
            );
                always_ff @(posedge clk) begin
                    state <= IDLE;
                end
            endmodule
            """
        
        else:
            raise ValueError(f"Unknown edge case type: {case_type}")
    
    @staticmethod
    def malformed_module(error_type: str) -> str:
        """Generate intentionally malformed RTL.
        
        Args:
            error_type: Type of malformation
                - "missing_semicolon": Missing semicolons
                - "unclosed_block": Unclosed begin/end
                - "mismatched_parens": Parenthesis mismatch
                - "invalid_syntax": General syntax errors
                - "incomplete_module": Module cut off
        """
        if error_type == "missing_semicolon":
            return """
            module missing_semi (
                input wire clk,
                input wire [31:0] data
                output wire [31:0] result  // Missing comma
            );
                reg [31:0] temp
                
                always @(posedge clk) begin
                    temp <= data  // Missing semicolon
                    result <= temp
                end
            endmodule
            """
        
        elif error_type == "unclosed_block":
            return """
            module unclosed_block (
                input wire clk,
                input wire [7:0] data
            );
                always @(posedge clk) begin
                    if (data > 0) begin
                        // Do something
                        data <= data - 1;
                    // Missing end
                end
            endmodule
            """
        
        elif error_type == "mismatched_parens":
            return """
            module mismatched_parens #(
                parameter WIDTH = 32
            ) (
                input wire clk,
                input wire [WIDTH-1:0] data,
                output wire [(WIDTH*2-1:0] result  // Missing )
            );
                assign result = {data, data)};  // Wrong bracket
            endmodule
            """
        
        elif error_type == "invalid_syntax":
            return """
            module invalid_syntax (
                input wire clk,
                input wire 32'data,  // Invalid width spec
                output wire result[31:0]  // Wrong array syntax
            );
                always posedge clk begin  // Missing @()
                    result = data;
                end
            endmodule
            """
        
        elif error_type == "incomplete_module":
            return """
            module incomplete (
                input wire clk,
                input wire [31:0] data_in,
                output wire [31:0] data_out
            );
                reg [31:0] buffer;
                
                always @(posedge clk) begin
                    buffer <= data_in;
                    // Module ends abruptly
            """
        
        else:
            raise ValueError(f"Unknown error type: {error_type}")
    
    @staticmethod
    def nested_constructs() -> str:
        """Create deeply nested language constructs."""
        return """
        module deeply_nested #(
            parameter DEPTH = 4
        ) (
            input wire clk,
            input wire rst,
            input wire [7:0] data_in,
            output reg [7:0] data_out
        );
            genvar i, j;
            
            // Nested generate blocks
            generate
                for (i = 0; i < DEPTH; i = i + 1) begin : gen_outer
                    for (j = 0; j < DEPTH; j = j + 1) begin : gen_inner
                        if (i == j) begin : gen_diagonal
                            reg [7:0] diag_reg;
                            
                            always @(posedge clk) begin
                                if (rst) begin
                                    diag_reg <= 0;
                                end else begin
                                    if (data_in > 0) begin
                                        if (data_in < 255) begin
                                            diag_reg <= data_in + i + j;
                                        end else begin
                                            diag_reg <= 255;
                                        end
                                    end else begin
                                        diag_reg <= 0;
                                    end
                                end
                            end
                        end
                    end
                end
            endgenerate
            
            // Nested case statements
            always @(posedge clk) begin
                case (data_in[7:6])
                    2'b00: begin
                        case (data_in[5:4])
                            2'b00: data_out <= 8'h00;
                            2'b01: data_out <= 8'h40;
                            2'b10: data_out <= 8'h80;
                            2'b11: data_out <= 8'hC0;
                        endcase
                    end
                    2'b01: begin
                        case (data_in[3:2])
                            2'b00: data_out <= 8'h10;
                            2'b01: data_out <= 8'h50;
                            default: data_out <= 8'hFF;
                        endcase
                    end
                    default: data_out <= 8'hFF;
                endcase
            end
        endmodule
        """
    
    @staticmethod
    def preprocessor_directives() -> str:
        """Module with preprocessor directives."""
        return """
        `define DATA_WIDTH 32
        `define FIFO_DEPTH 16
        `define DEBUG
        
        `ifdef SYNTHESIS
            `define CLK_PERIOD 10
        `else
            `define CLK_PERIOD 100
        `endif
        
        module with_preprocessor (
            input wire clk,
            input wire rst,
            `ifdef DEBUG
                input wire debug_en,
                output wire [7:0] debug_out,
            `endif
            input wire [`DATA_WIDTH-1:0] data_in,
            output wire [`DATA_WIDTH-1:0] data_out
        );
            reg [`DATA_WIDTH-1:0] fifo [`FIFO_DEPTH-1:0];
            
            `include "some_functions.vh"
            
            `ifdef DEBUG
                assign debug_out = fifo[0][7:0];
            `endif
            
            `ifndef SYNTHESIS
                initial begin
                    $display("Simulation mode");
                end
            `endif
            
            assign data_out = fifo[0];
            
        endmodule
        
        `undef DATA_WIDTH
        `undef FIFO_DEPTH
        """
    
    @staticmethod
    def unicode_identifiers() -> str:
        """Module with Unicode characters in identifiers."""
        return """
        module unicode_test_αβγ (
            input wire clk_φ,
            input wire [31:0] data_π,
            output wire [31:0] result_Σ
        );
            // Unicode in comments: ∑∏∫∂∇
            reg [31:0] temp_Δ;
            wire enable_μ;
            
            parameter real α = 1.414;
            parameter real β = 3.14159;
            
            always @(posedge clk_φ) begin
                temp_Δ <= data_π * 2;
            end
            
            assign result_Σ = temp_Δ;
        endmodule
        """
    
    @staticmethod
    def complex_port_declarations() -> str:
        """Module with complex port declaration styles."""
        return """
        module complex_ports 
        #(
            parameter int WIDTH = 32,
            parameter type data_t = logic [WIDTH-1:0]
        )
        (
            // Interface ports
            interface.master axi_m,
            interface.slave axi_s,
            
            // Parameterized types
            input data_t data_in,
            output data_t data_out,
            
            // Multidimensional
            input logic [7:0] array_2d [0:3][0:3],
            output logic [WIDTH-1:0] array_3d [0:1][0:1][0:1],
            
            // Packed and unpacked
            input logic [3:0][7:0] packed_array,
            output logic [7:0] unpacked_array [3:0],
            
            // With modports
            axi_if.master m_axi_port,
            axi_if.slave s_axi_port,
            
            // Traditional style mixed in
            input wire clk,
            input wire rst_n
        );
            
            // Port expressions in body
            assign data_out = data_in;
            
        endmodule
        """
    
    @staticmethod
    def macro_expanded_module() -> str:
        """Module that would result from macro expansion."""
        return (RTLBuilder()
                .module("macro_expanded")
                .comment("`define CREATE_REG(name, width) reg [(width)-1:0] name")
                .comment("`CREATE_REG(my_reg, 32)")
                .parameter("WIDTH", "32")
                .port("clk", "input")
                .port("data_in", "input", "WIDTH-1:0")
                .port("data_out", "output", "WIDTH-1:0")
                .body("// Expanded from macro:")
                .body("reg [32-1:0] my_reg;")
                .body("")
                .body("always @(posedge clk) begin")
                .body("    my_reg <= data_in;")
                .body("end")
                .body("")
                .body("assign data_out = my_reg;")
                .build())
    
    @staticmethod
    def mixed_ansi_nonansi() -> str:
        """Module mixing ANSI and non-ANSI port styles (invalid)."""
        return """
        module mixed_ports (
            input wire clk,  // ANSI style
            data_in,         // Non-ANSI style - invalid mix
            output wire data_out
        );
            input wire [31:0] data_in;  // Port declaration in body
            
            assign data_out = data_in[0];
        endmodule
        """
    
    @staticmethod
    def generate_blocks() -> str:
        """Complex generate block constructs."""
        return """
        module generate_test #(
            parameter NUM_INST = 4,
            parameter WIDTH = 8
        ) (
            input wire clk,
            input wire [WIDTH-1:0] data_in [0:NUM_INST-1],
            output wire [WIDTH-1:0] data_out [0:NUM_INST-1]
        );
            genvar i;
            
            // Generate with different styles
            generate
                // For loop generate
                for (i = 0; i < NUM_INST; i = i + 1) begin : gen_for
                    reg [WIDTH-1:0] buffer;
                    
                    always @(posedge clk) begin
                        buffer <= data_in[i];
                    end
                    
                    assign data_out[i] = buffer;
                end
                
                // If generate
                if (NUM_INST > 2) begin : gen_if
                    reg extra_flag;
                    always @(posedge clk) begin
                        extra_flag <= 1'b1;
                    end
                end
                
                // Case generate
                case (WIDTH)
                    8: begin : gen_case_8
                        reg [7:0] width_specific_reg;
                    end
                    16: begin : gen_case_16
                        reg [15:0] width_specific_reg;
                    end
                    default: begin : gen_case_default
                        reg [31:0] width_specific_reg;
                    end
                endcase
            endgenerate
        endmodule
        """
    
    @staticmethod
    def partial_module_stages(stage: str) -> str:
        """Module at various stages of completion for incremental parsing.
        
        Args:
            stage: Parsing stage
                - "header_only": Just module declaration
                - "params_only": With parameters
                - "ports_partial": Incomplete port list
                - "body_partial": Incomplete body
        """
        if stage == "header_only":
            return "module partial_module"
        
        elif stage == "params_only":
            return """
            module partial_module #(
                parameter WIDTH = 32,
                parameter DEPTH = 16
            )"""
        
        elif stage == "ports_partial":
            return """
            module partial_module #(
                parameter WIDTH = 32
            ) (
                input wire clk,
                input wire [WIDTH-1:0] data,
                // More ports would go here
            """
        
        elif stage == "body_partial":
            return """
            module partial_module (
                input wire clk,
                input wire [31:0] data_in,
                output reg [31:0] data_out
            );
                reg [31:0] buffer;
                
                always @(posedge clk) begin
                    buffer <= data_in;
                    // Rest of logic would go here
            """
        
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    @staticmethod
    def comment_variations() -> str:
        """Module with various comment styles and positions."""
        return """
        // Single line comment before module
        /* Multi-line comment
           before module */
        module /* inline comment */ comment_test // trailing comment
        #(
            parameter WIDTH = 32  // Parameter comment
            /* Multi-line parameter
               comment */
        ) (
            // Port comments
            input wire clk,  /* clock signal */
            input wire /* comment in declaration */ [WIDTH-1:0] data,
            output wire [WIDTH-1:0] /* another comment */ result
        );
            
            // Body comments
            /* Multi-line
               body comment */
            
            always @(posedge clk) begin  // Process comment
                result <= data;  /* Assignment comment */
            end
            
            // Nested comments (usually not allowed)
            /* Outer comment /* nested comment */ end outer */
            
        endmodule  // End module comment
        """