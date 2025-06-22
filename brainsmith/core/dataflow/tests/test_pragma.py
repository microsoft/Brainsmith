############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for pragma constraint system"""

import pytest
from brainsmith.core.dataflow.core.types import InterfaceDirection, INT16, INT32
from brainsmith.core.dataflow.core.interface import Interface
from brainsmith.core.dataflow.core.pragma import (
    Pragma, TiePragma, ConstrPragma,
    evaluate_expr, parse_pragma
)


class TestExpressionEvaluation:
    """Test expression evaluation with interfaces"""
    
    def setup_method(self):
        """Set up test interfaces and environment"""
        self.interfaces = {
            "vec": Interface(
                name="vec",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(512,),
                block_dims=(512,),
                stream_dims=(16,)
            ),
            "mat": Interface(
                name="mat",
                direction=InterfaceDirection.WEIGHT,
                dtype=INT16,
                tensor_dims=(256, 512),
                block_dims=(256, 512),
                stream_dims=(8, 16)
            ),
            "out": Interface(
                name="out",
                direction=InterfaceDirection.OUTPUT,
                dtype=INT32,
                tensor_dims=(256,),
                block_dims=(256,),
                stream_dims=(8,)
            )
        }
        
        self.env = {
            "SIMD": 16,
            "PE": 8,
            "BURST": 64,
            "ALIGN": 4
        }
    
    def test_simple_interface_reference(self):
        """Test simple interface name evaluation"""
        # Interface name returns total size
        assert evaluate_expr("vec", self.interfaces, self.env) == 512
        assert evaluate_expr("mat", self.interfaces, self.env) == 256 * 512
        assert evaluate_expr("out", self.interfaces, self.env) == 256
    
    def test_interface_subscript(self):
        """Test interface dimension access"""
        assert evaluate_expr("vec[0]", self.interfaces, self.env) == 512
        assert evaluate_expr("mat[0]", self.interfaces, self.env) == 256
        assert evaluate_expr("mat[1]", self.interfaces, self.env) == 512
        
        # Out of bounds
        with pytest.raises(ValueError):
            evaluate_expr("vec[1]", self.interfaces, self.env)
    
    def test_environment_symbols(self):
        """Test environment symbol evaluation"""
        assert evaluate_expr("SIMD", self.interfaces, self.env) == 16
        assert evaluate_expr("PE", self.interfaces, self.env) == 8
        assert evaluate_expr("BURST", self.interfaces, self.env) == 64
    
    def test_arithmetic_expressions(self):
        """Test arithmetic operations"""
        # Basic arithmetic
        assert evaluate_expr("SIMD + PE", self.interfaces, self.env) == 24
        assert evaluate_expr("SIMD * PE", self.interfaces, self.env) == 128
        assert evaluate_expr("BURST / ALIGN", self.interfaces, self.env) == 16
        
        # With interfaces
        assert evaluate_expr("mat[0] * 2", self.interfaces, self.env) == 512
        assert evaluate_expr("vec[0] / SIMD", self.interfaces, self.env) == 32
        
        # Complex expressions
        assert evaluate_expr("(mat[0] + mat[1]) * PE", self.interfaces, self.env) == (256 + 512) * 8
    
    def test_modulo_operations(self):
        """Test modulo operations for alignment"""
        assert evaluate_expr("vec[0] % SIMD", self.interfaces, self.env) == 0  # 512 % 16
        assert evaluate_expr("mat[0] % PE", self.interfaces, self.env) == 0   # 256 % 8
        assert evaluate_expr("17 % ALIGN", self.interfaces, self.env) == 1    # 17 % 4
    
    def test_invalid_expressions(self):
        """Test error handling for invalid expressions"""
        # Unknown identifier
        with pytest.raises(ValueError):
            evaluate_expr("unknown", self.interfaces, self.env)
        
        # Invalid syntax
        with pytest.raises(ValueError):
            evaluate_expr("vec +", self.interfaces, self.env)
        
        # Invalid subscript
        with pytest.raises(ValueError):
            evaluate_expr("vec[x]", self.interfaces, self.env)


class TestTiePragma:
    """Test TIE equality constraints"""
    
    def setup_method(self):
        """Set up test data"""
        self.interfaces = {
            "a": Interface(
                name="a",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(32, 64),
                block_dims=(32, 64),
                stream_dims=(1, 8)
            ),
            "b": Interface(
                name="b",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(64, 32),
                block_dims=(64, 32),
                stream_dims=(8, 1)
            ),
            "c": Interface(
                name="c",
                direction=InterfaceDirection.OUTPUT,
                dtype=INT16,
                tensor_dims=(32, 32),
                block_dims=(32, 32),
                stream_dims=(4, 4)
            )
        }
        self.env = {"CONST": 32}
    
    def test_tie_pragma_creation(self):
        """Test TiePragma creation"""
        pragma = TiePragma("a[0]", "b[1]")
        assert pragma.left_expr == "a[0]"
        assert pragma.right_expr == "b[1]"
        assert pragma.to_string() == "TIE a[0] b[1]"
    
    def test_tie_pragma_evaluation(self):
        """Test TiePragma evaluation"""
        # Equal dimensions
        pragma = TiePragma("a[0]", "b[1]")
        assert pragma.evaluate(self.interfaces, self.env) == True
        
        # Equal values
        pragma = TiePragma("a[0]", "CONST")
        assert pragma.evaluate(self.interfaces, self.env) == True
        
        # Not equal
        pragma = TiePragma("a[0]", "a[1]")
        assert pragma.evaluate(self.interfaces, self.env) == False
    
    def test_tie_complex_expressions(self):
        """Test TIE with complex expressions"""
        # Arithmetic expressions
        pragma = TiePragma("a[0] * 2", "a[1]")
        assert pragma.evaluate(self.interfaces, self.env) == True  # 32*2 == 64
        
        pragma = TiePragma("a[0] + c[0]", "b[1] + CONST")
        assert pragma.evaluate(self.interfaces, self.env) == True  # 32+32 == 32+32


class TestConstrPragma:
    """Test CONSTR unary constraints"""
    
    def setup_method(self):
        """Set up test data"""
        self.interfaces = {
            "data": Interface(
                name="data",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(128, 256),
                block_dims=(32, 64),
                stream_dims=(4, 8)
            )
        }
        self.env = {
            "SIMD": 16,
            "MIN_SIZE": 32,
            "MAX_SIZE": 512
        }
    
    def test_constr_pragma_creation(self):
        """Test ConstrPragma creation"""
        pragma = ConstrPragma("data[0]", "%", "SIMD")
        assert pragma.expr == "data[0]"
        assert pragma.op == "%"
        assert pragma.value == "SIMD"
        assert pragma.to_string() == "CONSTR data[0] % SIMD"
    
    def test_equality_constraint(self):
        """Test equality constraints"""
        pragma = ConstrPragma("data[0]", "=", 128)
        assert pragma.evaluate(self.interfaces, self.env) == True
        
        pragma = ConstrPragma("data[1]", "=", 128)
        assert pragma.evaluate(self.interfaces, self.env) == False
    
    def test_inequality_constraints(self):
        """Test inequality constraints"""
        # Greater than or equal
        pragma = ConstrPragma("data[0]", ">=", "MIN_SIZE")
        assert pragma.evaluate(self.interfaces, self.env) == True  # 128 >= 32
        
        # Less than or equal
        pragma = ConstrPragma("data[0]", "<=", "MAX_SIZE")
        assert pragma.evaluate(self.interfaces, self.env) == True  # 128 <= 512
        
        # Not equal
        pragma = ConstrPragma("data[0]", "!=", 64)
        assert pragma.evaluate(self.interfaces, self.env) == True  # 128 != 64
        
        # Strict inequalities
        pragma = ConstrPragma("data[0]", ">", 100)
        assert pragma.evaluate(self.interfaces, self.env) == True  # 128 > 100
        
        pragma = ConstrPragma("data[0]", "<", 200)
        assert pragma.evaluate(self.interfaces, self.env) == True  # 128 < 200
    
    def test_modulo_constraint(self):
        """Test modulo constraints for alignment"""
        # Aligned to SIMD
        pragma = ConstrPragma("data[0]", "%", "SIMD")
        assert pragma.evaluate(self.interfaces, self.env) == True  # 128 % 16 == 0
        
        # Not aligned
        pragma = ConstrPragma("data[0]", "%", 7)
        assert pragma.evaluate(self.interfaces, self.env) == False  # 128 % 7 != 0
    
    def test_unknown_symbol(self):
        """Test error on unknown symbol"""
        pragma = ConstrPragma("data[0]", "=", "UNKNOWN")
        with pytest.raises(ValueError):
            pragma.evaluate(self.interfaces, self.env)


class TestPragmaParsing:
    """Test parsing pragmas from strings"""
    
    def test_parse_tie_pragma(self):
        """Test parsing TIE pragmas"""
        pragma = parse_pragma("TIE mat[1] vec")
        assert isinstance(pragma, TiePragma)
        assert pragma.left_expr == "mat[1]"
        assert pragma.right_expr == "vec"
        
        pragma = parse_pragma("TIE a[0] b[0]")
        assert isinstance(pragma, TiePragma)
        assert pragma.left_expr == "a[0]"
        assert pragma.right_expr == "b[0]"
    
    def test_parse_constr_pragma(self):
        """Test parsing CONSTR pragmas"""
        # With symbol
        pragma = parse_pragma("CONSTR vec % BURST")
        assert isinstance(pragma, ConstrPragma)
        assert pragma.expr == "vec"
        assert pragma.op == "%"
        assert pragma.value == "BURST"
        
        # With integer
        pragma = parse_pragma("CONSTR mat[0] >= 16")
        assert isinstance(pragma, ConstrPragma)
        assert pragma.expr == "mat[0]"
        assert pragma.op == ">="
        assert pragma.value == 16
        
        # Complex expression
        pragma = parse_pragma("CONSTR a[0] + b[1] <= MAX_SIZE")
        assert isinstance(pragma, ConstrPragma)
        assert pragma.expr == "a[0] + b[1]"
        assert pragma.op == "<="
        assert pragma.value == "MAX_SIZE"
    
    def test_parse_errors(self):
        """Test parsing error cases"""
        # Empty pragma
        with pytest.raises(ValueError):
            parse_pragma("")
        
        # Invalid pragma type
        with pytest.raises(ValueError):
            parse_pragma("INVALID a b c")
        
        # Wrong number of arguments
        with pytest.raises(ValueError):
            parse_pragma("TIE only_one_arg")
        
        # Missing operator
        with pytest.raises(ValueError):
            parse_pragma("CONSTR expr value")
    
    def test_case_insensitive(self):
        """Test case insensitive pragma parsing"""
        pragma1 = parse_pragma("tie a b")
        pragma2 = parse_pragma("TIE a b")
        pragma3 = parse_pragma("Tie a b")
        
        assert all(isinstance(p, TiePragma) for p in [pragma1, pragma2, pragma3])
        assert all(p.left_expr == "a" for p in [pragma1, pragma2, pragma3])


class TestPragmaIntegration:
    """Integration tests with real kernel scenarios"""
    
    def test_matrix_multiply_constraints(self):
        """Test pragmas for matrix multiply kernel"""
        interfaces = {
            "vec": Interface(
                name="vec",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(1, 512),
                block_dims=(1, 512),
                stream_dims=(1, 16)
            ),
            "mat": Interface(
                name="mat",
                direction=InterfaceDirection.WEIGHT,
                dtype=INT16,
                tensor_dims=(256, 512),
                block_dims=(8, 512),
                stream_dims=(8, 16)
            ),
            "out": Interface(
                name="out",
                direction=InterfaceDirection.OUTPUT,
                dtype=INT32,
                tensor_dims=(1, 256),
                block_dims=(1, 256),
                stream_dims=(1, 8)
            )
        }
        
        env = {"SIMD": 16, "PE": 8, "BURST": 64}
        
        # Matrix columns must match vector size
        pragma1 = TiePragma("mat[1]", "vec[1]")
        assert pragma1.evaluate(interfaces, env) == True
        
        # Output size matches matrix rows
        pragma2 = TiePragma("out[1]", "mat[0]")
        assert pragma2.evaluate(interfaces, env) == True
        
        # Alignment constraints
        pragma3 = ConstrPragma("vec[1]", "%", "BURST")
        assert pragma3.evaluate(interfaces, env) == True  # 512 % 64 == 0
        
        # Parallelism constraints
        pragma4 = ConstrPragma("mat[0]", "%", "PE")
        assert pragma4.evaluate(interfaces, env) == True  # 256 % 8 == 0