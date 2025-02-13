# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse


def main(args):

    runjob(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBERT FINN demo script')
    parser.add_argument('-m', '--model', help='Output ONNX model name', required=True)
    parser.add_argument('-o', '--output', type=str, default='./builds/', help='Output build path', required=True)
    parser.add_argument('-b', '--bitwidth', type=int, default=8, help='The quantisation bitwidth (either 4 or 8)')
    parser.add_argument('-f', '--fps', type=int, default=3000, help='The target fps for auto folding')
    parser.add_argument('-c', '--clk', type=float, default=3.33, help='The target clock rate for the hardware')
    parser.add_argument('-s', '--stop_step', type=str, default=None, help='Step to stop at in the build flow')
    parser.add_argument('-p', '--param', type=str, default=None, help='Use a preconfigured file for the folding parameters')
    parser.add_argument('-x', '--fifodepth', type=bool, default=True, help='Skip the FIFO depth stage')
    parser.add_argument('-d', '--dcp', type=bool, default=True, help='Generate a DCP')
    # Model specific
    parser.add_argument('-z', '--hidden_size', type=int, default=384, help='Sets BERT hidden_size parameter')
    parser.add_argument('-n', '--num_attention_heads', type=int, default=12, help='Sets BERT num_attention_heads parameter')
    parser.add_argument('-l', '--num_hidden_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-i', '--intermediate_size', type=int, default=1536, help='Sets BERT intermediate_size parameter')
    parser.add_argument('-q', '--seqlen', type=int, default=128, help='Sets the sequence length parameter')

    args = parser.parse_args()
    main(args)
