############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
# A simple python script for generating some initial folding configs for DSE based on some specific rules.
import argparse
import json

def mvau(simd:int, pe:int, runtime_writeable:int)->dict:
    d = {}
    d["PE"] = pe
    d["SIMD"] = simd
    d["ram_style"] = "auto"
    d["resType"] = "auto"
    d["mem_mode"] = "internal_decoupled"
    d["runtime_writeable_weights"] = runtime_writeable
    return d

def dupstreams(pe:int)->dict:
    d={}
    d["PE"] = pe
    return d

def shuffle(simd:int)->dict:
    d={}
    d["SIMD"] = simd
    return d

def thresholding(pe:int, runtime_writeable:int)->dict:
    d = {}
    d["PE"] = pe
    d["runtime_writeable_weights"] = runtime_writeable
    d["depth_trigger_uram"] = 0
    d["depth_trigger_bram"] = 0
    return d

def dynmvu(pe:int, simd:int)->dict:
    d = {}
    d["PE"] = pe
    d["SIMD"] = simd
    d["ram_style"] = "auto"
    d["resType"] = "auto"
    return d


def eltwiseadd(pe:int)->dict:
    d = {}
    d["PE"] = pe
    d["ram_style"] = "auto"
    return d

def eltwisemul(pe:int)->dict:
    d = {}
    d["PE"] = pe
    d["ram_style"] = "auto"
    return d

def softmax(simd:int)->dict:
    d = {}
    d['SIMD'] = simd
    return d

def layernorm(simd:int)->dict:
    d = {}
    d['SIMD'] = simd
    return d

def main(args):
    c = {}

    c["Defaults"] = {}
    for n in range(args.num_layers):

        # Generate all MVAUs
        for m in range(0, 8):
            if m == 7 or m == 8:
                d = mvau(2 * args.simd, 2 * args.pe, args.runtime_writeable_weights)
            # dyn mvau
            elif m == 3 or m == 4:
                if args.simd % 3 == 0:
                    d = dynmvu(args.pe, int(args.simd/3))
                elif args.simd % 4 == 0:
                    d = dynmvu(args.pe, int(args.simd/4))
                else:
                    d = dynmvu(args.pe, args.simd)
            else:
                d = mvau(args.simd, args.pe, args.runtime_writeable_weights)
            c[f"MVAU_rtl_{m + (8 * n)}"] = d

        # Duplicate streams
        for m in range(0, 3):
            d = dupstreams(args.other)
            c[f"DuplicateStreams_hls_{m + (3 * n)}"] = d

        # Shuffles
        for m in range(0, 4):
            d = shuffle(args.other)
            c[f"Shuffle_hls_{m + (4 * n)}"] = d

        # Thresholding
        for m in range(0, 9):
            d = thresholding(args.other, 0)
            c[f"Thresholding_rtl_{m + (9 * n)}"] = d

        # EltwiseAdds
        for m in range(0, 2):
            d = eltwiseadd(args.other)
            c[f"ElementwiseAdd_hls_{m + (2 * n)}"] = d

        # EltwiseMul
        for m in range(0, 5):
            d = eltwisemul(args.other)
            c[f"ElementwiseMul_hls_{m + (5 * n)}"] = d

        # SoftMax
        for m in range(0, 1):
            d = softmax(args.other)
            c[f"HWSoftmax_hls_{m + (n * 1)}"] = d

        for m in range(0, 2):
            d = layernorm(args.other)
            c[f"LayerNorm_hls_{m + (n * 2)}"] = d

    with open(args.output, "w") as fp:
        json.dump(c, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBert folding config gen')
    parser.add_argument('-o', '--output', help='Output JSON config', default='config.json')
    parser.add_argument('-s', '--simd', type=int, help='Sets the common SIMD setting for the MVAU', default=48)
    parser.add_argument('-p', '--pe', type=int, help='Sets the common SIMD setting for the MVAU', default=32)
    parser.add_argument('-t', '--other', type=int, help='Sets the SIMD/PE for the other operators between the MVAUs', default=4)
    parser.add_argument('-n', '--num_layers', type=int, help='Sets the number of hidden layers', default=3)
    parser.add_argument('-w', '--runtime_writeable_weights', type=int, help='if 1 Make the weights runtime writeable for the MVAUs', default=0)
    parser.add_argument('-f', '--shuffleb', type=bool, help='Is shuffleB parallelisable yet?', default=False)

    args = parser.parse_args()
    main(args)