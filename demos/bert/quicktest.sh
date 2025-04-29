python gen_initial_folding.py --simd 12 --pe 8 --num_layers 1 -t 1 -o ./configs/l1_simd12_pe8.json
python end2end_bert.py -o quicktest -n 12 -l 1 -z 384 -i 1536 -x True -p ./configs/l1_simd12_pe8.json -d False
