#!/bin/bash

# mkdir -p out/final/

# # GIN
# for FILE in configs/final/gin/C_2/*
# do
# echo -e "$FILE"
# # rm -rf models out
# python main.py $FILE
# done

# # GCN
# for FILE in configs/final/graphconv/C_2/N_1000_C_2_p_0.025_q_0.0017_Ktrain_1000_Ktest_100_L_32_fs_rn_opt_sgd_use_W1_false.json
# do
# echo -e "$FILE"
# # rm -rf models out
# python main.py $FILE
# done

# SMPNN
for FILE in configs/final/smpnn/C_2/*
do
echo -e "$FILE"
# rm -rf models out
python main.py $FILE
done

# Graph Transformer
for FILE in configs/final/easygt/C_2/*
do
echo -e "$FILE"
# rm -rf models out
python main.py $FILE
done

# GCN with the same parameters (few layers, tiny hidden size)
# for FILE in configs/final/sample/T1/*
# do
# echo -e "$FILE"
# rm -rf models out
# python main.py $FILE
# done

# for FILE in configs/final/graphconv/C_4/*
# do
# echo -e "$FILE"
# python main.py $FILE
# done


# for FILE in configs/final/graphconv_hetero/C_2/*
# do
# echo -e "$FILE"
# python main.py $FILE
# done

# for FILE in configs/final/graphconv_hetero/C_4/*
# do
# echo -e "$FILE"
# python main.py $FILE
# done
