#!/bin/bash

mkdir -p out/final/

# for FILE in configs/final/graphtrans/T1/*
# do
# echo -e "$FILE"
# rm -rf models out
# python main.py $FILE
# done

# Comparing GT and GCN with equivalent parameters
for FILE in configs/final/compare_gt_gcn/*/*
do
echo -e "$FILE"
rm -rf models out
python main.py $FILE
done

# # Graph Transformer
# for FILE in configs/final/easygt/C_2/*
# do
# echo -e "$FILE"
# rm -rf models out
# python main.py $FILE
# done

# GIN
# for FILE in configs/final/gin/C_2/*
# do
# echo -e "$FILE"
# rm -rf models out
# python main.py $FILE
# done

# # GIN
# for FILE in configs/final/gin/C_2/*
# do
# echo -e "$FILE"
# rm -rf models out
# python main.py $FILE
# done

# SMPNN
for FILE in configs/final/smpnn/C_2/*
do
echo -e "$FILE"
rm -rf models out
python main.py $FILE
done

# GCN with the same parameters (few layers, tiny hidden size)
# for FILE in configs/final/sample/T1/*
# do
# echo -e "$FILE"
# rm -rf models out
# python main.py $FILE
# done

# for FILE in configs/final/graphconv/C_2/*
# do
# echo -e "$FILE"
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
