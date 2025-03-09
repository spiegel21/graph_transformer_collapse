# Graph Transformer
for FILE in configs/final/easygt/C_2/*
do
echo -e "$FILE"
rm -rf models out
python main.py $FILE
done