x=1
while [ $x -le 1000 ]
do
  echo "Launch train: $x"
  py main_openai.py
  x=$(( $x + 1 ))
done