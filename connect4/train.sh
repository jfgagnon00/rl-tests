x=1
while [ $x -le 1000 ]
do
  echo "Launch train: $x"
  py main.py train
  x=$(( $x + 1 ))
done