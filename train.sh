#!/bin/bash

HOW_MANY=60000

chmod +x train/input_generator.py
chmod +x stop.sh
rm -f .end

/usr/local/cuda/bin/nvcc *.cu && 
while [ ! -s .end ]
do
  time train/input_generator.py -n $HOW_MANY | ./a.out >> log.stdout 2>> log.stderr
done
echo "'.end' file found!"