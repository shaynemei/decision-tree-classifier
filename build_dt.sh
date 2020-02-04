#!/bin/sh

if [ -z "$1" ]; then
	echo "Usage: build dt.sh \$training_data \$test_data \$max_depth \$min_gain \$model_file \$sys_output > \$acc_file"
else
	/opt/python-3.6/bin/python3 build_dt.py $1 $2 $3 $4 $5 $6
	#python3 build_dt.py $1 $2 $3 $4 $5 $6
fi