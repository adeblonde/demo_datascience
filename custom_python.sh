#!/bin/bash

### little script to convert spaces to tabs in python file
for python_file in $(find . -name '*.py')
do
	echo $python_file
	sed -i 's#    #	#g' $python_file
done