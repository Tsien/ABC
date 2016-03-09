#!/bin/bash
#this script will run the main generating program, 
#then create a new directory under ./results
#and put three output files into that directory

sudo python ./fixed_text_generate.py
echo finish training ...

#convert generated ABC notation to midi format file
abc2midi generated.txt > music.mid

#plot figures
./plot.gp

#rename the new directory here
dir=../../results/SGD_50
mkdir $dir

#moving output files
mv generated.txt $dir
mv weights.txt $dir
mv loss.csv $dir
mv *.mid $dir
mv loss.png $dir
mv time.png $dir

