#!/usr/bin/gnuplot

reset

set terminal png

# set your chart name here:
set output "loss.png"

set style data lines
set key right


###### Fields in the data file are
###### Iters TrainLoss Seconds TestLoss Seconds

set title "Loss vs. training iterations"
set xlabel "Training iterations"
set ylabel "Loss"

# set input file here:
plot "./loss.csv" using 1:2 title 'Training Loss', \
     "./loss.csv" using 1:4 title 'Testing Loss'


reset

set terminal png

# set your chart name here:
set output "time.png"

set style data lines
set key right


###### Fields in the data file are
###### Iters Seconds TestAccuracy TestLoss

set title "Training Loss vs. Time"
set xlabel "Time"
set ylabel "Training loss"

# set input file here:
plot "./loss.csv" using 3:2 title 'Training Loss', \
     "./loss.csv" using 5:4 title 'Testing Loss'
