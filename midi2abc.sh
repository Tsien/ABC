#!/bin/bash
#this script will convert all midi files to one abc file

#rename the path to .midi files
path=./test
FILES=$path/*.mid

#convert all midi files
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  #cat $f
  midi2abc -f $f > "$f.abc"
done

#combine all abc files into one file
echo "combining files..."
FILES=$path/*.abc
cat $FILES >> all.txt
mv all.txt $path
