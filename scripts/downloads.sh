#!/bin/bash

DATA_DIR=processed
URL=https://nlp.stanford.edu/data/glove.6B.zip
ZIP_FILE=$DATA_DIR/glove.6B.zip
UNZIPPED_FILE=$DATA_DIR/glove.6B.100d.txt

mkdir -p $DATA_DIR

if [ -f $UNZIPPED_FILE ]; then
  echo "Files already unzipped in $DATA_DIR. Skipping download and extraction."
else
  if [ ! -f $ZIP_FILE ]; then
    echo "Downloading $URL..."
    wget -N $URL -O $ZIP_FILE
  else
    echo "Zip file already exists at $ZIP_FILE. Skipping download."
  fi

  echo "Unzipping $ZIP_FILE to $DATA_DIR..."
  unzip -o $ZIP_FILE -d $DATA_DIR

  echo "Removing zip file $ZIP_FILE..."
  rm $ZIP_FILE
fi