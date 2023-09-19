#! /bin/bash

set -e
wget https://fpcp.s3.us-east-2.amazonaws.com/ckpts.tar.gz 
tar -xvf ckpts.tar.gz
rm ckpts.tar.gz
