#! /bin/bash

set -e
wget https://fpcp.s3.us-east-2.amazonaws.com/data.tar.gz
tar -xvf data.tar.gz
rm data.tar.gz
