#! bin/bash

OUTFILE="CS-test.log"

exec 1>&3 2>&4
exec > >(tee -i $OUTFILE)
exec 2>&1

/u/ybahe/anaconda3/bin/python3 cantor_soprano_test.py soprano_test.yml





