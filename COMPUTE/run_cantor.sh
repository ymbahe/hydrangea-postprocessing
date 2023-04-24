#! bin/bash

OUTFILE="cantor.CE-0T.0.log"

exec 1>&3 2>&4
exec > >(tee -i $OUTFILE)
exec 2>&1

nice /u/ybahe/anaconda3/bin/python3 cantor_base_jul19.py cantor.yml 0 0 29



