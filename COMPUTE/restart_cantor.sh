#! bin/bash

OUTFILE="cantor.CE-29.RT.log"

exec 1>&3 2>&4
exec > >(tee -i $OUTFILE)
exec 2>&1

/u/ybahe/anaconda3/bin/python3 cantor_base.py cantor_restart.yml 29

