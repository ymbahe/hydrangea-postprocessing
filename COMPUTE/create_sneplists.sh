#! /bin/bash
#
# Shell script to generate the various snepshot lists
# Adapted 15 Feb 2018

# =================
# SET UP PARAMETERS
# =================

BASEDIR=$1   
HALO_START=$2
HALO_END=$3
TYPE=$4

source ~/.bashrc


# ==================
# Basic setup
# ==================


cd ${BASEDIR}
BASEDIR=`pwd`
echo "Base dir is [${BASEDIR}]"

echo -n "Python command is: "
echo `type python`


# ====
# Loop
# ====

for i in $(seq $2 $3)
do
    
    if [ ! -d HaloF$i ]; then
	continue
    fi

    cd HaloF$i
    cd ${TYPE}
    
    echo -n "PROCESSING CLUSTER: "
    pwd

   
    if [ ! -d sneplists ]; then
	mkdir sneplists
    fi
    
    cd sneplists

    if [ ${TYPE} == "HYDRO" ]; then
	if [ -f snipshot_times.dat ]; then
	    echo "snipshot_times.dat exists..." 
	else
	    echo "snipshot_times.dat not found..."
	    #python ~/ANALYSIS/list_snipshot_times.py ${BASEDIR}/HaloF${i}/${TYPE}
	fi    
    fi
    
    # Create root list
    if [ ${TYPE} == "DM" ]; then
	LISTS="allsnaps"
    else

	if [ ${i} -eq 11 ]; then
	    LISTS="default_long basic allsnaps full_movie short_movie"
	else
	    #LISTS="default_long basic allsnaps"
	    #LISTS="basic allsnaps"
	    LISTS="allsnaps"
	fi
    fi

    echo ""
    echo "Creating root list, containing '${LISTS}'"
    echo ""

    python ~/ANALYSIS/make_tailored_sneplist.py ${BASEDIR}/HaloF${i}/${TYPE} -l "${LISTS}" -r

    
    # Create individual lists

    echo ""
    echo "Create 'regsnaps' list..."
    echo ""
    python ~/ANALYSIS/make_tailored_sneplist.py ${BASEDIR}/HaloF${i}/${TYPE} -l "regsnaps" 	

    echo ""
    echo "Create 'allsnaps' list..."
    echo ""
    python ~/ANALYSIS/make_tailored_sneplist.py ${BASEDIR}/HaloF${i}/${TYPE} -l "allsnaps"
    
    echo ""
    echo "Create 'z0_only' list..."
    echo ""
    python ~/ANALYSIS/make_tailored_sneplist.py ${BASEDIR}/HaloF${i}/${TYPE} -l "z0_only" 	

    if [ ${TYPE} == "HYDRO" ]; then

	#echo ""
	#echo "Create 'basic' list..."
	#echo ""
	#python ~/ANALYSIS/make_tailored_sneplist.py ${BASEDIR}/HaloF${i}/${TYPE} -l "basic" 	
	
	#echo ""
	#echo "Create 'default_long' list..."
	#echo ""
	#python ~/ANALYSIS/make_tailored_sneplist.py ${BASEDIR}/HaloF${i}/${TYPE} -l "default_long" 	
	
	if [ ${i} -eq 11 ]; then
	    echo ""
	    echo "Create 'short_movie' list..."
	    echo ""
	    python ~/ANALYSIS/make_tailored_sneplist.py ${BASEDIR}/HaloF${i}/${TYPE} -l "short_movie" 	

	    if [ ${i} -eq 11 ]; then
		echo ""
		echo "Create 'full_movie' list..."
		echo ""
		python ~/ANALYSIS/make_tailored_sneplist.py ${BASEDIR}/HaloF${i}/${TYPE} -l "full_movie" 	

	    fi
	fi
	
    fi
    
    cd ${BASEDIR}

done
