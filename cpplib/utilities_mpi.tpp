#include <mpi.h>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <dirent.h>
#include <cstring>
#include <glob.h>
#include <sstream>

//#include "/u/ybahe/ANALYSIS/Protea-H/Config.h"
//#include "/u/ybahe/ANALYSIS/Protea-H/globals.h"

#include "/u/ybahe/cpplib/utilities.h"


template <typename T>
void collect_vector_mpi_fast(std::vector<T> &v,
			     int dest_task,
			     int offset_flag,
			     int clear_flag) {   // 0: leave as is, 1: clear, 2: broadcast
  

  
  // 1.) Establish how much data is to be sent by each task

  int nVectorSize = v.size();
  T nCoda = v.back();

  std::vector<int> vNumFromTasks;
  std::vector<int> vNumBytesFromTasks;
  std::vector<T> vCodaeFromTasks;
  std::vector<int> vTDOlist;

  if (mpirank() == dest_task) {
    vNumFromTasks.resize(numtasks(), 0);
    vNumBytesFromTasks.resize(numtasks(), 0);
    if (offset_flag == 1)
      vCodaeFromTasks.resize(numtasks(),0);
    vTDOlist.resize(numtasks()+1, 0);
  }

  MPI_Gather(&nVectorSize, 1, MPI_INT, &vNumFromTasks.front(), 1, MPI_INT, dest_task, MPI_COMM_WORLD);
  
  if (offset_flag == 1)
    MPI_Gather(&nCoda, sizeof(T), MPI_BYTE, &vCodaeFromTasks.front(), sizeof(T), MPI_BYTE, dest_task, MPI_COMM_WORLD);

  if (mpirank() == dest_task) {
    for (int iitask = 0; iitask < numtasks(); iitask++) {
      if (offset_flag == 1)
	vNumFromTasks.at(iitask) -= 1;
      
      vNumBytesFromTasks.at(iitask) = vNumFromTasks.at(iitask)*sizeof(T);
      vTDOlist.at(iitask+1) = vTDOlist.at(iitask) + vNumFromTasks.at(iitask)*sizeof(T);
    }
  }

  int nSendCount = nVectorSize;
  if (offset_flag == 1)
    nSendCount--;
    
  std::vector<T> vFull;
  
  if (mpirank() == dest_task) {
    long nFullLength = vTDOlist.back()/sizeof(T);
    if (offset_flag == 1)
      nFullLength++;
    vFull.resize(nFullLength);
  }

  
  MPI_Gatherv(&v.front(), nSendCount*sizeof(T), MPI_BYTE, &vFull.front(), &vNumBytesFromTasks.front(), &vTDOlist.front(), MPI_BYTE, dest_task, MPI_COMM_WORLD);
  
  if (mpirank() == dest_task && offset_flag == 1) {

    T ncurroffset = 0;
    int ncurrtask = 0;
    int ncurrfromthistask = 0;
    for (int ii = 0; ii < (vFull.size()-1); ii++) {
      vFull.at(ii) += ncurroffset;
      ncurrfromthistask++;
      
      if (ncurrfromthistask == vNumFromTasks.at(ncurrtask)) {
	ncurroffset += vCodaeFromTasks.at(ncurrtask);
	ncurrtask++;
	ncurrfromthistask = 0;
      }
    }
   
    vFull.back() = ncurroffset;
  }
  
  v.swap(vFull);

#ifdef USEOLD

  // Only need to send if this is not the target task!
  if (mpirank() != dest_task) {
    
    MPI_Request reqs[2];
    MPI_Status stats[2];
    
    
    if (nVectorSize > 0)
      MPI_Isend(&v.front(), v.size()*sizeof(T), MPI_BYTE, dest_task, 77642, MPI_COMM_WORLD, &reqs[1]);
    
    if (nVectorSize > 0)
      MPI_Wait(&reqs[1], &stats[1]);
    
    if (clear_flag == 1)
      v.clear();
    
  } else {
    
    // The more fun part: Current task is the one that everyone
    //                    sends their junk to. We need to 
    //                    (a) receive the data, and
    //                    (b) build the output.

    
    MPI_Request *reqs = new MPI_Request[numtasks()];
    MPI_Status stats[1];

    /*
    for (int ii = 0; ii< numtasks(); ii++) {
      if (ii != mpirank())
      MPI_Irecv(&vNumFromTasks.at(ii), 1, MPI_LONG, ii, 77641, MPI_COMM_WORLD, &reqs[ii]);
    }

    for (int ii = 0; ii < numtasks(); ii++) {
      if (ii != mpirank())
	MPI_Wait(&reqs[ii], &stats[0]);     
    }
    
    vNumFromTasks.at(mpirank()) = v.size();
    */

    // Find out how long full output vector will be:
    long nLengthOfFullVector = 0;

    for (int ii = 0; ii < numtasks(); ii++) {
      nLengthOfFullVector += vNumFromTasks.at(ii); }
    
    if (offset_flag == 1)
      nLengthOfFullVector -= (numtasks()-1); // Remove internal codae

    std::vector<T> vTempOwnVecPart = v;
    v.resize(nLengthOfFullVector, 0);

    long nCurrOffsetInFullVector = 0;
    
    // Now receive full vectors:

    for (int ii = 0; ii < numtasks(); ii++) {
      std::vector<T> vTempVecPart; 
     
      if (ii == mpirank()) {
	vTempVecPart = vTempOwnVecPart;
	vTempOwnVecPart.clear();

      } else {
	if (vNumFromTasks.at(ii) > 0) {
	  
	  vTempVecPart.resize(vNumFromTasks.at(ii));
	  MPI_Request rreqs[1];
	  MPI_Status rstats[1];
	  
	  MPI_Irecv(&vTempVecPart.front(), vTempVecPart.size()*sizeof(T), MPI_BYTE, ii, 77642, MPI_COMM_WORLD, &rreqs[0]);

	  MPI_Wait(&rreqs[0], &rstats[0]);
	  
	}
      }

      // We now have the vTempVecPart vector, which has size 0 if nothing was sent from task ii
      
      if (offset_flag == 0) {
	for(long kk = 0; kk < vNumFromTasks.at(ii); kk++)
	  v.at(kk+nCurrOffsetInFullVector) = vTempVecPart.at(kk);
	nCurrOffsetInFullVector += vNumFromTasks.at(ii);
	
      } else {

	int nCurrOffsetOfFullVector = v.at(nCurrOffsetInFullVector);  // zero for first segment! 
	
	for(long kk = 0; kk < vNumFromTasks.at(ii); kk++)
	  v.at(kk+nCurrOffsetInFullVector) = vTempVecPart.at(kk) + nCurrOffsetOfFullVector;
	nCurrOffsetInFullVector += vNumFromTasks.at(ii)-1; 
      }
      
      if (clear_flag == 1)
	vTempVecPart.clear();
      
    } // ends loop through tasks
  } // ends 'master'-only section

  // New bit added 22 Nov 15: Broadcast resulting vector to all tasks if required

#endif 

  if (clear_flag == 2) {

    long nFullLength;
    if (mpirank() == dest_task)
      nFullLength = v.size();

    MPI_Bcast(&nFullLength, 1, MPI_LONG, dest_task, MPI_COMM_WORLD);
    
    v.resize(nFullLength);
    MPI_Bcast(&v.front(), nFullLength * sizeof(T), MPI_BYTE, dest_task, MPI_COMM_WORLD); 

  } // ends broadcast section

  return;
}


template <typename T>
void collect_vector_mpi(std::vector<T> &v,
			int dest_task,
			int offset_flag,
			int clear_flag,    // 0: leave as is, 1: clear, 2: broadcast
			int nFlagBarrier) { // 0 = no barrier, 1 [default] = barrier at end.

  int rc;
  
  
  // 1.) Establish how much data is to be sent by each task

  
  long nVectorSize = v.size();
  std::vector<long> vNumFromTasks;
  
  if (mpirank() == dest_task) {
    vNumFromTasks.resize(numtasks(), 0);
  }

  MPI_Gather(&nVectorSize, 1, MPI_LONG, &vNumFromTasks.front(), 1, MPI_LONG, dest_task, MPI_COMM_WORLD);
  
  
  // Only need to send if this is not the target task!
  if (mpirank() != dest_task) {
    
    MPI_Request reqs[1];
    MPI_Status stats[1];
    
    // Try to protect from stupid things if v.size() is too large
    if (nVectorSize*sizeof(T) > 2000000000) {
      std::cout << "Trying to send very large vector - this might violate INT32 limits..." << std::endl;
      exit(880);
    }

    if (nVectorSize > 0) {
      MPI_Isend(&v.front(), v.size()*sizeof(T), MPI_BYTE, dest_task, 77642, MPI_COMM_WORLD, &reqs[0]);
      MPI_Wait(&reqs[0], &stats[0]);
    }
    
    if (clear_flag == 1)
      v.clear();
    
  } else {
    
    // The more fun part: Current task is the one that everyone
    //                    sends their junk to. We need to 
    //                    (a) receive the data, and
    //                    (b) build the output.


    // Find out how long full output vector will be:
    long nLengthOfFullVector = 0;

    for (int ii = 0; ii < numtasks(); ii++) {
      nLengthOfFullVector += vNumFromTasks.at(ii); }
    
    if (offset_flag == 1)
      nLengthOfFullVector -= (numtasks()-1); // Remove internal codae

    std::vector<T> vTempOwnVecPart = v;
    v.resize(nLengthOfFullVector, 0);

    long nCurrOffsetInFullVector = 0;
    
    // Now receive full vectors:

    for (int ii = 0; ii < numtasks(); ii++) {
      std::vector<T> vTempVecPart; 
     
      if (ii == mpirank()) {
	vTempVecPart = vTempOwnVecPart;
	vTempOwnVecPart.clear();

      } else {
	if (vNumFromTasks.at(ii) > 0) {
	  
	  vTempVecPart.resize(vNumFromTasks.at(ii));
	  MPI_Request rreqs[1];
	  MPI_Status rstats[1];
	  
	  MPI_Irecv(&vTempVecPart.front(), vTempVecPart.size()*sizeof(T), MPI_BYTE, ii, 77642, MPI_COMM_WORLD, &rreqs[0]);

	  MPI_Wait(&rreqs[0], &rstats[0]);
	  
	}
      }

      // We now have the vTempVecPart vector, which has size 0 if nothing was sent from task ii
      
      if (offset_flag == 0) {
	for(long kk = 0; kk < vNumFromTasks.at(ii); kk++)
	  v.at(kk+nCurrOffsetInFullVector) = vTempVecPart.at(kk);
	nCurrOffsetInFullVector += vNumFromTasks.at(ii);
	
      } else {

	T nCurrOffsetOfFullVector = v.at(nCurrOffsetInFullVector);  // zero for first segment! 
	
	for(long kk = 0; kk < vNumFromTasks.at(ii); kk++)
	  v.at(kk+nCurrOffsetInFullVector) = vTempVecPart.at(kk) + nCurrOffsetOfFullVector;
	nCurrOffsetInFullVector += vNumFromTasks.at(ii)-1; 
      }
      
      if (clear_flag == 1)
	vTempVecPart.clear();
      
    } // ends loop through tasks
  } // ends 'master'-only section

  // New bit added 22 Nov 15: Broadcast resulting vector to all tasks if required

  if (clear_flag == 2) {

    long nFullLength;
    if (mpirank() == dest_task)
      nFullLength = v.size();

    MPI_Bcast(&nFullLength, 1, MPI_LONG, dest_task, MPI_COMM_WORLD);
    
    v.resize(nFullLength);
    MPI_Bcast(&v.front(), nFullLength * sizeof(T), MPI_BYTE, dest_task, MPI_COMM_WORLD); 

  } // ends broadcast section

  if (nFlagBarrier == 1)
    MPI_Barrier(MPI_COMM_WORLD);

  return;
}





template <typename T>
int check_for_consistency(T nVar,
			  int nMode,
			  std::string sDescr) {

  
  std::vector<T> vTempCheck;   		   
  if (mpirank() == 0)
     vTempCheck.resize(numtasks());  
  
  MPI_Gather(&nVar, sizeof(T), MPI_BYTE, &vTempCheck.front(), sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);

  int nCheckRes = 0;
  if (mpirank() == 0) {
    for (int ii = 0; ii < numtasks(); ii++) 
      if (vTempCheck.at(ii) != nVar) 
	if (nMode == 1) {
	  
	  nCheckRes = ii;
	  break;
	  
	} else {
	  
	  std::cout << "Value of " + sDescr + " in task " << ii << " [" << vTempCheck.at(ii) << "] differs from master value [" << nVar << "]. Strict checking was enforced. Investigate." << std::endl;
	  exit(42);
	
	}
  }

  MPI_Bcast(&nCheckRes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return nCheckRes;
}

  
// report on distribution of variables across tasks:
template <typename T>
void report(T var, std::string varname) {

  using namespace std;

  double doublevar = static_cast<double>(var);

  vector<double> vAllVars;

  if (mpirank() == 0)
    vAllVars.resize(numtasks());

  MPI_Gather(&doublevar, 1, MPI_DOUBLE, &vAllVars.front(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (mpirank() == 0) {

   double minvar = min<double>(vAllVars);
   double maxvar = max<double>(vAllVars);

   double varSum = 0;
   for (int ii = 0; ii<numtasks(); ii++)
     varSum += vAllVars.at(ii);

     double avvar = varSum/numtasks();

     std::cout << rp() + varname << " MIN = " << minvar << ", MAX = " << maxvar << ", AVERAGE = " << avvar << endl;

  }
  
  return;     	      
}

template <typename T>
void broadcast_struct(T &StructToBC, int nRoot) {

  MPI_Bcast(&StructToBC, sizeof(T), MPI_BYTE, nRoot, MPI_COMM_WORLD);

  return;
}


template <typename T>
void broadcast_vector(std::vector<T> &vVec, int orig_task) {


  long nSize = vVec.size();
  
  if (nSize * sizeof(T) > 2000000000) {
    std::cout << "Vector is too large to be reliably sent through broadcast_vector (n=" << nSize << ", sizeof(T)=" << sizeof(T) << std::endl;
    exit(9998);
  }

  MPI_Bcast(&nSize, 1, MPI_LONG, orig_task, MPI_COMM_WORLD);

  vVec.resize(nSize);
  MPI_Bcast(&vVec.front(), nSize*sizeof(T), MPI_BYTE, orig_task, MPI_COMM_WORLD);



  return;
}


template <typename T>
std::vector<T> find_global_minmax(std::vector<T> vTT, int nVerb) {
  
  if (nVerb == 1)
    function_switch("find_global_minmax");
  
  double dST = GetTime();

  T local_min, local_max;
  int nNonZero = 0;

  if (vTT.size() > 0) {
    // Find *local* minmax of vector:
    
    nNonZero = 1;
    std::vector<T> local_minmax = minmax<T>(vTT);
    
    if (nVerb == 1 && mpirank() == 0)
      std::cout << rp() + "Finding local minmax took " << GetTime()-dST << " sec." << std::endl;
    
    
    // Compare local extrema across tasks:
    dST = GetTime();
    
    local_min = local_minmax.front();
    local_max = local_minmax.back();
    
  }
  
  // NB: We do not use MPI_Allreduce here, to keep it general for all types... 
  //     There might be a small performance penalty, but this should be 
  //     irrelevant in practice.
  
  std::vector<T> vAllMin, vAllMax; /* will be expanded at root only */
  std::vector<int> vNonZero;

  if (mpirank() == 0) {
    vAllMin.resize(numtasks());
    vAllMax.resize(numtasks());
    vNonZero.resize(numtasks());
  }

  MPI_Gather(&local_min, sizeof(T), MPI_BYTE, 
	     &vAllMin.front(), sizeof(T), MPI_BYTE, 
	     0, MPI_COMM_WORLD);

  MPI_Gather(&local_max, sizeof(T), MPI_BYTE, 
	     &vAllMax.front(), sizeof(T), MPI_BYTE, 
	     0, MPI_COMM_WORLD);

  MPI_Gather(&nNonZero, 1, MPI_INT, 
	     &vNonZero.front(), 1, MPI_INT, 
	     0, MPI_COMM_WORLD);

  std::vector<T> vGlobMinMax(2);  
  if (mpirank() == 0) {
    vGlobMinMax.front() = min<T> (vAllMin, vNonZero);
    vGlobMinMax.back() = max<T> (vAllMax, vNonZero);
  }
  
  vAllMin.clear();
  vAllMax.clear();

  broadcast_vector<T> (vGlobMinMax, 0);
  
  if (nVerb == 1 && mpirank() == 0)
    std::cout << rp() + "Communication across MPI tasks took " << GetTime()-dST << " sec." << std::endl;

  if (nVerb == 1)
    function_switch("find_global_minmax");

  return vGlobMinMax;
}

// This is an MPI function to find the minimum or maximum (or both) of every
// element in a vector. Comparison between input vectors is performed in a 
// hierarchical way that is most efficient if the MPI-task number is a power of two.
// However, the routine will also work with arbitrary task numbers.


template <typename T>
void find_minmax_vector (std::vector<T> &vIn,      // [I/0] Vector to process
			 std::vector<T> &vExtreme, // [O] result (dummy if cFlagInPlace == 1)
			 char cFlagMinMax,         // 0: Min, 1: Max 
			 char cFlagInPlace,    // If == 1, input overwritten with result 
			 char cFlagBroadcast,  // If == 1, result broadcast to all tasks
			 char cFlagThreshold,  // If == 1, values </> Threshold ignored
			 T Threshold,           // Threshold value (ignored unless ^ ==1) 
			 char cVerb) 

{

  if (cVerb == 1)
    function_switch("find_minmax_vector");
  
  // The Plan:
  //
  // 1.) Find exchange level (log_2 numtasks)
  // 2.) Iterate through them: 
  // -- a) Calculate rank % 2^i
  // -- b) If ==0, receive vector from task += 2^(i-1)
  //       If !=0, send vector to task -= 2^(i-1)
  // -- c) If ==0, build received vector into (copy of) own
  
  // Make a copy if we are not combining vectors in-place
  std::vector<T> vInOrig;
  if (cFlagInPlace == 0)
    vInOrig = vIn;
    

  // Make sure the vector has the same length on all tasks
  long nVLength = vIn.size();
  check_for_consistency(nVLength);

  int nExchangeLevel = ilog2(numtasks());
  
  if (cVerb == 1 && mpirank() == 0)
    std::cout << rp() + "Using exchange level " << nExchangeLevel << " (up to " << ipow(2, nExchangeLevel) << " tasks)" << std::endl;
  

  for (int iixl = 1; iixl <= nExchangeLevel; iixl++) {

    if (cVerb == 1 && mpirank() == 0)
      std::cout << rp() + "Executing exchange iteration " << iixl << std::endl;

    MPI_Request reqs[1];
    MPI_Status stats[1];
    
    
    // Test whether we are receiving or sending a vector
    if (mpirank() % ipow(2, iixl) != 0) {
      // Sending vector back 2^(i-1)
      MPI_Isend(&vIn.front(), nVLength*sizeof(T), MPI_BYTE, mpirank()-ipow(2, iixl-1), 789, MPI_COMM_WORLD, &reqs[0]);
      MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
      break; // can stop iterating
    } else {

      // Receiving vector from 2^(i-1) ahead, if this task exists
      if ((mpirank()+ipow(2, iixl-1)) < numtasks()) {

	std::vector<T> vOther(nVLength);
	MPI_Irecv(&vOther.front(), nVLength*sizeof(T), MPI_BYTE, mpirank()+ipow(2, iixl-1),789, MPI_COMM_WORLD, &reqs[0]);
	MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);

	// Combine vectors
	
	if (cFlagMinMax == 0 && cFlagThreshold == 0) { // Minimum, no thresh
	  for (long ii = 0; ii < nVLength; ii++) {
	    if (vOther.at(ii) < vIn.at(ii))
	      vIn.at(ii) = vOther.at(ii);
	  }
	}
	else if (cFlagMinMax == 0 && cFlagThreshold == 1) { // Min, with thresh
	  for (long ii = 0; ii < nVLength; ii++) {
	    if (vIn.at(ii) < Threshold && vOther.at(ii) >= Threshold)
	      vIn.at(ii) = vOther.at(ii);
	    else if (vOther.at(ii) >= Threshold && vOther.at(ii) < vIn.at(ii))
	      vIn.at(ii) = vOther.at(ii);
	  }
	}
	else if (cFlagMinMax == 1 && cFlagThreshold == 0) { // Maximum, no thresh
	  for (long ii = 0; ii < nVLength; ii++) {
	    if (vOther.at(ii) > vIn.at(ii))
	      vIn.at(ii) = vOther.at(ii);
	  }
	}
	else if (cFlagMinMax == 1 && cFlagThreshold == 1) { // Max, with thresh
	  for (long ii = 0; ii < nVLength; ii++) {
	    if (vIn.at(ii) > Threshold && vOther.at(ii) <= Threshold)
	      vIn.at(ii) = vOther.at(ii);
	    else if (vOther.at(ii) <= Threshold && vOther.at(ii) > vIn.at(ii))
	      vIn.at(ii) = vOther.at(ii);
	  }
	}
      } // ends section only if this task has to do something this round 
    } // ends section if this task is receiving
  } // ends loop through exchange levels

  // Task 0 now has the final result in vIn.
  // Do final things:

  if (cFlagBroadcast == 1)
    broadcast_vector<T> (vIn, 0);

  if (cFlagInPlace == 0) {
    vExtreme.swap(vIn); // extreme now result (only on 0 unless broadcast)
    vIn.swap(vInOrig); // vIn now back to original
    vInOrig.clear();
  }

  if (cVerb == 1)
    function_switch("find_minmax_vector");

  return;
}
