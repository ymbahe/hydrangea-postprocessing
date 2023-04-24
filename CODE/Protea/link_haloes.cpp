/** Functions to analyse the matched IDs and produce the tracing result **/

#define VERBOSE

#include "Config.h"

#include <iostream>
#include <mpi.h>
#include <H5Cpp.h>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <stdexcept>

#include <cstdlib>

#include "utilities.h"



void remove_halo_duplicates(std::vector<Result> &FullResult, 
			    std::vector<long> &vHaloList);

void select_tracinghaloes(const std::vector<long> &vTaskHaloLocList, 
			  const std::vector<long> &vTHOlistA, 

			  std::vector<long> &vTraceListInternal, 
			  std::vector<long> &vTraceLocList) {

  using namespace std;
  
  int rank = 0;
  int rc = 0;
  int numtasks = 0;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  string rp = "[" + to_string(static_cast<long long>(rank)) + "]: ";

  double dDummy = ElTime();
  
  if (rank == 0)
    cout << rp + "Selecting target haloes on each core..." << endl;

 
  // --------------------------------------------------------------
 

  long nHaloesTask = vTHOlistA.at(rank+1)-vTHOlistA.at(rank);

  long nTraceHaloes = 0;
  for (long ii=0; ii<nHaloesTask; ii++) {
    if (vTaskHaloLocList.at(ii) >= 0)
      nTraceHaloes++;
  }


  vTraceListInternal.resize(nTraceHaloes);
  vTraceLocList.resize(nTraceHaloes);

  long j = 0;

  for (long ii=0; ii<nHaloesTask; ii++) {
    if (vTaskHaloLocList.at(ii) >= 0) {
      vTraceLocList.at(j) = vTaskHaloLocList.at(ii);
      vTraceListInternal.at(j) = ii;
      j++;
    }
  }

  if (rank == 0)
    cout << rp + "   ...finished (" << ElTime() << " sec.)" << endl;

  return;

}


void build_taskinfo(const std::vector<long long> &vFullPartOffsetA, 
		    const std::vector<long long> &vFullPartLengthA, 
		    const std::vector<long> &vTHOlistA, 
		    const std::vector<long long> &vTPOlistA,
		    long nCurrFirstHalo, 
		    const std::vector<long> &vTraceListInternal, 

		    std::vector<long long> &vTraceOffset, 
		    std::vector<long long> &vTraceLength) {

  using namespace std;
  
  int rank = 0;
  int rc = 0;
  int numtasks = 0;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  string rp = "[" + to_string(static_cast<long long>(rank)) + "]: ";

  double dDummy = ElTime();

  if (rank == 0)
    cout << rp + "Building info list for each task..." << endl;

 
  // --------------------------------------------------------------
  
  long nTraceHaloes = vTraceListInternal.size();
  long FirstHaloInFullOffset = vTHOlistA.at(rank)-nCurrFirstHalo;
  
  cout << rp + "nTraceHaloes = " << nTraceHaloes << endl;
  cout << rp + "Size of vTraceOffset = " << vTraceOffset.size() << endl;
  cout << rp + "Size of vFullPartOffsetA = " << vFullPartOffsetA.size() << endl;
  
  for (long ii=0; ii<nTraceHaloes; ii++) {

    long currHalo;
    try {
    currHalo = FirstHaloInFullOffset+vTraceListInternal.at(ii);
    //cout << rp + "Test: " << ii << " -> " << currHalo << endl;
    //cout << rp + "Test: vTPOlistA.at(rank) = " << vTPOlistA.at(rank) << endl;
    //cout << rp + "Test: vFullPartOffsetA.at(currHalo) = " << vFullPartOffsetA.at(currHalo) << endl;
    //cout << rp + "Test: vFullPartLengthA.at(currHalo) = " << vFullPartLengthA.at(currHalo) << endl;
    
    vTraceOffset.at(ii) = vFullPartOffsetA.at(currHalo)-vTPOlistA.at(rank);
    vTraceLength.at(ii) = vFullPartLengthA.at(currHalo);
    }

    catch (const std::out_of_range &oor) {
      std::cerr << "Out of Range error: " << oor.what() << endl;
      cerr << "ii = " << ii << endl;
      cerr << "currHalo = " << currHalo << endl;
      cerr << "rank = " << rank << endl;
      cerr << "vFullPartOffsetA.size() = " << vFullPartOffsetA.size() << endl;
      cerr << "vFullPartLengthA.size() = " << vFullPartLengthA.size() << endl;
      cerr << "vTraceOffset.size() = " << vTraceOffset.size() << endl;
      cerr << "vTraceLength.size() = " << vTraceLength.size() << endl;

      exit(420);
      
    }
      

  }
  
  cout << rp + "   ...finished (" << ElTime() << " sec.)" << endl; 

  return;

}


void link_haloes(const std::vector<long long> &vTraceOffset, 
		 const std::vector<long long> &vTraceLength, 
		 const std::vector<long> &vTaskMatchHaloes, 
		
		 std::vector<Result> &vResult) {
 

  using namespace std;
  
  int rank = 0;
  int rc = 0;
  int numtasks = 0;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  string rp = "[" + to_string(static_cast<long long>(rank)) + "]: ";

  double dDummy = ElTime();

  if (rank == 0)
    cout << rp + "Now linking haloes between snapshots..." << endl;
  
 
  // --------------------------------------------------------------

  // Loop through individual haloes on current task:

  double dTime_One = 0;
  double dTime_Two = 0;
  double dTime_Three = 0;

  // Set up FULL histogram here:
  
  double dStartMakeHisto = GetTime();
  int success;
  vector<long> vMinMax = minmax<long> (vTaskMatchHaloes, 0, success);
  long MinMatchHalo = vMinMax.at(0);
  vector<long> vHistogram(vMinMax.at(1)-vMinMax.at(0)+1,0);

  //if (success == 0) {
  //  cout << rp + "No matches at all for this task??" << endl;
  //  cout << rp + "vMinMax = " << vMinMax.at(0) << ", " << vMinMax.at(1) << endl;
  //  exit(47);
  //}

  report<double> (GetTime()-dStartMakeHisto,  "Histogram setup [sec.]: ");
  report<long> (MinMatchHalo,                 "Histogram minimum:      ");
  report<long> (vMinMax.at(1)-MinMatchHalo+1, "Histogram length:       ");

  for (long ii=0; ii<vTraceLength.size(); ii++) {
  
    double dStartTime = GetTime();
  
    long minval;
    long maxval;
    int success = 0;
    
    for (long long jj=vTraceOffset.at(ii);jj<vTraceOffset.at(ii)+vTraceLength.at(ii);jj++) {
      long CurrVal = vTaskMatchHaloes.at(jj);
      if (CurrVal >= 0) {

	vHistogram.at(CurrVal-MinMatchHalo) += 1;
	
	if (success == 0) {
	  success=1;
	  minval=CurrVal;
	  maxval=CurrVal;
	}
	
	if (CurrVal < minval)
	  minval=CurrVal;
	if (CurrVal > maxval)
	  maxval=CurrVal;
      }
    }
     
    double dTime_Int = GetTime();
    dTime_One += dTime_Int - dStartTime;

    if (success==0) {
      
    #ifdef VERBOSE
      cout << rp + "   No match at all for halo <" << ii << ">" << endl;
    #endif 
      
      vResult[ii].ind_first= -1;
      vResult[ii].frac_first = 0;
      vResult[ii].ind_second= -1;
      vResult[ii].frac_second= 0;
      vResult[ii].link_mass = 0;

    } else {
      
      double dTime_HistStart = GetTime();
      
      // Can make a shortcut if there is only ONE identified halo:
      if (minval == maxval) {
	vResult[ii].ind_first = minval;
	
	long long counter = vHistogram.at(minval-MinMatchHalo);
	
	// vResult[ii].frac_first = static_cast<double>(counter);//  /vTraceLength.at(ii);

	vResult[ii].frac_first = static_cast<double>(counter)/vTraceLength.at(ii);
	vResult[ii].link_mass = static_cast<int>(counter);

	vResult[ii].ind_second=-1;
	vResult[ii].frac_second=0;
	

      } else { // ends shortcut section if there is only one identified halo

	long ind_first = -1; 
	long ind_second = -1;
	
	//cout << rp + "TEST: MinMatchHalo = " << MinMatchHalo << endl;
	//cout << rp + "TEST: vTraceOffset[ii] = " << vTraceOffset.at(ii) << endl;
	//cout << rp + "TEST: vTaskMatchHaloes[ii] = " << vTraceOffset.at(ii) << endl;
	
	long long val_first = -5; 
	long long val_second = -5;

	long long CurrHistoVal;
	long CurrProgHalo;
	
	int nSucc = 0;

	for (long long jj=vTraceOffset.at(ii);jj<vTraceOffset.at(ii)+vTraceLength.at(ii);jj++) {
	  CurrProgHalo = vTaskMatchHaloes.at(jj);
	  if (CurrProgHalo < 0)
	    continue;
	  
	  CurrHistoVal = vHistogram.at(CurrProgHalo-MinMatchHalo);

	  if (CurrHistoVal > 0)
	    nSucc++;
	  
	  if (CurrHistoVal > val_second) {
	    if (CurrHistoVal > val_first) {
	      
	      // Current progenitor halo is new top entry:
	      val_second = val_first;
	      ind_second = ind_first;
	      val_first = CurrHistoVal;
	      ind_first = CurrProgHalo;

	    } else {

	      // Current progenitor halo is 'only' new second entry:
	      val_second = CurrHistoVal;
	      ind_second = CurrProgHalo;
	    
	    }
	  } // ends section only if current entry makes it into list

	  // Important: we need to re-set the histogram to zero!!!
	  // (this also ensures we don't re-load a later occurrence of the primary as secondary)

	  vHistogram.at(CurrProgHalo-MinMatchHalo) = 0;
	    

	  /**
	  
	  if (vProgHistogram.at(jj) > val_second) {
	    if (vProgHistogram.at(jj) > val_first) {
	      val_second = val_first;
	      val_first = vProgHistogram.at(jj);
	      ind_second=ind_first;
	      ind_first=jj+minval;
	    } else {
	      val_second = vProgHistogram.at(jj);
	      ind_second = jj+minval;
	    }
	  }

	  **/


	}
      
	if (nSucc < 2) {
	  cout << "Why are there not at least two potential progenitors??" << endl;
	  exit(44);
	}

	if (val_second == 0) {
	  cout << "Why is there no secondary?" << endl;
	  exit(45);
	}

      double dTime_End = GetTime();
      dTime_Two += dTime_End - dTime_HistStart;

      #ifdef VERBOSE
      if (rank == 0 && ii == 0) {
	cout << rp + "TEST: Secondary = " << ind_second << endl;
	cout << rp + "TEST:      size = " << val_second << endl;
	//exit(42);
      }
      #endif
      

      // We're almost done now - just write result!
      
      vResult[ii].ind_first = ind_first;
      vResult[ii].ind_second = ind_second;

      /* NOT MODIFIED */
      vResult[ii].frac_first = static_cast<double>(val_first)/vTraceLength.at(ii);
      vResult[ii].frac_second = static_cast<double>(val_second)/vTraceLength.at(ii);
      vResult[ii].link_mass = static_cast<int>(val_first);

      dTime_Three += GetTime() - dTime_End;
  
    } // ends section only for haloes with at least two potential progenitors.
    } // ends section only for haloes that have at least one traced particle.

    

  } // ends loop through haloes
  
  report<double>(ElTime(), "Linking haloes took [sec.]:");

  //cout << rp + "   --> Histogram creation: " << dTime_One << " sec." << endl;
  //cout << rp + "   --> Histogram analysis: " << dTime_Two << " sec." << endl;
  //cout << rp + "   --> Result writing:     " << dTime_Three << " sec." << endl;
  
  return;

}


std::vector<Result> tracing_result(const std::vector<Result> &vTaskResult, 
				   std::vector<long> &vTraceLocList, 
				   std::vector<long> &vHaloList) {


  using namespace std;
  
  int rank = 0;
  int rc = 0;
  int numtasks = 0;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  string rp = "[" + to_string(static_cast<long long>(rank)) + "]: ";
  
  double dDummy = ElTime();

  if (rank == 0)
    cout << rp + "Combining results from individual tasks..." << endl;

 
  // --------------------------------------------------------------
  
  
  vector<Result> FullResult(1); // Make dummy, and resize later on task 0

  
  // If this is NOT task 0, we need to send our results to task 0.

  MPI_Status send_stats[7];
  MPI_Request send_reqs[7];
  
  if (rank != 0) {
    
    long nTraceHaloes = vTaskResult.size(); 
    vector<long long> IndFirst(nTraceHaloes, -1);
    vector<long long> IndSec(nTraceHaloes, -1);
    
    vector<double> FracFirst(nTraceHaloes, -1);
    vector<double> FracSec(nTraceHaloes, -1);
    vector<int> LinkMass(nTraceHaloes, -1);


    for (long ii=0; ii<nTraceHaloes; ii++) {
      
      IndFirst.at(ii) = vTaskResult[ii].ind_first;
      IndSec.at(ii) = vTaskResult[ii].ind_second;
      FracFirst.at(ii) = vTaskResult[ii].frac_first;
      FracSec.at(ii) = vTaskResult[ii].frac_second;
      LinkMass.at(ii) = vTaskResult[ii].link_mass;
    }

    rc = MPI_Isend(&nTraceHaloes, 1, MPI_LONG, 0, 29, MPI_COMM_WORLD, &send_reqs[5]);

    rc = MPI_Isend(&vTraceLocList.front(), nTraceHaloes, MPI_LONG, 0, 28, MPI_COMM_WORLD, &send_reqs[4]);

    rc = MPI_Isend(&IndFirst.front(), nTraceHaloes, MPI_LONG_LONG_INT, 0, 24, MPI_COMM_WORLD, &send_reqs[0]);
    rc = MPI_Isend(&IndSec.front(), nTraceHaloes, MPI_LONG_LONG_INT, 0, 25, MPI_COMM_WORLD, &send_reqs[1]);
    rc = MPI_Isend(&FracFirst.front(), nTraceHaloes, MPI_DOUBLE, 0, 26, MPI_COMM_WORLD, &send_reqs[2]);
    rc = MPI_Isend(&FracSec.front(), nTraceHaloes, MPI_DOUBLE, 0, 27, MPI_COMM_WORLD, &send_reqs[3]);
    rc = MPI_Isend(&LinkMass.front(), nTraceHaloes, MPI_INT, 0, 270, MPI_COMM_WORLD, &send_reqs[6]); 

    MPI_Waitall(7, send_reqs, send_stats);
     
    // cout << rp + "Finished sending my partial results! " << endl;

  } else {
    // We are now on rank==0 only
    
    /**
    cout << rp + "Special section for master..." << endl;

    cout << rp + "vHaloList.size()     = " << vHaloList.size() << endl;
    cout << rp + "vTaskResult.size()   = " << vTaskResult.size() << endl;
    cout << rp + "vTraceLocList.size() = " << vTraceLocList.size() << endl;
    cout << rp + "FullResult.size()    = " << FullResult.size() << endl;
    **/

    long nFullTraceHaloes = vHaloList.size();
    FullResult.resize(nFullTraceHaloes);
    
    // Reset vHaloList, and initialise FullResult
    for (long ii=0; ii<nFullTraceHaloes; ii++) {
      vHaloList.at(ii) = -10;
      FullResult[ii].ind_first = -9;
      FullResult[ii].ind_second = -9;
      FullResult[ii].frac_first = -9;
      FullResult[ii].frac_second = -9;
    }

    // Easy bit first: Use its own results.
    long nTaskTraceHaloes = vTaskResult.size();

    #ifdef VERBOSE
    cout << rp + "Now looping through " << nTaskTraceHaloes << " individual haloes..." << endl;
    #endif
    
    for (long ii=0; ii<nTaskTraceHaloes; ii++) {
      long CurrLoc = vTraceLocList.at(ii);
      FullResult[CurrLoc] = vTaskResult[ii];
      vHaloList.at(CurrLoc) = vTaskResult[ii].ind_first;
    }

    //cout << rp + "Finished incorporating own results." << endl;

    // Now the more involved bit: Receive and incorporate the results from other tasks...

    for (int ii=1; ii<numtasks; ii++) {

      MPI_Status rec_stats[7];
      MPI_Request rec_reqs[7];

      MPI_Status num_stats[1];
      MPI_Request num_reqs[1];
      
      long nTraceHaloes = -1;

      rc = MPI_Irecv(&nTraceHaloes, 1, MPI_LONG, ii, 29, MPI_COMM_WORLD, &num_reqs[0]);
      MPI_Wait(&num_reqs[0], &num_stats[0]);

      vector<long> vRemoteTraceLocList(nTraceHaloes, -1);

      vector<long long> IndFirst(nTraceHaloes, -1);
      vector<long long> IndSec(nTraceHaloes, -1);
      
      vector<double> FracFirst(nTraceHaloes, -1);
      vector<double> FracSec(nTraceHaloes, -1);

      vector<int> LinkMass(nTraceHaloes, -1);

      rc = MPI_Irecv(&vRemoteTraceLocList.front(), nTraceHaloes, MPI_LONG, ii, 28, MPI_COMM_WORLD, &rec_reqs[4]);
      rc = MPI_Irecv(&IndFirst.front(), nTraceHaloes, MPI_LONG_LONG_INT, ii, 24, MPI_COMM_WORLD, &rec_reqs[0]);
      rc = MPI_Irecv(&IndSec.front(), nTraceHaloes, MPI_LONG_LONG_INT, ii, 25, MPI_COMM_WORLD, &rec_reqs[1]);
      rc = MPI_Irecv(&FracFirst.front(), nTraceHaloes, MPI_DOUBLE, ii, 26, MPI_COMM_WORLD, &rec_reqs[2]);
      rc = MPI_Irecv(&FracSec.front(), nTraceHaloes, MPI_DOUBLE, ii, 27, MPI_COMM_WORLD, &rec_reqs[3]);
      rc = MPI_Irecv(&LinkMass.front(), nTraceHaloes, MPI_INT, ii, 270, MPI_COMM_WORLD, &rec_reqs[5]); 

      MPI_Waitall(6, rec_reqs, rec_stats);

      
      for (long jj = 0; jj<nTraceHaloes; jj++) {
	
	long CurrLoc = vRemoteTraceLocList.at(jj);
	FullResult[CurrLoc].ind_first = IndFirst.at(jj);
	FullResult[CurrLoc].ind_second = IndSec.at(jj);

	FullResult[CurrLoc].link_mass = LinkMass.at(jj);

	FullResult[CurrLoc].frac_first = FracFirst.at(jj);
	FullResult[CurrLoc].frac_second = FracSec.at(jj);
	
	vHaloList.at(CurrLoc) = IndFirst.at(jj);
	
      }


    } // ends loop through tasks

    // --- NEW BIT: Need to check for duplicates in new vHaloList... ---

    cout << "Beginning to remove duplicates..." << endl;
    remove_halo_duplicates(FullResult, vHaloList);
    
  } // ends section for rank==0 only
  
  if (rank == 0)
    cout << rp + "   ...finished (" << ElTime() << " sec.) " << endl;

  return FullResult;

} 

void remove_halo_duplicates(std::vector<Result> &FullResult, 
			    std::vector<long> &vHaloList) {
  

  using namespace std;
  
  int success=0;
  vector<long> vHaloMinMax = minmax<long>(vHaloList,0,success);
  if (success==0) {
    cerr << "Halo list empty!!!" << endl;
    cerr << "Continuing, but things are probably going to go downhill from here..." << endl;
    return;
  }

  print_vector<long>(vHaloMinMax, "vHaloMinMax");

  cout << "RHD: Setting up initial histogram..." << endl;

  // Set up a histogram to find duplicate halo numbers:
  long nHaloSpan = vHaloMinMax[1]-vHaloMinMax[0]+1;
  long nHaloes = vHaloList.size();
  long nHaloOffset = vHaloMinMax[0];
  vector<long> vHaloHistogram(nHaloSpan,0);
  long CurrHalo;
  
  if (nHaloOffset < 0) {
    cout << "Why is the Halo Offset negative???" << endl;
    exit(48); }

  for (long ii=0; ii<nHaloes; ii++) {
    CurrHalo = vHaloList[ii];
    if (CurrHalo >= 0)
      vHaloHistogram[CurrHalo-nHaloOffset] += 1;
  }
  
  #ifdef VERBOSE
    cout << "RHD: Making reverse-index offsets..." << endl;
    #endif


  // Make reverse-index list

  vector<long> vRevIndOffset(nHaloSpan+1);
  vRevIndOffset[0]=0;
  
  for (long ii=0; ii<nHaloSpan; ii++) {
    vRevIndOffset.at(ii+1) = vRevIndOffset.at(ii)+vHaloHistogram[ii];
  }
  
  vector<long> vRevIndList(nHaloes);
  vector<int> vSublist(nHaloSpan);

  #ifdef VERBOSE
  cout << "RHD: Making reverse-index list..." << endl;
  #endif

  long CurrHistoInd;
  for (long ii=0; ii<nHaloes; ii++) {

    try{

    CurrHalo = vHaloList.at(ii);
    if (CurrHalo < 0)
      continue;

    CurrHistoInd = CurrHalo-nHaloOffset;
    vRevIndList.at(vRevIndOffset.at(CurrHistoInd) + vSublist.at(CurrHistoInd)) = ii;
    vSublist.at(CurrHistoInd) += 1;
    }

    catch (const std::out_of_range &oor) {
      std::cerr << "Out of Range error: " << oor.what() << endl;
      cerr << "ii = " << ii << endl;
      cerr << "nHaloes = " << nHaloes << endl;
      cerr << "CurrHalo = " << CurrHalo << endl;
      cerr << "nHaloOffset = " << nHaloOffset << endl;
      cerr << "CurrHistoInd = " << CurrHistoInd << endl;
      
      cerr << "vRevIndOffset[CurrHistoInd] = " << vRevIndOffset.at(CurrHistoInd) << endl;
      cerr << "vSublist[CurrHistoInd] = " << vSublist.at(CurrHistoInd) << endl;
      
      exit(47);
    }
    
  }

  cout << "RHD: Deleting temporary vSublist vector..." << endl;
  vSublist.resize(1);

  #ifdef VERBOSE
  cout << "RHD: Now searching for duplicates..." << endl;
  #endif

  // Find number of duplicated halo numbers:
  int nDuplicates = 0;

  for (long ii=0; ii<nHaloSpan; ii++) {
    if (vHaloHistogram[ii] > 1) {
      nDuplicates++;

      #ifdef VERBOSE
      cout << " ---> Dealing with duplicate " << nDuplicates << endl;
      #endif

      vector<long> vDuplLocs(vHaloHistogram[ii]);
      vector<int> vDuplOrigMass(vHaloHistogram[ii]);

      for (int jj = 0; jj < vHaloHistogram[ii]; jj++) {
	vDuplLocs[jj] = vRevIndList.at(vRevIndOffset.at(ii)+jj); 
	vDuplOrigMass[jj] = FullResult[vDuplLocs[jj]].link_mass;
      }
      
      cout << "     Made temporary vectors..." << endl;

      // Find instance with maximum mass:
      int nMaxMass = 0;
      int nIndOfMaxMass = -1;
      
      for (int jj = 0; jj < vHaloHistogram[ii]; jj++) {
	if (vDuplOrigMass[jj] > nMaxMass) {
	  nMaxMass = vDuplOrigMass[jj];
	  nIndOfMaxMass = jj; }
      }

      cout << "     Found primary instance: " << nIndOfMaxMass << endl;

      // Almost done! Set all other entries in halo list to -4
      
#ifdef VERBOSE
      cout << "nHaloOffset = " << nHaloOffset << endl;
      cout << "Size of vHaloList = " << vHaloList.size() << endl;
      print_vector<long>(vDuplLocs, "vDuplLocs");
      
      for (int jj=0; jj<vDuplLocs.size(); jj++)
	cout << jj << ": " << vHaloList.at(vDuplLocs.at(jj)) << endl;

      cout << "Haloes: ";
#endif

      for (int jj = 0; jj< vHaloHistogram[ii]; jj++) {
	#ifdef VERBOSE
	cout << vDuplLocs.at(jj)+nHaloOffset;
	#endif

	if (jj != nIndOfMaxMass) {
	  #ifdef VERBOSE
	  cout << "X ";
	  #endif

	  vHaloList.at(vDuplLocs.at(jj)) = -4;
	  FullResult[vDuplLocs.at(jj)].ind_first = -4;
	} 
	#ifdef VERBOSE
	else {
	  cout << "< ";
	}
	#endif
      }
      #ifdef VERBOSE
      cout << endl;
      #endif

    } // ends section for duplicated histogram entries

  } // ends loop through histogram

    cout << "---> Dealt with " << nDuplicates << " instances of duplicated progenitors." << endl;

  return;

}
