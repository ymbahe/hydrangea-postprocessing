// snipshots.cpp
// Functions to extract IDs of most bound particles, to be used for snipshot tracing
// Started WED 18 NOV 2015


#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "Config.h"
#include "globals.h"

#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/io_hdf5.h"


// **************************************************************************************
// Main function: extract_ids_for_snipshots()
// This is the (public) wrapper function which is called from main().
// It distributes tracing results to tasks, identifies relevant IDs, 
// collects results again at master task, and finally writes result.
// **************************************************************************************

void extract_ids_for_snipshots(const Result &TracingResult,         // Result structure 
			       const std::vector<int> &vTHOlist,    // vTHOlistA
			       const std::vector<unsigned long> &vIDs,       // vTaskIDsA
			       const std::vector<long> &vPartOffset,  // Full part offset (A)
			       const std::vector<int> &vMatchHaloB,  // Match halo in B
			       const std::vector<int> &vMatchHaloC,  // Match halo in C
			       const std::vector<long> &vOrderB,  // order in B
			       const std::vector<long> &vOrderC,  // order in C
			       const std::vector<int> &vMHLListA,    // SH->Ind
			       int nGalaxiesTot,                // Number of galaxies (at A)
			       int nFlagReverse=0               // [1 = reverse tracing]
			       )    
  
{
  function_switch("extract_ids_for_snipshots");
  using namespace std;

  static vector<unsigned long> vStoreIDs_Fwd;   // IDs to be kept for next iteration ('in store')
  static vector<int> vStoreGalList_Fwd; // Full galaxy number for each 'store inhabitant'
  static vector<int> vStoreOffset_Fwd;  // Offset into vStoreIDs for each inhabitant

  static vector<unsigned long> vStoreIDs_Rev;   // IDs to be kept for next iteration ('in store')
  static vector<int> vStoreGalList_Rev; // Full galaxy number for each 'store inhabitant'
  static vector<int> vStoreOffset_Rev;  // Offset into vStoreIDs for each inhabitant

  vector<unsigned long> &vStoreIDs = (nFlagReverse == 0) ? vStoreIDs_Fwd : vStoreIDs_Rev;
  vector<int> &vStoreGalList = (nFlagReverse == 0) ? vStoreGalList_Fwd : vStoreGalList_Rev;
  vector<int> &vStoreOffset = (nFlagReverse == 0) ? vStoreOffset_Fwd : vStoreOffset_Rev;
  
  int rc = 0;
  double dDummy = ElTime();
  double dStartTime = GetTime();

  // ------------------------------------
  // 1.) Send results to individual tasks
  // ------------------------------------

#ifdef VERBOSE_SNIPSHOTS
  if (mpirank() == 0)
    cout << rp() + "Sending tracing results to individual tasks..." << endl;
#endif 

  vector<int> vTempMatch, vTempLength;   /* declare here, only significant for root */
  vector<int> vHaloesPerTask;
  

  if (mpirank() == 0) {
    vTempMatch.resize(g_nHaloesA);
    vTempLength.resize(g_nHaloesA);
    
    for (int ii = 0; ii < g_nHaloesA; ii++) {
      vTempMatch.at(ii) = TracingResult.Match.at(ii);
      vTempLength.at(ii) = TracingResult.Length.at(ii);

    }

    vHaloesPerTask.resize(numtasks());
    for (int ii = 0; ii < numtasks(); ii++)
      vHaloesPerTask.at(ii) = vTHOlist.at(ii+1)-vTHOlist.at(ii);
  }

  int nHaloesThisTask = vTHOlist.at(mpirank()+1) - vTHOlist.at(mpirank());
  vector<int> vTT_Match(nHaloesThisTask,-1);
  vector<int> vTT_Length(nHaloesThisTask,-1);

  vector<int> vTempOffset(vTHOlist.size());
  for (int iii = 0; iii < vTHOlist.size(); iii++)
    vTempOffset.at(iii) = vTHOlist.at(iii);
  
  rc = MPI_Scatterv(&vTempMatch.front(), &vHaloesPerTask.front(), &vTempOffset.front(), MPI_INT, 
		    &vTT_Match.front(), nHaloesThisTask, MPI_INT, 0, MPI_COMM_WORLD);

  rc = MPI_Scatterv(&vTempLength.front(), &vHaloesPerTask.front(), &vTempOffset.front(), MPI_INT, 
		    &vTT_Length.front(), nHaloesThisTask, MPI_INT, 0, MPI_COMM_WORLD);
  
  vTempOffset.clear();

  if (mpirank() == 0) {
    vTempMatch.clear();
    vTempLength.clear();
  }

  report<double> (ElTime(), "... sending results to tasks took [sec.]");

  // ----------------------------------------------------------
  // 2.) Each task individually extracts the IDs for its haloes
  // ----------------------------------------------------------
  
  vector<unsigned long> vTT_SnipIDs(0);  // Initialise as EMPTY
  vTT_SnipIDs.reserve(nHaloesThisTask * NUM_SNIPIDS);
  vector<int> vTT_SnipOffset(nHaloesThisTask+1, 0); // Know the length of this in advance!
  
  int nTT_HOffset = vTHOlist.at(mpirank());
  long nTT_POffset = vPartOffset.at(vTHOlist.at(mpirank()));

  // Loop over the haloes belonging to current task
  for (int iihalo = 0; iihalo < nHaloesThisTask; iihalo++) {

    int nHaloLength = vTT_Length.at(iihalo);
    int nHaloMatch = vTT_Match.at(iihalo);

    // Loop over particles belonging to current halo

    int nParticleCounter = 0; // to keep track of how many particles we've already loaded
                              // in this current halo

    if (nFlagReverse == 0) {

    for (long nCurrPart = vPartOffset.at(iihalo+nTT_HOffset) - nTT_POffset;
	 nCurrPart < vPartOffset.at(iihalo+1+nTT_HOffset) - nTT_POffset;
	 nCurrPart++) {

      unsigned long nCurrPartID = vIDs.at(nCurrPart);
      int nCurrMatch = -1;
      int LoadParticle = 0;

      if (nHaloMatch < 0)
	LoadParticle = 1;  // Rationale: If this is the last snapshot of the galaxy, just take 20 most bound
      else {
	
	if (nHaloLength == 1) 
	  {
	    if (vMatchHaloB.at(nCurrPart) == nHaloMatch)
	      LoadParticle = 1;
	  }
	
	else if (nHaloLength == 2) 
	  {
	    if (vMatchHaloC.at(nCurrPart) == nHaloMatch)
	      LoadParticle = 1;
	  }
	
      } // ends section only if there is a match for current halo
      
      if (LoadParticle == 1) 
	{
	  vTT_SnipIDs.push_back(nCurrPartID);
	  nParticleCounter++;
	  if (nParticleCounter >= NUM_SNIPIDS)
	    break;
	}
      
    } // ends loop through particles in current subhalo
    } // ends "traditional" identification (particles most bound in A)
    else {

      // This bit is the 'Reverse' ID identification (added Apr 2016)
      // Need to check whether this is the last snapshot of galaxy
      // (nHaloMatch < 0), because in this case there IS NO POSSIBLE 'reverse' identification
      // Instead, we write '0', and let sniplocate handle this situation.
      // NB: We cannot write '-1', because this will cause a very large ID and will
      // cause a tripout of verify_id_range() in sniplocate...

      // (1) Write out, into separate vectors, the IDs and orders of link particles 

      long nPartThisHalo = vPartOffset.at(iihalo+1+nTT_HOffset)-vPartOffset.at(iihalo+nTT_HOffset);
      long nLinkParts = 0;

      if (nHaloMatch >= 0) {
	
	std::vector<unsigned long> vLinkIDs(0);
	std::vector<long> vLinkOrder(0);
	vLinkIDs.reserve(nPartThisHalo); // Reserve maximum possible
	vLinkOrder.reserve(nPartThisHalo);
	
	for (long nCurrPart = vPartOffset.at(iihalo+nTT_HOffset) - nTT_POffset;
	     nCurrPart < vPartOffset.at(iihalo+1+nTT_HOffset) - nTT_POffset;
	     nCurrPart++) {
	  
	  unsigned long nCurrPartID = vIDs.at(nCurrPart);
	  int nCurrMatch = -1;
	  int PreLoadParticle = 0;
	  long nCurrOrder = -1;
	  
	  if (nHaloLength == 1) {
	    if (vMatchHaloB.at(nCurrPart) == nHaloMatch) {
	      PreLoadParticle = 1;
	      nCurrOrder = vOrderB.at(nCurrPart);
	    }
	  }
	  
	  else if (nHaloLength == 2) { 
	    if (vMatchHaloC.at(nCurrPart) == nHaloMatch) {
	      PreLoadParticle = 1;
	      nCurrOrder = vOrderC.at(nCurrPart);
	    }
	  }
	  
	  else {
	    std::cout << "Unexpected nHaloLength = " << nHaloLength << " encountered, terminating." << std::endl;
	    exit(99999);
	  }

	  if (PreLoadParticle == 1) 
	    {
	      vLinkIDs.push_back(nCurrPartID);
	      vLinkOrder.push_back(nCurrOrder);
	    }

	} // ends loop through particles
	     
	// We now order the extracted (link) particles by their 'vLinkOrder'
	std::vector<long> curr_order_sort = sort_indices<long>(vLinkOrder); 

	// Finally, we go through the list in the just-calculated order,
	// and extract the first X particles.

	nLinkParts = vLinkOrder.size();
	if (nLinkParts > NUM_SNIPIDS)
	  nLinkParts = NUM_SNIPIDS;       // Limit on number of particles returned
	
	for (long iiLinkPart = 0; iiLinkPart < nLinkParts; iiLinkPart++) {
	  long nIndCurr = curr_order_sort.at(iiLinkPart);
	  vTT_SnipIDs.push_back(vLinkIDs.at(nIndCurr));
	  nParticleCounter++;
	}
	

      } // ends section only if current halo could be traced.

      else {
	nParticleCounter = NUM_SNIPIDS;
	if (nParticleCounter > nPartThisHalo)
	  nParticleCounter = nPartThisHalo;
	
	for (long iiLinkPart = 0; iiLinkPart < nParticleCounter; iiLinkPart++) {
	  vTT_SnipIDs.push_back(0);
	}
      }
      
    } // ends 'reverse' ID identification
    
    vTT_SnipOffset.at(iihalo+1) = vTT_SnipOffset.at(iihalo) + nParticleCounter;
  
  }  // ends loop through subhaloes
  
  int nTT_SnipIDs = vTT_SnipOffset.back();

  report<double> (ElTime(), "... extracting tracing IDs for snipshots took [sec.]");


  // --------------------------------------
  // 3.) Assemble full ID list on root task
  // --------------------------------------
  
  
  // Build full SnipOffset vector on task 0
  // NOTE: This CLEARS the data from the original tasks
  collect_vector_mpi(vTT_SnipOffset, 0, 1, 1);  /* the last '1' specifies an offset list */

  // For clarity, re-name the (full) offset vector (significant only on task 0)
  vector<int> vAllSH_SnipOffset;
  vAllSH_SnipOffset.swap(vTT_SnipOffset);

  // Allocate the (final) vector holding all the IDs (including those from store)
  int nSnipIDs = vAllSH_SnipOffset.back();

  int nSnipIDs_Store = vStoreIDs.size();

  vector<unsigned long> vFullSnipIDs;

  MPI_Request reqs[2];
  MPI_Status stats[2];
  
  // Every NON-ROOT task now sends their IDs towards root
  if (mpirank() > 0) {
    if (vTT_SnipIDs.size() > 0) {
      rc = MPI_Isend(&vTT_SnipIDs.front(), vTT_SnipIDs.size(), MPI_UNSIGNED_LONG, 0, 2223, MPI_COMM_WORLD, &reqs[0]);
      MPI_Wait(&reqs[0], &stats[0]);
    } // ends section only if this task needs to send any data
  }
  
  // The next bit is only for root:
  if (mpirank() == 0) {
    vFullSnipIDs.resize(nSnipIDs+nSnipIDs_Store, 0);
    
    // ---> Build 'allocation plan' in full ID list

    // (a) Set up a vector holding the number of (for-snipshot) IDs per galaxy
    vector<int> vNumIDsByGal(nGalaxiesTot, 0);
    
    // Fill in numbers for each Halo (! not galaxy !)
    for (int iihalo = 0; iihalo < vTHOlist.back(); iihalo++) {
      int nCurrGal = vMHLListA.at(iihalo);
      int nIDsCurrHalo = vAllSH_SnipOffset.at(iihalo+1)-vAllSH_SnipOffset.at(iihalo);
      vNumIDsByGal.at(nCurrGal) = nIDsCurrHalo;
    }

    // (b) Fill in numbers for each galaxy FROM STORE

    for (int iistore = 0; iistore < vStoreGalList.size(); iistore++) {
      int nCurrGal = vStoreGalList.at(iistore);
      int nIDsCurrHalo = vStoreOffset.at(iistore+1)-vStoreOffset.at(iistore);
      vNumIDsByGal.at(nCurrGal) = nIDsCurrHalo;
    }
    

    // (c) Convert vNumIDsByGal into an offset list
    vector<int> vOffsetsByGal(nGalaxiesTot+1,0);
    for (int iigal = 0; iigal < nGalaxiesTot; iigal++) 
      vOffsetsByGal.at(iigal+1) = vOffsetsByGal.at(iigal) + vNumIDsByGal.at(iigal);

    // <--- Allocation plan done.
    // Probably best to empty the store first...

    cout << rp() + "Processing IDs stored from last snapshot..." << endl;

    int nCurrPartFromStore = 0;
    for (int iistore = 0; iistore < vStoreGalList.size(); iistore++) {
      
      // Find galaxy index of current halo:
      int nCurrGalaxy = vStoreGalList.at(iistore);
      int nCurrOffsetInResult = vOffsetsByGal.at(nCurrGalaxy);
      int nCurrIDs = vStoreOffset.at(iistore+1) - vStoreOffset.at(iistore);
	
      int nExpectedPartOffset = vStoreOffset.at(iistore);
	
      if (nCurrPartFromStore != nExpectedPartOffset) {
	  cout << rp() + "Inconsistency detected: nCurrPartFromStore (= " << nCurrPartFromStore << ") != nExpectedPartOffset (= " << nExpectedPartOffset << ")" << endl;
	  exit(5681); }
	
      for (int kkpart = 0; kkpart < nCurrIDs; kkpart++) {
	vFullSnipIDs.at(nCurrOffsetInResult+kkpart) = vStoreIDs.at(nCurrPartFromStore); 
	nCurrPartFromStore++;
      }

    } // ends loop through store haloes. Store can now be emptied!

    vStoreIDs.clear();
    vStoreOffset.clear();
    vStoreOffset.resize(1,0);   // to prepare it for the next load
    vStoreGalList.clear();
        
    cout << rp() + "...done!" << endl;

    // Now loop through individual tasks and incorporate their IDs...
    for (int iitask = 0; iitask < numtasks(); iitask++) {
      
      int nCT_Haloes = vTHOlist.at(iitask+1)-vTHOlist.at(iitask);
      int nCT_HaloOffset = vTHOlist.at(iitask);
      int nCT_IDs = vAllSH_SnipOffset.at(nCT_HaloOffset+nCT_Haloes)-vAllSH_SnipOffset.at(nCT_HaloOffset);
      
#ifdef DEBUG
      cout << rp() + "Processing task " << iitask << ":" << endl;
      cout << rp() + "   " << nCT_Haloes << " haloes (first = " << nCT_HaloOffset << ")" << endl;
      cout << rp() + "   " << nCT_IDs << " IDs" << endl;
#endif      

      vector<unsigned long> vCurrTaskIDs(nCT_IDs, 0);
	
      // First bit is special: its own data
      if (iitask == 0) {

	if (nCT_IDs != vTT_SnipIDs.size()) {
	  cout << rp() + "Inconsistency detected: nCT_IDs (= " << nCT_IDs << ") != vTT_SnipIDs.size (= " << vTT_SnipIDs.size() << ")" << endl;
	  exit(5679); }
	
	for (int jj = 0; jj < nCT_IDs; jj++)
	  vCurrTaskIDs.at(jj) = vTT_SnipIDs.at(jj);
	
      }	else {

#ifdef DEBUG
	cout << rp() + "   ... receiving " << nCT_IDs << " IDs..." << endl; 
#endif 
	
	if (nCT_IDs > 0) {
	  rc = MPI_Irecv(&vCurrTaskIDs.front(), nCT_IDs, MPI_UNSIGNED_LONG, iitask, 2223, MPI_COMM_WORLD, &reqs[1]);
	  MPI_Wait(&reqs[1], &stats[1]);
	}

#ifdef DEBUG
	cout << rp() + "   ......done! " << endl;
#endif 
	
      } 
      
      // Rest is the same for all tasks - IDs now in vCurrTaskIDs.

      // Need to go through each halo, find out where its IDs sit in full list, and 
      // put them there.
      // NB: jjhalo is the FULL halo index (over all chunks)

      int nCurrPartFromTask = 0;
      for (int jjhalo = nCT_HaloOffset; jjhalo < (nCT_HaloOffset + nCT_Haloes) ; jjhalo++) {

	// Find galaxy index of current halo:
	int nCurrGalaxy = vMHLListA.at(jjhalo);
	int nCurrOffsetInResult = vOffsetsByGal.at(nCurrGalaxy);
	int nCurrIDs = vAllSH_SnipOffset.at(jjhalo+1) - vAllSH_SnipOffset.at(jjhalo);
	
	int nExpectedPartOffset = static_cast<int>(vAllSH_SnipOffset.at(jjhalo) - vAllSH_SnipOffset.at(nCT_HaloOffset));
	
	if (nCurrPartFromTask != nExpectedPartOffset) {
	  cout << rp() + "Inconsistency detected: nCurrPartFromTask (= " << nCurrPartFromTask << ") != nExpectedPartOffset (= " << nExpectedPartOffset << ")" << endl;
	  exit(5680); }
	
	for (int kkpart = 0; kkpart < nCurrIDs; kkpart++) {
	  vFullSnipIDs.at(nCurrOffsetInResult+kkpart) = vCurrTaskIDs.at(nCurrPartFromTask); 
	  nCurrPartFromTask++;
	}
	
	// NOW check whether current halo is bypassed in NEXT snapshot - if it is, 
	// then we need to add it to the store

	if (TracingResult.Length.at(jjhalo) == 2) {
	  
	  vStoreIDs.reserve(vStoreIDs.size()+nCurrIDs);  // make sure it can take all IDs
	  
	  for (int kkpart = 0; kkpart < nCurrIDs; kkpart++)
	    vStoreIDs.push_back(vCurrTaskIDs.at(nExpectedPartOffset+kkpart));
	  
	  vStoreGalList.push_back(nCurrGalaxy);
	  vStoreOffset.push_back(vStoreOffset.back()+nCurrIDs);

	}

      } // ends loop through haloes
      
    } // ends loop through all tasks to incorporate their IDs
    
    // Righty-oh. We now have the two output vectors (on task 0):
    // -- vFullSnipIDs    --> The IDs to be written out, sorted by galaxy number
    // -- vOffsetsByGal   --> The offset by galaxy number into the ID list
    

    std::string sSnipGroup;
    if (nFlagReverse == 0)
      sSnipGroup = "Snapshot_" + to_string(g_nSnapA, 3) + "/SnipshotIDs";
    else
      sSnipGroup = "Snapshot_" + to_string(g_nSnapA, 3) + "/SnipshotIDs_Reverse";
    
    hdf5_create_group(g_sFileNameOut, sSnipGroup);

    // Write IDs
    std::vector<int> vIDsSize(1);
    vIDsSize.front() = vFullSnipIDs.size();
    hdf5_create_dataset(g_sFileNameOut, sSnipGroup + "/IDs", vIDsSize, "unsigned long", "All IDs to be used for tracing galaxies through snipshots until the next snapshot. The accompanying dataset 'Offset' says where the IDs for each individual galaxy start.", 0);
    write_hdf5_data(g_sFileNameOut, sSnipGroup + "/IDs", vFullSnipIDs);


    // Write offsets
    std::vector<int> vOffsetSize(1);
    vOffsetSize.front() = vOffsetsByGal.size();
    hdf5_create_dataset(g_sFileNameOut, sSnipGroup + "/Offset", vOffsetSize, "int", "The offset of each galaxy in the ID list. The IDs for galaxy N are located from position Offset[N] - Offset[N+1]-1.", -1);
    write_hdf5_data(g_sFileNameOut, sSnipGroup + "/Offset", vOffsetsByGal);
    
    cout << rp() + "Finished writing ID datasets." << rt() << endl;
    
  } // ends of root-only section
  
  function_switch("extract_ids_for_snipshots");
  return;

}

