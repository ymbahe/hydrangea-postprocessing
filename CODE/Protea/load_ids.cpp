#include "Config.h"

#include <mpi.h>
#include <iostream>
//#include "/ptmp/mpa/ybahe/HDF5/include/H5Cpp.h"
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <cstdlib>

#include "init.h"
#include "globals.h"

#include "load_ids.h"
#include "utilities_special.h"

#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/hydrangea.h"
#include "/u/ybahe/cpplib/utilities_mpi.h"



// *********************************
// EXTERNAL Function implementations
// *********************************

// make_tpolist_parallel is the main (wrapper) function to generate
// the TPOlists. It is called from main(). 

std::vector<long> 
make_tpolist_parallel(std::string sFileName,                   // Input (subdir) file name
		      int &nHaloIni,                           // First halo to be used
		      int &nHaloFin,                           // Last halo to be used
		      std::vector<int> &vTHOlist,              // [O] Haloes across tasks
		      std::vector<long> &vFullPartOffset,  // [O] Offsets of all used haloes
		      std::vector<long> &vFullPartLength) {  // [O] Lengths of all used haloes

  
  function_switch("make_tpolist_parallel");


  // ----- Setup (incl. MPI and HDF5) ----
  
  using namespace std;
  

  std::vector<long> vDummyX, vDummyY;

  // NB: Added palaver after eagleread() is necessary to convert vectors
  //     to format used in rest of program.

  std::vector<unsigned int> vFullPartOffsetUI, vFullPartLengthUI;
  eagleread<unsigned int>(sFileName, "Subhalo/SubOffset", vFullPartOffsetUI, vDummyX, 2, vDummyY, 0, "Header/NumFilesPerSnapshot", "Header/Nsubgroups");

  std::vector<long> vFullPartOffsetLong(vFullPartOffsetUI.begin(), vFullPartOffsetUI.end()); 
  vFullPartOffsetUI.clear();
  vFullPartOffset.swap(vFullPartOffsetLong);
  vFullPartOffsetLong.clear();
  
  eagleread<unsigned int>(sFileName, "Subhalo/SubLength", vFullPartLengthUI, vDummyX, 2, vDummyY, 0, "Header/NumFilesPerSnapshot", "Header/Nsubgroups");

  std::vector<long> vFullPartLengthLong(vFullPartLengthUI.begin(), vFullPartLengthUI.end()); 
  vFullPartLengthUI.clear();
  vFullPartLength.swap(vFullPartLengthLong);
  vFullPartLengthLong.clear();

  // Append Coda to vFullPartOffset (!!!!)
  if (vFullPartOffset.size() > 0)
    vFullPartOffset.push_back(vFullPartOffset.back()+vFullPartLength.back());
  
  // Build TPO/THO lists:
  std::vector<long> vTPOlist(numtasks()+1);
  
  build_tpolist(vFullPartOffset, vFullPartLength, 0, vTPOlist, vTHOlist);
  
  nHaloIni = vTHOlist.front();
  nHaloFin = vTHOlist.back()-1;

  function_switch("make_tpolist_parallel");
  return vTPOlist;

}



void reject_baryons(std::vector<unsigned long> &vThisTaskIDs,          // ID list for current task (changed)
		    std::vector<long> &vFullPartOffset,       // Overall offset list (changed)
		    std::vector<long> &vFullPartLength,       // Overall length list (erased)
		    std::vector<long> &vTPOlist,              // Distrib. of IDs across tasks (ch.)
		    const std::vector<int> &vTHOlist) {       // Distrib. of Haloes across tasks
		
  function_switch("reject_baryons");
  using namespace std;

  /** General structure:
      
      - Find number of star particles
      - Calculate number of tasks over which stars are spread
      - Load star IDs on first set of these
      - Distribute to "siblings"
      - Sort star IDs
  
      - Make "keep" list (of char size) for IDs
      - Then loop over star tasks: Run through (sorted) full IDs, check for match to (local) star list, if yes, mark it as '1' in keep list. Pass star ID/sort lists to next in chain.
      - Loop over IDs (in unsorted order) and reduce the list by eliminating ones with keep == 0. Also need to update halo offset/length info...
      - Re-sort the shortened ID list
 
      CURRENTLY this is simplified to REJECT ALL BARYONS, 
      i.e. the tracing is performed ONLY using DM particles.
 
  **/

  
#ifndef DEBUG
  if (mpirank() == 0)
#endif
    cout << rp() + "Num IDs before: " << vThisTaskIDs.size() << endl;

#ifdef DEBUG
  cout << rp() + "vTPOlist.size() = " << vTPOlist.size() << endl;
  cout << rp() + "vTHOlist.size() = " << vTHOlist.size() << endl;
#endif

  long nCounterKeptIDs = 0;
  long PartOffsetOfTask = vTPOlist.at(mpirank());
  int HaloOffsetOfTask = vTHOlist.at(mpirank());
  
  int nNumHaloesThisTask = vTHOlist.at(mpirank()+1)-vTHOlist.at(mpirank());
  long nNumParticlesThisTask = vTPOlist.at(mpirank()+1)-vTPOlist.at(mpirank());
  
  vector<long> vTT_NewPartOffset(nNumHaloesThisTask+1,-1); // New particle offsets [temp.]

#ifdef DEBUG
  cout << rp() + "HaloOffsetOfTask = " << HaloOffsetOfTask << endl;
  cout << rp() + "PartOffsetOfTask = " << PartOffsetOfTask << endl;
  cout << rp() + "nNumHaloesThisTask = " << nNumHaloesThisTask << endl;
  cout << rp() + "nNumParticlesThisTask = " << nNumParticlesThisTask << endl;

#endif
  
  // Loop through haloes:
  for (int ii=0; ii<nNumHaloesThisTask; ii++) {
    
    long nRelOffsetOfCurrHalo = vFullPartOffset.at(HaloOffsetOfTask + ii)-vFullPartOffset.at(HaloOffsetOfTask);
    long nLastParticleOfCurrHalo = nRelOffsetOfCurrHalo + vFullPartLength.at(HaloOffsetOfTask + ii)-1;
    // Some bookkeeping: Need to write out the NEW (relative) offset of current halo
    vTT_NewPartOffset.at(ii) = nCounterKeptIDs;
    
    // And now loop through individual particles of current halo:
    for (long jj = nRelOffsetOfCurrHalo; jj <= nLastParticleOfCurrHalo; jj++) {

      // Check particle type of current ID:
      if (vThisTaskIDs.at(jj) % 2 == 0) {
	// Even index == DM == good
	// Move particle to position nCounterKeptIDs:
	
	vThisTaskIDs.at(nCounterKeptIDs) = vThisTaskIDs.at(jj);
	nCounterKeptIDs++;

	if (Params.nMaxTracers >= 0)
	  if (nCounterKeptIDs >= Params.nMaxTracers)
	    break;
	
      } // ends section only if current particle is DM
    } // ends loop through halo's particles
  } // ends loop through haloes
  
  // Attach coda to this task's (new) particle offset list:
  vTT_NewPartOffset.at(nNumHaloesThisTask) = nCounterKeptIDs;
  
  // Last section: Need to update particle offset lists...
  
  vThisTaskIDs.resize(nCounterKeptIDs); // All the re-allocated IDs are in this section! 
  vFullPartLength.clear(); // Not needed anymore -- ID list is now dense!

  // Send number of remaining particles to root:
  
  std::vector<long> vAllTasksNumKeptPart;
  if (mpirank() == 0)
    vAllTasksNumKeptPart.resize(numtasks());
  
  MPI_Gather(&nCounterKeptIDs, 1, MPI_LONG, &vAllTasksNumKeptPart.front(), 1, MPI_LONG, 0, MPI_COMM_WORLD);
  
  if (mpirank() == 0) {
    
    long nCurrFullOffset = 0;
    for (int kk = 0; kk < numtasks(); kk++) {
      vTPOlist.at(kk) = nCurrFullOffset;
      nCurrFullOffset += vAllTasksNumKeptPart.at(kk);
    }
    
    // Coda:
    vTPOlist.at(numtasks()) = nCurrFullOffset;

  } // ends master-modifies-vTPOlist section

  MPI_Bcast(&vTPOlist.front(), numtasks()+1, MPI_LONG, 0, MPI_COMM_WORLD);
  
  // Finally, build new halo-particle offset list:
  collect_vector_mpi<long> (vTT_NewPartOffset, 0, 1, 2);
  
  vFullPartOffset.clear();
  vFullPartOffset.swap(vTT_NewPartOffset);

  
  if (mpirank() == 0) {
    cout << rp() + "Num IDs after: " << vThisTaskIDs.size() << endl;
  }
    
  function_switch("reject_baryons");
  return;
  
}


// *******************************************************************************
// Function to distribute haloes across tasks


void build_tpolist(const std::vector<long> &vFullPartOffset, 
		   const std::vector<long> &vFullPartLength, 
		   int nHaloIni,
		   std::vector<long> &vTPOlist,
		   std::vector<int> &vTHOlist) {

  function_switch("build_tpolist");
  using namespace std;
  
  int nHaloSpan = vFullPartLength.size();

  long currcounter = 0;
  int currtask = 0;
  
  vTPOlist.clear();
  vTHOlist.clear();
  vTPOlist.resize(numtasks()+1,0);
  vTHOlist.resize(numtasks()+1,0);
  
  if (nHaloSpan == 0) 
    {
      function_switch("build_tpolist");
      return; 
    }

  long nPartTot = vFullPartOffset.back()-vFullPartOffset.front();
  
#ifdef VERBOSE
  cout << rp() + "nHaloSpan = " << nHaloSpan << endl;
  cout << rp() + "vFullPartOffset[0]   = " << vFullPartOffset.front() << endl;
#endif 
  
  long nDesPartCore = static_cast<long>(static_cast<double>(nPartTot) / numtasks());
  long nCurrDesPartCore = nDesPartCore;
  
#ifdef VERBOSE
  cout << rp() + "There are " << nPartTot << " particles in total." << endl;
  cout << rp() + "Aiming for " << nDesPartCore << " particles per core" << endl;  
#endif  

  // Initialise the FIRST element (easy):
  vTPOlist.front() = vFullPartOffset.front();
  vTHOlist.front() = nHaloIni;
  
#ifdef DEBUG
      std::cout << rp() + "Sizes of vFullPartOffset / Length: " << vFullPartOffset.size() << ", " << vFullPartLength.size() << std::endl;
      std::cout << rp() + "nHaloSpan = " << std::endl;
#endif
      
  
  // Now we need to loop through the halo list...
  for (int ii=0;ii<nHaloSpan;ii++)
    {

      long Length = vFullPartOffset.at(ii+1)-vFullPartOffset.at(ii);
      currcounter += Length;

      if (currcounter >= nCurrDesPartCore and currtask < (numtasks()-1))
	{
	  vTPOlist.at(currtask+1) = vFullPartOffset.at(ii+1);
	  vTHOlist.at(currtask+1) = nHaloIni+ii+1;
	  
	  currcounter = 0;
	  currtask++;

	  nCurrDesPartCore = static_cast<long>(static_cast<double>(vFullPartOffset.at(nHaloSpan)-vFullPartOffset.at(ii+1))/(numtasks()-currtask));
	  
	}
    }
  
  // New bit added 27-APR-2016, to cope with nSpan < parts:
  // (N.B.: This includes writing the coda)

  for (int iifill = currtask; iifill < numtasks(); iifill++) {
    vTPOlist.at(iifill+1) = vFullPartOffset.back();
    vTHOlist.at(iifill+1) = nHaloSpan;
  }

  // Done!
  
  function_switch("build_tpolist");
  return; 

}


