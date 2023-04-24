// ------------ Program to parallel-trace L3200 haloes -----------------
// Started 13 OCT 2014
// Modification started 19 OCT 2014 
// Modification started 31 OCT 2014 to adapt this to the full Eagle sim
// ---------------------------------------------------------------------

// General modification-switch file:
#include "Config.h" 

// Standard libraries:
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <cstring>

// Small helpers and global vars:
#include "init.h"
#include "globals_declare.h"

// Headers for sub-files:
#include "load_ids.h"
#include "network.h"
#include "write_result.h"
#include "snipshots.h"
#include "utilities_special.h"

#include "/u/ybahe/cpplib/search_katamaran.h"
#include "/u/ybahe/cpplib/correlate_ids.h"
#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/hydrangea.h"
#include "/u/ybahe/cpplib/utilities_mpi.h"



// *********************************
// DECLARE INTERNALLY USED FUNCTIONS
// *********************************

int load_input_parameters(int argc, 
			  char *argv[], 
			  std::string &sParamFile);

int read_parameter_file(std::string sParamFile, 
			RunParams &runParams);

void print_config_flags();


// **********************
// PROTEA's MAIN FUNCTION
// **********************

int main(int argc, char *argv[])
{

  using namespace std;
  
  // ---- MPI initialisation ----
  
  MPI_Init(&argc, &argv);
  rp("[" + to_string(mpirank()) + "]: ");

  rp("", 0);
  
  // ---- Welcome messages -------
  
  if (mpirank() == 0) {
     
    cout << endl;
    cout << endl;
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;
    cout << endl;
    cout << "Welcome to PROTEA-HYDRANGEA (" << numtasks() << " cores)!" << endl;
    cout << endl;
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;
    cout << endl;
  }


  // ============================================================
  // First: Deal with input parameters (snapshot and halo ranges)
  // ============================================================

  double dDummy = ElTime(), dProgStartTime = GetTime();
  
  rt(); /* Initialize cumulative run time counter */
    
  string sCodaOut = "halo_histories_TEST";

  // The following are the standard values, but can be overridden later
  string sParamFile = "protea.param";  

  // ----- Read in the run parameters from file ---

  if (mpirank() == 0) {
    load_input_parameters(argc, argv, sParamFile);
    int nFileParams = read_parameter_file(sParamFile, Params);  
  }
  broadcast_struct<RunParams> (Params);
  
  sCodaOut = string(Params.cCodaOut);
  g_sFileNameOut  = string(Params.cOutputDir) + "/" + sCodaOut + ".hdf5"; 

  string sFileNameOutLinks = string(Params.cOutputDir) + string(Params.cCodaOut) + "_Links";
  

  if (sizeof(int) != 4 || sizeof(long) != 8) {
    cout << "Sizes are not standard on this machine. Exiting..." << endl;
    exit(74); }
  
  if (mpirank() == 0)
    print_config_flags();

  // ------ Set up snapshot list --------

  g_nSnapshots = Params.nSnapFin - Params.nSnapIni + 1; // we're going FORWARDS here!

  std::vector<int> vSnapshotList(g_nSnapshots);
  
  if (mpirank() == 0) {
    std::vector<int> vFullSnapList = read_snapshot_list(Params.cSnapshotListName, Params.nSnapshotListEntries);
    for (int iisnap = 0; iisnap < g_nSnapshots; iisnap++)
      vSnapshotList.at(iisnap) = vFullSnapList.at(Params.nSnapIni+iisnap);
  }

  // Update snapshot list on all other tasks
  broadcast_vector<int>(vSnapshotList, 0);
  
  
  if (mpirank() == 0) {
    cout << rp() + "Set up snapshot list with " << vSnapshotList.size() << " elements. " << rt() << endl;
    
#ifdef VERBOSE
    cout << rp() + "Now declaring a whole lot of persistent variables..." << endl;
#endif
  }
  
  // ------------------------------------------------------------
  // ------------------ Now begin the main loop -----------------
  // ------------------ (over different snapshots) --------------
  // ------------------------------------------------------------


  // Declare vectors to be used across snapshots here, so they 
  // are not re-set at the beginning of each snapshot iteration!
  
  vector<long> vTPOlistB, vTPOlistC;
  vector<long> vFullPartOffsetB, vFullPartOffsetC;
  vector<long> vFullPartLengthB, vFullPartLengthC;
  vector<unsigned long> vThisTaskIDsB, vThisTaskIDsC;

#ifdef DENSE_IDS
  vector<long> vMedIDLocListB, vMedIDLocListC;
  vector<int> vMedOrigTaskListB, vMedOrigTaskListC;
#else 
  vector<long> vSortedIndicesB, vSortedIndicesC;
  vector<int> vDummy;
#endif
  
  // vTPOlist -->  Distribution of PARTICLES across TASKS
  // vTHOlist -->  Distribution of HALOES across TASKS
  // vFPOlist -->  Distribution of PARTICLES across FILES
  // vFHOlist -->  Distribution of HALOES across FILES

  vector<int> vFHOlistB, vFHOlistC;
  vector<long> vFPOlistB, vFPOlistC;
  vector<int> vTHOlistB, vTHOlistC;
  
  // Initialise link vectors:
  vector<int> vLinkSenderBC, vLinkSenderAC;
  vector<int> vLinkReceiverBC, vLinkReceiverAC;
  vector<int> vLinkRankBC, vLinkRankAC;
  vector<int> vLinkNumPartBC, vLinkNumPartAC;
  vector<float> vLinkSenderFractionBC, vLinkSenderFractionAC;

  vector<long> vTaskMatchIndsBC; // Index in target list of particles in original list
  vector<int> vTaskMatchHaloesBC; // For corresponding haloes
  vector<long> vTaskMatchOrderBC; // for order in remote haloes

  Result SelRecOut;
  Link LinksBC, LinksAC;

#ifdef VERBOSE
  if (mpirank() == 0) 
    cout << rp() + "  ...done! Begin main loop now. " << rt() << endl;
#endif

  long nParts = 0;
#ifdef FIXED_ID_RANGE
  nParts = static_cast<long>(FIXED_ID_RANGE);
  if (mpirank() == 0)
    cout << rp() + "******* WARNING: Using fixed ID range, nParts=" << nParts << "*******" << std::endl;
#endif

  for (int tt = 0; tt<(g_nSnapshots-1); tt++) { // don't need to iterate from last snap!

    
    vector<long> vTPOlistA, vFullPartOffsetA, vFullPartLengthA, vFPOlistA;
    vector<unsigned long> vThisTaskIDsA;

#ifdef DENSE_IDS
  vector<long> vMedIDLocListA;
  vector<int> vMedOrigTaskListA;
#else
    vector<long> vSortedIndicesA;
#endif

    vector<int> vTaskMatchHaloesAB; 
    vector<long> vTaskMatchOrderAB;
    vector<int> vFHOlistA, vTHOlistA;
    vector<int> vLinkSenderAB, vLinkReceiverAB, vLinkRankAB, vLinkNumPartAB;
    vector<float> vLinkSenderFractionAB;
    Link LinksAB;

    double dMatchFrac = 0;
    
    g_ttSnap = tt; // make global copy of this, for ease of access from functions.

    if (mpirank() == 0) {
      cout << endl;
      cout << rp() + "*** Start time step " << tt << " / " << g_nSnapshots-1 << " *** " << rt() << endl;
      cout << endl;
    }
    
    double dItStartTime = GetTime();
    int flag_success = -1;
    
    g_nSnapA = vSnapshotList.at(tt);
    g_nSnapB = vSnapshotList.at(tt+1);
    g_sSubDirA = eagle_filename(string(Params.cSimBaseDir), g_nSnapA, 1, mpirank());
    g_sSubDirB = eagle_filename(string(Params.cSimBaseDir), g_nSnapB, 1, mpirank());
    

    std::string sSnapDirA;

#ifndef FIXED_ID_RANGE
    sSnapDirA = eagle_filename(string(Params.cSimBaseDir), g_nSnapA, 0, mpirank());
#endif

    if (tt < (g_nSnapshots-2)) {
      g_nSnapC = vSnapshotList.at(tt+2);
      g_sSubDirC = eagle_filename(string(Params.cSimBaseDir), g_nSnapC, 1, mpirank());
    }

    if (mpirank() == 0)
      std::cout << rp() + "SubDirA = '" << g_sSubDirA << "'" << std::endl;
    
    int nFirstHaloA = -1, nLastHaloA = -1;
    int nFirstHaloB = -1, nLastHaloB = -1;
    int nFirstHaloC = -1, nLastHaloC = -1;

    double dStartTime = GetTime();
    
    
    /*** 
	 In this version, make_tpolist() is a parallel function that ALL tasks participate in
	 It also includes distribution of the resulting lists to all tasks
    ***/
    
    
    // Build TPO lists
    if (mpirank() == 0)
      cout << rp() + "Beginning to make TPOlists etc. " << rt() << endl;
    
    // Some special treatment for early snaps:
    if (tt == 0) {
      if (mpirank() == 0)
	cout << rp() + "Timestep == 0: Need to build list A. " << rt() << endl;
      
      

      vTPOlistA = make_tpolist_parallel(g_sSubDirA, nFirstHaloA, nLastHaloA, vTHOlistA, vFullPartOffsetA, vFullPartLengthA);
      g_nHaloesA = nLastHaloA - nFirstHaloA + 1;

      if (mpirank() == 0)
	cout << rp() + "Timestep == 0: Need to build list B. " << rt() << endl;
      
      vTPOlistB = make_tpolist_parallel(g_sSubDirB, nFirstHaloB, nLastHaloB, vTHOlistB, vFullPartOffsetB, vFullPartLengthB);
      g_nHaloesB = nLastHaloB - nFirstHaloB + 1;
      
      if (mpirank() == 0)
	cout << rp() + "Finished building list B. " << rt() << endl;
      
    }    else {  // ends section only for first step
      
      if (mpirank() == 0)
	cout << rp() + "Copying existing lists B-->A and C-->B..." << endl;
      
      vTPOlistA.swap(vTPOlistB);
      nFirstHaloA = nFirstHaloB;
      nLastHaloA = nLastHaloB;
      vTHOlistA.swap(vTHOlistB);
      vFullPartOffsetA.swap(vFullPartOffsetB);
      vFullPartLengthA.swap(vFullPartLengthB);
      g_nHaloesA = g_nHaloesB;
  
      vTPOlistB.swap(vTPOlistC);
      nFirstHaloB = nFirstHaloC;
      nLastHaloB = nLastHaloC;
      vTHOlistB.swap(vTHOlistC);
      vFullPartOffsetB.swap(vFullPartOffsetC);
      vFullPartLengthB.swap(vFullPartLengthC);
      g_nHaloesB = g_nHaloesC;

      vThisTaskIDsA.swap(vThisTaskIDsB);
      vThisTaskIDsB.swap(vThisTaskIDsC);

      vTaskMatchHaloesAB.swap(vTaskMatchHaloesBC); // AB is now filled, BC empty
      vTaskMatchOrderAB.swap(vTaskMatchOrderBC);

#ifdef DENSE_IDS
      vMedIDLocListA.swap(vMedIDLocListB);
      vMedIDLocListB.swap(vMedIDLocListC);
      vMedOrigTaskListA.swap(vMedOrigTaskListB);
      vMedOrigTaskListB.swap(vMedOrigTaskListC);
      
#else
      vSortedIndicesA.swap(vSortedIndicesB);
      vSortedIndicesB.swap(vSortedIndicesC);
#endif

      if (mpirank() == 0)
	cout << rp() + "...done. " << rt() << endl;
    }
    

    // List C always needs to be made from scratch:
    if (tt < (g_nSnapshots-2)) {

      if (mpirank() == 0)
	cout << rp() + "Making list C (snapshot " << g_nSnapC << ")... " << rt() << endl;
      
      vTPOlistC = make_tpolist_parallel(g_sSubDirC, nFirstHaloC, nLastHaloC, vTHOlistC, vFullPartOffsetC, vFullPartLengthC); // ALWAYS all haloes
      g_nHaloesC = nLastHaloC-nFirstHaloC+1;
      
      if (mpirank() == 0)
	cout << rp() + "Finished Making list C (snapshot " << g_nSnapC << ")... " << rt() << endl;
     
    } else {
      vFullPartOffsetC.clear();
      vFHOlistC.clear();
      vFPOlistC.clear();
      vTPOlistC.clear();
      g_nHaloesC = 0;
    }
 

    if (mpirank() == 0) {
      cout << endl;
      cout << "************************************************" << endl;
      cout << "Finished making TPOlists (snapshots " << g_nSnapA << ", " << g_nSnapB << ", " << g_nSnapC << ")" << endl;
      cout << "************************************************" << endl; 
      cout << endl;

      cout << rp() + "Length of vFullPartOffsetC = " << vFullPartOffsetC.size() << endl;
      cout << rp() + "Length of vFullPartLengthC = " << vFullPartLengthC.size() << endl;
      
      cout << rp() + "Loading and distributing input offset/length lists took " << GetTime() - dStartTime << " sec.  " << rt() << endl;
    }


    // ----- Load actual IDs ---------

    double IDStartTime = GetTime();

    // Do IDs A first
    // This is different for first iteration, because then we have to load it from disk:

    long nDummy = -1;

    if (mpirank() == 0)
      cout << rp() + "Loading particle IDs...  " << rt() << endl;

    if (tt == 0) {
      vFPOlistA.clear(); /* to make sure it is constructed internally */
      vFPOlistB.clear(); /* to make sure it is constructed internally */

      eagleread<unsigned long>(g_sSubDirA, "IDs/ParticleID", vThisTaskIDsA, vTPOlistA, 0, vFPOlistA, 0, "Header/NumFilesPerSnapshot", "Header/Nids");
      eagleread<unsigned long>(g_sSubDirB, "IDs/ParticleID", vThisTaskIDsB, vTPOlistB, 0, vFPOlistB, 0, "Header/NumFilesPerSnapshot", "Header/Nids");
    }

    if (tt < (g_nSnapshots-2)) {
      vFPOlistC.clear();
      eagleread<unsigned long>(g_sSubDirC, "IDs/ParticleID", vThisTaskIDsC, vTPOlistC, 0, vFPOlistC, 0, "Header/NumFilesPerSnapshot", "Header/Nids");
      
    } else {
      vThisTaskIDsC.clear();
    }
    
    if (mpirank() == 0)
      cout << rp() + "Done loading IDs!  " << rt() << endl;

    
    report<long>(static_cast<long>(vThisTaskIDsC.size()), "Num of C-IDs");
    report<double>((GetTime()-IDStartTime), "Loading IDs took [sec.]");
    
    
    
    // **************************************************
    // EXTRA BIT FOR EAGLE: Select desired particle type!
    // **************************************************

    if (mpirank() == 0) 
      cout << rp() + "Beginning to reject baryons from ID lists...  " << rt() << endl;
    
    
    // Only need to do this for the A/B data at the beginning:
    if (tt == 0) {
      reject_baryons(vThisTaskIDsA, vFullPartOffsetA, vFullPartLengthA, vTPOlistA, vTHOlistA);	
      reject_baryons(vThisTaskIDsB, vFullPartOffsetB, vFullPartLengthB, vTPOlistB, vTHOlistB);
      
#ifndef DENSE_IDS
      vSortedIndicesA = sort_indices<unsigned long>(vThisTaskIDsA);
      vSortedIndicesB = sort_indices<unsigned long>(vThisTaskIDsB);
#endif
    } 
    

    // ... and don't need to do it for C at the end!
    if (tt < (g_nSnapshots-2)) {
      reject_baryons(vThisTaskIDsC, vFullPartOffsetC, vFullPartLengthC, vTPOlistC, vTHOlistC);
#ifndef DENSE_IDS
      vSortedIndicesC = sort_indices<unsigned long>(vThisTaskIDsC);
#endif
    }

    if (mpirank() == 0) 
      cout << rp() + "Finished rejecting baryons for iteration " << tt << " (" << g_nSnapA << " --> " << g_nSnapB << " --> " << g_nSnapC << ")  " << rt() << endl;
    
    
    
    // *************
    // Now match IDs
    // *************
  
    vector<long> vTaskMatchIndsAB; // Index in target list of particles in original list
    
    if (mpirank() == 0)
      cout << rp() + "Matching IDs across snapshots...  " << rt() << endl;

    // First A --> B (only has to be done at first step!)
    if (tt == 0) {
      long PartOffsetB = vTPOlistB.at(mpirank()); // Offset of this task's first particle in full list
      long PartOffsetA = vTPOlistA.at(mpirank());

      // Now perform the actual particle matching!

#ifndef DENSE_IDS
      long nTaskMatches = search_katamaran(vThisTaskIDsA, vSortedIndicesA, vThisTaskIDsB, vSortedIndicesB, vTaskMatchIndsAB, vDummy, PartOffsetB, 0, 0);
#else

      // *** This section is FOR HYDRANGEA ONLY -- it relies on the ability
      //     to make reverse-index lists from the IDs, which will almost
      //     certainly fail with the original EAGLE run (IDs are too big).
      //
      // PART I: 'build_mediator_list()'
      //         This constructs a 'mediator' list
      //
      // *** Please note: ***
      // build_mediator_list() is a COLLECTIVE function. 
      // Internally, it exchanges data between tasks to build the output vectors 
      // for each task.
      //
      // Similar to search_katamaran(), we need to provide it with the offset of the 
      // chunk of the full ID list stored on this task (PartOffsetA/B) - it needs this information
      // to calculate the position in the full (hypothetical) ID list which is stored
      // in vMedIDLocListA/B.

      build_mediator_list<long> (vThisTaskIDsA, vMedIDLocListA, vMedOrigTaskListA, sSnapDirA, std::string(Params.cRunType), PartOffsetA, nParts);
      build_mediator_list<long> (vThisTaskIDsB, vMedIDLocListB, vMedOrigTaskListB, sSnapDirA, std::string(Params.cRunType), PartOffsetB, nParts);

      
      vTaskMatchIndsAB.clear();
      vTaskMatchIndsAB.resize(vThisTaskIDsA.size(),-1);
      long nTaskMatches = correlate_ids(vMedIDLocListA, vMedOrigTaskListA, vTPOlistA, vMedIDLocListB, vTaskMatchIndsAB);
      
#endif
      
      
      dMatchFrac = static_cast<double>(nTaskMatches)/static_cast<double>(vThisTaskIDsA.size())*100;

      if (mpirank() == 0)
	cout << rp() + "Successful ID matches = " << nTaskMatches << " (out of " << vThisTaskIDsA.size() << ", " << static_cast<double>(nTaskMatches)/static_cast<double>(vThisTaskIDsA.size())*100 <<"%)" << endl;

      report<double>(dMatchFrac, "Fraction of matched particles (AB):");
      vTaskMatchHaloesAB = index_to_halo(vTaskMatchIndsAB, vFullPartOffsetB, vTaskMatchOrderAB);

    } // Ends section ONLY for first snapshot
    
    
    // *** AND AGAIN ***, this time A --> C, and B --> C (!!)

    // Declare AC vectors. NOTE: BC is declared outside of snapshot loop!!
    vector<long> vTaskMatchIndsAC; // Index in target list of particles in original list
    vector<int> vTaskMatchHaloesAC; // For corresponding haloes
    vector<long> vTaskMatchOrderAC; // For ordering in remote halo

    if (tt < (g_nSnapshots-2)) {
      long PartOffsetC = vTPOlistC.at(mpirank()); // Offset of this task's first particle in full list

#ifndef DENSE_IDS
      long nTaskMatchesAC = search_katamaran(vThisTaskIDsA, vSortedIndicesA, vThisTaskIDsC, vSortedIndicesC, vTaskMatchIndsAC, vDummy, PartOffsetC, 0, 0);
#else
      build_mediator_list<long> (vThisTaskIDsC, vMedIDLocListC, vMedOrigTaskListC, sSnapDirA, std::string(Params.cRunType), PartOffsetC, nParts);

      vTaskMatchIndsAC.clear();
      vTaskMatchIndsAC.resize(vThisTaskIDsA.size());
      long nTaskMatchesAC = correlate_ids(vMedIDLocListA, vMedOrigTaskListA, vTPOlistA, vMedIDLocListC, vTaskMatchIndsAC);
#endif 

      dMatchFrac = static_cast<double>(nTaskMatchesAC)/static_cast<double>(vThisTaskIDsA.size())*100;
      if (mpirank() == 0)
	cout << rp() + "Particle ID [AC] matches = " << nTaskMatchesAC << " (out of " << vThisTaskIDsA.size() << ", " << dMatchFrac << "%)" << endl;
      report<double>(dMatchFrac, "Fraction of matched particles (AC):");
      
      vTaskMatchHaloesAC = index_to_halo(vTaskMatchIndsAC, vFullPartOffsetC, vTaskMatchOrderAC);
      

      // ---------- And now B-->C -----------
      
#ifdef DENSE_IDS
      
      vTaskMatchIndsBC.clear();
      vTaskMatchIndsBC.resize(vThisTaskIDsB.size());
      long nTaskMatchesBC = correlate_ids(vMedIDLocListB, vMedOrigTaskListB, vTPOlistB, vMedIDLocListC, vTaskMatchIndsBC);
#else      
      long nTaskMatchesBC = search_katamaran(vThisTaskIDsB, vSortedIndicesB, vThisTaskIDsC, vSortedIndicesC, vTaskMatchIndsBC, vDummy, PartOffsetC, 0, 0);
#endif

      dMatchFrac = static_cast<double>(nTaskMatchesBC)/static_cast<double>(vThisTaskIDsB.size())*100;
      if (mpirank() == 0)
	cout << rp() + "ParticleID [BC] matches = " << nTaskMatchesBC << " (out of " << vThisTaskIDsA.size() << ", " << dMatchFrac << "%)" << endl;

      report<double>(dMatchFrac, "Fraction of matched particles (BC):");
      
      vTaskMatchHaloesBC = index_to_halo(vTaskMatchIndsBC, vFullPartOffsetC, vTaskMatchOrderBC);
    
    }

    if (mpirank() == 0) {
      cout << endl;
      cout << "******************************************************************" << endl;
      cout << "Finished with ID matching (snapshots " << g_nSnapA << ", " << g_nSnapB << ", " << g_nSnapC << ")  " << rt() << endl;
      cout << "******************************************************************" << endl; 
      cout << endl;
    }

    // ******** After this point, there should be NO MORE DIFFERENCE 
    // between the 'EAGLE' and 'HYDRANGEA' versions of Protea... ***
    // (i.e. no more occurrence of 'vSortedIndicesXXX' or 'DENSE_IDS')

    
    // =========================================
    // Now we need to use this to link haloes...
    // =========================================

    
    // First step: BUILD LINKS on each task, from particle matching output
    // This function also sends them to master when done.
    
    if (mpirank() == 0)
      cout << rp() + "Building links from ID information... " << rt() << endl;

    if (tt == 0) {
      build_links(vTaskMatchHaloesAB, vFullPartOffsetA, vTHOlistA, vLinkSenderAB, vLinkReceiverAB, vLinkRankAB, vLinkNumPartAB, vLinkSenderFractionAB);
    }

    if (tt < (g_nSnapshots-2)) {    
      build_links(vTaskMatchHaloesBC, vFullPartOffsetB, vTHOlistB, vLinkSenderBC, vLinkReceiverBC, vLinkRankBC, vLinkNumPartBC, vLinkSenderFractionBC, 0);
      build_links(vTaskMatchHaloesAC, vFullPartOffsetA, vTHOlistA, vLinkSenderAC, vLinkReceiverAC, vLinkRankAC, vLinkNumPartAC, vLinkSenderFractionAC, 1);
    }

    if (mpirank() == 0)
      cout << rp() + "Finished building links!" << rt() << endl;
	

    // Need to declare some variables here that we will use outside the next block:
    Result SelSendB;
    std::vector<int> vMasterHaloLocListA;
    int nLengthA;

    vector<int> vMergeTargets;


    // The next bit is only for master (who now has all links anyway):
    if (mpirank() == 0) {
      
      // Build "proper" struct-vector for ease of life:

      if (tt == 0) {
	LinksAB = build_link_struct(vLinkSenderAB, vLinkReceiverAB, vLinkRankAB, vLinkNumPartAB, vLinkSenderFractionAB, g_nHaloesA,0);
 	invert_links(LinksAB, vFullPartOffsetB);
      } else {
	LinksAB.Sender.swap(LinksBC.Sender);
	LinksAB.Receiver.swap(LinksBC.Receiver);
	LinksAB.Rank.swap(LinksBC.Rank);
	LinksAB.Choice.swap(LinksBC.Choice);
	LinksAB.SenderFraction.swap(LinksBC.SenderFraction);
	LinksAB.ReceiverFraction.swap(LinksBC.ReceiverFraction);
	LinksAB.NumPart.swap(LinksBC.NumPart);
	LinksAB.SortedByRecv.swap(LinksBC.SortedByRecv);
	LinksAB.SenderOffset.swap(LinksBC.SenderOffset);
	LinksAB.ReceiverOffset.swap(LinksBC.ReceiverOffset);
      }

      if (tt < (g_nSnapshots-2)) {
	LinksBC = build_link_struct(vLinkSenderBC, vLinkReceiverBC, vLinkRankBC, vLinkNumPartBC, vLinkSenderFractionBC, g_nHaloesB,0);
	invert_links(LinksBC, vFullPartOffsetC);
	
	LinksAC = build_link_struct(vLinkSenderAC, vLinkReceiverAC, vLinkRankAC, vLinkNumPartAC, vLinkSenderFractionAC, g_nHaloesA,0);
      }


#ifdef DEBUG
      output_links(LinksAB, sFileNameOutLinks+".AB");
      output_links(LinksBC, sFileNameOutLinks+".BC");
      output_links(LinksAC, sFileNameOutLinks+".AC");
#endif

      
      initialize_result(SelSendB, g_nHaloesA);
      if (tt < (g_nSnapshots-2)) {

	// Call high-level function to do the whole link-network analysis!
	evaluate_link_network(LinksAB, LinksBC, LinksAC, SelSendB, vFullPartOffsetC, SelRecOut);
      } else {

	// *Slightly* easier for last snapshot...
	cout << rp() + "Executing simplified select_links() for final snapshot..." << endl;

	//	Result SSTemp;
	//initialize_result(SSTemp, g_nHaloesB);
	select_links(LinksAB, SelSendB, SelRecOut, 1);

      } 

      
      // Additional bit added (or at least begun) 06 FEB 2015: 
      // Rudimentary merger identification

      /* N.B.: At the moment, a galaxy in A which could not be traced to a subhalo
	 in either B or C is now assumed to have merged with the target of its primary
	 (AB) shortlink.

	 It may be possible to do more fancy things (i.e., allow mergers onto AC targets),
	 but the method here should be sufficient for now. */
      

      identify_mergers(LinksAB, SelSendB, vMergeTargets);
      
    } // ends section only for root

    // *** We need to go back to collective operation for flagging dodgy subhaloes ***
    // This bit was begun 12 JUL 2016

    std::vector<int> vFlagList = flag_contaminated_subhaloes(vTaskMatchHaloesAB, vTaskMatchOrderAB, vTHOlistA, vFullPartOffsetA, SelSendB, LinksAB);
    
    if (mpirank() == 0) {

      // Write Output (!!!!)
      // Now all we really need is SelSendB!
      
      cout << endl;
      cout << "******************************************************************" << endl;
      cout << "Finished linking haloes (snapshots " << g_nSnapA << ", " << g_nSnapB << ", " << g_nSnapC << ")" << endl;
      cout << "******************************************************************" << endl; 
      cout << endl;
      
      cout << rp() + "Now writing output... " << rt() << endl; 

      
      // Finally, write result to HDF5 file.
      
#ifdef DEBUG
      std::cout << rp() + "SSB.Match[0] = " << SelSendB.Match.at(0) << endl;
#endif
      
      write_result(SelSendB, vMergeTargets, vTHOlistB, vMasterHaloLocListA, nLengthA, vFlagList); 

      if (SelSendB.Match.size() > 0) {
	cout << rp() + "Result for the first halo: " << endl;
	print_result(SelSendB, 0);
      }
      
    } // ends master-only section here.
    
    
#ifdef EXTRACT_IDS_FOR_SNIPSHOTS

    // Function to write out IDs to trace galaxies in snipshots
    // NB: This is a collective function, so has to be executed outside the 
    //     'master-only' section we have just left.
    
    extract_ids_for_snipshots(SelSendB, vTHOlistA, vThisTaskIDsA, vFullPartOffsetA, 
			      vTaskMatchHaloesAB, vTaskMatchHaloesAC,
			      vTaskMatchOrderAB, vTaskMatchOrderAC,
			      vMasterHaloLocListA, nLengthA,0);

#ifdef EXTRACT_IDS_FOR_SNIPSHOTS_REVERSE
    extract_ids_for_snipshots(SelSendB, vTHOlistA, vThisTaskIDsA, vFullPartOffsetA, 
			      vTaskMatchHaloesAB, vTaskMatchHaloesAC,
			      vTaskMatchOrderAB, vTaskMatchOrderAC,
			      vMasterHaloLocListA, nLengthA, 1);
#endif

#endif

    if (mpirank() == 0) {
      cout << endl;
      cout << "******************************************************************" << endl;
      cout << "Time step " << tt << " took " <<  GetTime() - dItStartTime << " sec. " << rt() << endl;
      cout << "******************************************************************" << endl; 
      cout << endl;
    } 
    
    MPI_Barrier(MPI_COMM_WORLD);
    
  } // ends time iteration loop
  
  if (mpirank() == 0) {
    cout << endl;
    cout << rp() + "PROTEA took " << GetTime() - dProgStartTime << " sec. to trace " << g_nGalaxies << " haloes over " << vSnapshotList.size() << " snapshots." << endl;
  }
  
  MPI_Finalize();
  
  return 0;
}
  

// ************************************
// Implementation of internal functions
// ************************************

int load_input_parameters(int argc, 
			  char* argv[], 
			  std::string &sParamFile) {

  
  using namespace std;

  if (argc > 1) {
    sParamFile = argv[1];
    std::cout << rp() + "Using supplied parameter file '" << sParamFile << "'" << std::endl;
  }

  return argc-1;
}	




int read_parameter_file(std::string sParamFile, RunParams &runParams) {

  using namespace std;

  int nParams = 0;

  ifstream file;
  char tab2[1024];
  strcpy(tab2, sParamFile.c_str());
  file.open(tab2);
  
  std::string line;
  string param_name, param_val;

  cout << rp() + "Reading parameter file '" << sParamFile << "':" << endl;

  int nCount = 0;
  while (std::getline(file, line))
    {
      nCount++;
#ifdef VERBOSE
      cout << "Line " << nCount << ": " << line << endl;
#endif

      if (line.size() == 0)
	continue;

      if (line.at(0) == '#') {
#ifdef VERBOSE
	cout << "Skipping line " << nCount << endl; 
#endif
	continue; 
      }

      std::istringstream iss(line);
      if (!(iss >> param_name >> param_val)) {
	std::cout << "Erroneous format of line " << nCount << "!" << endl;
	exit(42);
      }

#ifdef VERBOSE
      cout << param_name << " is " << param_val << endl;
#endif
      nParams++;

      if(param_name == "StartSnap") 
	runParams.nSnapIni = string_to_num<int>(param_val);
      else if (param_name == "EndSnap")
	runParams.nSnapFin = string_to_num<int>(param_val);
      else if (param_name == "SnapshotListEntries")
	runParams.nSnapshotListEntries = string_to_num<int>(param_val);
      else if (param_name == "OutputCoda")
	strcpy(runParams.cCodaOut, param_val.c_str());
      else if (param_name == "SimBaseDir")
	strcpy(runParams.cSimBaseDir, param_val.c_str());
      else if (param_name == "OutputDir")
	strcpy(runParams.cOutputDir, param_val.c_str());
      else if (param_name == "SnapshotListName")
	strcpy(runParams.cSnapshotListName, param_val.c_str());
      else if (param_name == "RunType")
	strncpy(runParams.cRunType, param_val.c_str(), 19);
      else if (param_name == "MinFractionLowChoice")
	runParams.dMinFracLowChoice = string_to_num<double>(param_val);

      else if (param_name == "MaxTracers")
	runParams.nMaxTracers = string_to_num<int>(param_val);

#ifdef LINK_MIN_RECVFRAC
      else if (param_name == "MinReceiverFraction")
	runParams.dMinRecvFrac = string_to_num<double>(param_val);
#endif
#ifdef LINK_MIN_SENDFRAC
      else if (param_name == "MinSenderFraction")
	runParams.dMinSendFrac = string_to_num<double>(param_val);
#endif

      else
	cout << "Unknown parameter " << param_name << endl;
      
    } // ends loop to read
  
  cout << rp() + "   ...read " << nParams << " parameters." << endl;
  
  if (std::string(runParams.cOutputDir).compare("SimBaseDir") == 0)
    strcpy(runParams.cOutputDir, runParams.cSimBaseDir);
  
  cout << endl;
  cout << "---------------------------" << endl;

  cout << "  Settings: " << endl;
  cout << "    StartSnap            = " << runParams.nSnapIni << endl;
  cout << "    EndSnap              = " << runParams.nSnapFin << endl;
  cout << "    SnapshotListEntries  = " << runParams.nSnapshotListEntries << endl;
  cout << "    OutputCoda           = " << runParams.cCodaOut << endl;
  cout << "    SimBaseDir           = " << runParams.cSimBaseDir << endl;
  cout << "    OutputDir            = " << runParams.cOutputDir << endl;
  cout << "    SnapshotListName     = " << runParams.cSnapshotListName << endl;
  cout << "    RunType              = " << runParams.cRunType << endl;
  cout << "    MinFractionLowChoice = " << runParams.dMinFracLowChoice << endl;
#ifdef LINK_MIN_RECVFRAC
  cout << "    MinReceiverFraction  = " << runParams.dMinRecvFrac << endl;
#endif
#ifdef LINK_MIN_SENDFRAC
  cout << "    MinSenderFraction    = " << runParams.dMinSendFrac << endl;
#endif
  cout << "    MaxTracers           = " << runParams.nMaxTracers << endl;

  cout << "---------------------------" << endl;
  cout << endl;


  return nParams;
  
}


void print_config_flags() {
  
  std::cout << std::endl;
  std::cout << "Code was compiled with following options:" << std::endl;
  std::cout << "-----------------------------------------" << std::endl;

#ifdef VERBOSE_SNIPSHOTS
  std::cout << "VERBOSE_SNIPSHOTS" << std::endl;
#endif

#ifdef DENSE_IDS
  std::cout << "DENSE_IDS" << std::endl;
#endif

#ifdef STRICT_LINK_LIMIT
  std::cout << "STRICT_LINK_LIMIT = " << STRICT_LINK_LIMIT << std::endl;
#endif

#ifdef EXHAUST_LINKS
  std::cout << "EXHAUST_LINKS = " << EXHAUST_LINKS << std::endl;
#endif

#ifdef LINK_MIN_RECVFRAC
  std::cout << "LINK_MIN_RECVFRAC" << std::endl;
#endif

#ifdef LINK_MAX_CHOICE
  std::cout << "LINK_MAX_CHOICE = " << LINK_MAX_CHOICE << std::endl;
#endif

#ifdef LINK_MAX_RANK
  std::cout << "LINK_MAX_RANK = " << LINK_MAX_RANK << std::endl;
#endif

#ifdef EXTRACT_IDS_FOR_SNIPSHOTS
  std::cout << "EXTRACT_IDS_FOR_SNIPSHOTS" << std::endl;
#endif 

#ifdef EXTRACT_IDS_FOR_SNIPSHOTS_REVERSE
  std::cout << "EXTRACT_IDS_FOR_SNIPSHOTS_REVERSE" << std::endl;
#endif

#ifdef NUM_SNIPIDS
  std::cout << "NUM_SNIPIDS = " << NUM_SNIPIDS << std::endl;
#endif

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << std::endl;

  return;

}
