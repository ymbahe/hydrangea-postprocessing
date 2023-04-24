#include <mpi.h>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <string>
#include <stdexcept>

#include "Config.h"
#include "globals.h"

#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/io_hdf5.h"


// ****************************************
// Declaration of internally used functions
// ****************************************

std::vector<int> find_new_haloes(const std::vector<int> &vHaloListB, 
				 int g_nHaloesB); 

void create_snapshot_datasets(int nLength);


void fill_mergelist(const std::vector<int> &vMergeTargets,   // Merge targets by SUBHALO
		    const std::vector<int> &vMergeListA,     // Previous (starter) mergelist
		    std::vector<int> &vMergeListB,           // [0] New merge list
		    const std::vector<int> &vHaloListA,      // Starter halolist
		    const std::vector<int> &vMasterHaloLocListB,  // B-rev. halo list
		    std::vector<int> &vHaloListB);


// *********************************
// EXTERNAL Function implementations
// *********************************


// ***********************************************************************************
// Function to make a list which says where in the (big) output list haloes go

void make_master_haloloclist(const std::vector<int> &vTHOlist,         // For range of SH indices 
			     const std::vector<int> &vHaloList,        // Input (ind -> SH)
			     std::vector<int> &vMasterHaloLocList) {   // Output (SH -> ind)
  
#ifdef VERBOSE
  function_switch("make_master_haloloclist");
#endif   

  using namespace std;
  
  int nHaloSpan = vTHOlist.back()-vTHOlist.front();
  int nHaloIni = vTHOlist.front();
  int nHaloListLength = vHaloList.size();

  vMasterHaloLocList.clear();
  vMasterHaloLocList.resize(nHaloSpan,-77);

  for (int ii=0; ii<nHaloListLength; ii++) {
    if (vHaloList.at(ii) >= 0)
      vMasterHaloLocList.at(vHaloList.at(ii)-nHaloIni) = ii;
  }
  
#ifdef VERBOSE
  function_switch("make_master_haloloclist");
#endif 

  return;
  
}



void write_result(const Result &SelSendB,                       // The result to be written
		  const std::vector<int> &vMergeTargets,        // Merge list by subhaloes
		  const std::vector<int> &vTHOlistB,                // THOlistB
		  std::vector<int> &vMasterHaloLocListA,        // [O] MHL list (required outside)
		  int &nLengthA,                    // [O] num of galaxies at A
		  const std::vector<int> &vFlagList)          { 
  
  function_switch("write_result");

  // First, actually write the result into (temporary) vectors...

  static std::vector<int> vHaloListC;
  static std::vector<char> vLengthListC;
  static std::vector<int> vRankListC;
  static std::vector<int> vChoiceListC;
  static std::vector<float> vSendFracListC;
  static std::vector<float> vRecFracListC;
  
  std::vector<char> vLengthListB;
  std::vector<int> vRankListB;
  std::vector<int> vChoiceListB;
  std::vector<float> vSendFracListB;
  std::vector<float> vRecFracListB;
  
  // Declare MHL-list for B static, so it can be re-used next time
  static std::vector<int> vMasterHaloLocListB;
  // std::vector<int> vMasterHaloLocListA; --> now an outside variable
  
  // Declare halo list (vHaloListB, which is now PURELY INTERNAL)
  static std::vector<int> vHaloListB;
  std::vector<int> vHaloListA;

  // Declare merge lists:
  static std::vector<int> vMergeListB;
  std::vector<int> vMergeListA;
  
  // If this is the first snapshot, need to set up vHaloListB:
  if (g_ttSnap == 0) {
    vHaloListB.resize(g_nHaloesA, IDENTFAIL);
    vLengthListB.resize(g_nHaloesA, -1);
    vRankListB.resize(g_nHaloesA, -1);
    vChoiceListB.resize(g_nHaloesA, -1);
    vSendFracListB.resize(g_nHaloesA, -1);
    vRecFracListB.resize(g_nHaloesA, -1);
    
    vMasterHaloLocListA.resize(g_nHaloesA, -1);
    vMergeListA.resize(g_nHaloesA);
    vHaloListA.resize(g_nHaloesA);

    for (int ll = 0; ll < g_nHaloesA; ll++) {
      vMasterHaloLocListA.at(ll) = ll;
      vMergeListA.at(ll) = ll;
      vHaloListA.at(ll) = ll;
    }

    
  } else {

    vHaloListA.swap(vHaloListB);
    vHaloListB.swap(vHaloListC);

    vLengthListB.swap(vLengthListC);
    vRankListB.swap(vRankListC);
    vChoiceListB.swap(vChoiceListC);
    vSendFracListB.swap(vSendFracListC);
    vRecFracListB.swap(vRecFracListC);

    vMasterHaloLocListA.swap(vMasterHaloLocListB);
    vMergeListA.swap(vMergeListB);

  }

  vHaloListC.clear();

  // Now make new C list:
  nLengthA = vHaloListA.size();  // not needed here, but in snipshot-id-identification
  int nLengthB = vHaloListB.size();
  if (g_ttSnap < (g_nSnapshots-2)) {
    
    
    vHaloListC.resize(nLengthB, IDENTFAIL);  // initialize to this, overwrite later

    vLengthListC.clear();
    vLengthListC.resize(nLengthB, -1);

    vRankListC.clear();
    vRankListC.resize(nLengthB, -1);

    vChoiceListC.clear();
    vChoiceListC.resize(nLengthB, -1);

    vSendFracListC.clear();
    vSendFracListC.resize(nLengthB, -1);

    vRecFracListC.clear();
    vRecFracListC.resize(nLengthB, -1);
  }

  // -------------------------
  // Write result into lists!
  // -------------------------

  // Loop through A-haloes for this

  std::vector<int> vLengthListA(nLengthA, -1);

  for (int ii = 0; ii < g_nHaloesA; ii++) {
    int nCurrLoc = vMasterHaloLocListA.at(ii);
    int nTraceResultCurr = SelSendB.Length.at(ii);

    switch(nTraceResultCurr) {
      
    case -1: // Tracing was UNSUCCESSFUL
      vHaloListB.at(nCurrLoc) = TRACEFAIL;
      break;

    case 1:  // Tracing resulted in 1-link (standard case)
      vHaloListB.at(nCurrLoc) = SelSendB.Match.at(ii);
      vLengthListB.at(nCurrLoc) = 1;
      vRankListB.at(nCurrLoc) = SelSendB.Rank.at(ii);
      vChoiceListB.at(nCurrLoc) = SelSendB.Choice.at(ii);
      vSendFracListB.at(nCurrLoc) = SelSendB.SenderFraction.at(ii);
      vRecFracListB.at(nCurrLoc) = SelSendB.ReceiverFraction.at(ii);
      vLengthListA.at(nCurrLoc) = 1;
      break;

    case 2:  // Tracing resulted in 2-link
      vHaloListB.at(nCurrLoc) = BYPASSED;
      vHaloListC.at(nCurrLoc) = SelSendB.Match.at(ii);
      vLengthListC.at(nCurrLoc) = 2;
      vRankListC.at(nCurrLoc) = SelSendB.Rank.at(ii);
      vChoiceListC.at(nCurrLoc) = SelSendB.Choice.at(ii);
      vSendFracListC.at(nCurrLoc) = SelSendB.SenderFraction.at(ii);
      vRecFracListC.at(nCurrLoc) = SelSendB.ReceiverFraction.at(ii);
      vLengthListA.at(nCurrLoc) = 2;
      break;

    default:
      std::cout << "Unexpected value of Length in SelSendB[" << ii << "] = " << nTraceResultCurr << std::endl;
      std::cout << "Error occurred in function write_result(), snap-iter " << g_ttSnap << std::endl;
      exit(42);
      break;
    }
  } // ends loop through A-haloes

  // The following bit has CHANGED on 26 Apr 2016
  // We now *FIRST* append the new haloes, and *THEN* fill in the merge list
  // This is to allow for the possibility of a merger onto a newly formed halo,
  // which may happen if not all links are exhausted (big waste of time)
  // and also to prevent unphysical 'take-overs' of newly formed galaxies
  // by some small thing merging onto it.
    

  // Next thing is to find NEW subhaloes in snap B:
  std::vector<int> vNewHaloesB = find_new_haloes(vHaloListB, g_nHaloesB);
  int nNumNewHaloes = vNewHaloesB.size();
  
  // Append vHaloListB accordingly, and also list C:
  vHaloListB.resize(nLengthB + nNumNewHaloes, IDENTFAIL);
  vLengthListB.resize(nLengthB + nNumNewHaloes, -1);
  vRankListB.resize(nLengthB + nNumNewHaloes, -1);
  vChoiceListB.resize(nLengthB + nNumNewHaloes, -1);
  vSendFracListB.resize(nLengthB + nNumNewHaloes, -1);
  vRecFracListB.resize(nLengthB + nNumNewHaloes, -1);
  vMergeListB.resize(nLengthB + nNumNewHaloes,-1);

  if (g_ttSnap < (g_nSnapshots - 2)) {
    vHaloListC.resize(nLengthB + nNumNewHaloes, IDENTFAIL);
  
    vLengthListC.resize(nLengthB + nNumNewHaloes, -1);
    vRankListC.resize(nLengthB + nNumNewHaloes, -1);
    vChoiceListC.resize(nLengthB + nNumNewHaloes, -1);
    vSendFracListC.resize(nLengthB + nNumNewHaloes, -1);
    vRecFracListC.resize(nLengthB + nNumNewHaloes, -1);
    
  }
  
  // Fill in B-list with NEW haloes (C --> only next time)
  for (int ii = 0; ii < nNumNewHaloes; ii++) {
    vHaloListB.at(nLengthB+ii) = vNewHaloesB.at(ii);
    vMergeListB.at(nLengthB+ii) = nLengthB+ii;
  }
  
  // Store length of B halo list in a global variable:
  g_nGalaxies = vHaloListB.size();

  // Need form the B master-halo-loclist, including newly added subhaloes (!)
  // This is needed for fill_mergelist below, and for next iteration.
  make_master_haloloclist(vTHOlistB, vHaloListB, vMasterHaloLocListB);
  
  // Fill in the merge list:
  fill_mergelist(vMergeTargets, vMergeListA, vMergeListB, vHaloListA, vMasterHaloLocListB, vHaloListB);
  
  // Finally, incorporate 'Flag' info about corrupted subhaloes
  // (this bit was added 15 July 2016)

  int nNumGalA = vHaloListA.size();

  std::vector<int> vHaloListBMasked(vHaloListB.size());
  for (int ii = 0; ii < vHaloListB.size(); ii++) {

    vHaloListBMasked.at(ii) = vHaloListB.at(ii);

    if (ii >= nNumGalA)
      continue;  // new condition: do not copy flag if this is a new galaxy

    if (vHaloListA.at(ii) < 0)
      continue;  // new condition: do not copy flag if galaxy is bypassed in A 

    if (vHaloListB.at(ii) < 0)
      continue;

    if (vFlagList.at(vHaloListB.at(ii)) > 0)
      vHaloListBMasked.at(ii) = -vFlagList.at(vHaloListB.at(ii));

  }
  
  

  
  // =========== FINISHED WRITING VECTORS ========================
  // ----- Now we need to write these data to HDF5... ------------
  // ----- BUT: only the data for list "B" need to be written! ---
  // ------ (small mercy...) -------------------------------------
  // ----- Actually, as of 20-APR-2016, we also need to write LengthA...
  // ===================================================================
  
  static int nNumCalledBefore = 0; // counter to check whether file has been set up yet

  if (nNumCalledBefore == 0) {
    hdf5_create_file(std::string(g_sFileNameOut));
      
    std::string sSnapGroup = "Snapshot_" + to_string(g_nSnapA, 3);
    hdf5_create_group(g_sFileNameOut, sSnapGroup);
  
    std::vector<int> vDims(1);
    vDims.front() = g_nHaloesA;
    
    hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/SubHaloIndex", vDims, "int", "Subhalo index of galaxy in current snapshot", -100);
    
    write_hdf5_data(g_sFileNameOut, sSnapGroup + "/SubHaloIndex", vHaloListA);
  }
 
  // New bit (20-APR-2016):
  // Write out length FOR A:
  std::string sSnapGroup = "Snapshot_" + to_string(g_nSnapA, 3);
  std::vector<int> vDims(1);
  vDims.front() = nLengthA;
    
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/LinkLengthForward", vDims, "int", "Length of link from this snapshot (-1 if not traced)", -1);
  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/LinkLengthForward", vLengthListA);
    

  // Now create the basic dataset structure
  create_snapshot_datasets(nLengthB+nNumNewHaloes);

  // And finally write the data from list B into these datasets!

  sSnapGroup = "Snapshot_" + to_string(g_nSnapB, 3);

  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/SubHaloIndex", vHaloListBMasked);
  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/SubHaloIndexAll", vHaloListB);
  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/LinkLength", vLengthListB);
  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/LinkRank", vRankListB);
  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/LinkChoice", vChoiceListB);
  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/SenderFraction", vSendFracListB);
  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/ReceiverFraction", vRecFracListB);
  write_hdf5_data(g_sFileNameOut, sSnapGroup + "/MergeList", vMergeListB);

  nNumCalledBefore++; // Increase for next time

  function_switch("write_result");
  return;

}
  



// *********************************
// INTERNAL function implementations
// *********************************

// --------------------------------------------------------------------------
// ----- Wrapper function to create group+datasets for current snapshot -----
// --------------------------------------------------------------------------

void create_snapshot_datasets(int nLength) {

#ifdef VERBOSE
  function_switch("create_snapshot_datasets");
#endif  

  std::string sSnapGroup = "Snapshot_" + to_string(g_nSnapB, 3);
  hdf5_create_group(g_sFileNameOut, sSnapGroup);

  std::vector<int> vDims(1);
  vDims.front() = nLength;
  
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/SubHaloIndex", vDims, "int", "Subhalo index of galaxy in current snapshot. Subhaloes identified as corrupted are marked with values between -1 and -3", -100);
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/SubHaloIndexAll", vDims, "int", "Subhalo index of galaxy in current snapshot. No masking of corrupted subhaloes.", -100);
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/LinkLength", vDims, "char", "Length of link to this subhalo. 1 = normal link between adjacent snapshots. 2 = long link omitting one intervening snapshot.", 255);
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/MergeList", vDims, "int", "Index of the galaxy of which this galaxy now forms part (in current snapshot). If [entry] == [index], then the galaxy is still alive and has not (yet) been swallowed by another galaxy. If [entry] < 0, then the galaxy has been completely disrupted.", -100);
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/LinkRank", vDims, "int", "Rank of link to this subhalo. Rank 0 means that this link is the most massive (largest number of particles) of all those sent by the progenitor. Rank 1 means second-most massive, and so on.", -100);
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/LinkChoice", vDims, "int", "Choice of link to this subhalo. Choice 0 means that this link is the most massive (largest number of particles) of all those received by the subhalo. Rank 1 means second-most massive, and so on.", -100);
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/SenderFraction", vDims, "float", "Fraction of eligible (i.e. DM) particles in the progenitor that were in the link to this subhalo. In an ideal world, this would be 1.0", -100);
  hdf5_create_dataset(g_sFileNameOut, sSnapGroup + "/ReceiverFraction", vDims, "float", "Fraction of eligible (i.e. DM) particles in this subhalo that were in the link from its selected progenitor. In an ideal world, this would be 1.0", -100);


  return;
  
}


// --------------------------------------------------------------------
// --- Identify subhaloes that are newly formed in current snapshot ---
// --------------------------------------------------------------------


std::vector<int> find_new_haloes(const std::vector<int> &vHaloListB, 
				 int g_nHaloesB) {

  using std::cout;
  using std::endl;
  
  function_switch("find_new_haloes");


  // Make a map that is '1' for each halo in B that EXISTS
  std::vector<char> vHaloBMap(g_nHaloesB);

  int ii, nCurrVal;

  try {
  for (ii = 0; ii < vHaloListB.size(); ii++) {
    nCurrVal = vHaloListB.at(ii);
    if (nCurrVal >= 0)
      vHaloBMap.at(nCurrVal) = 1;
  }
  
  }
  catch (const std::out_of_range& oor)
    {
      cout << rp() + "Out of range occurred in BLOCK 1!" << endl;
      cout << rp() + "ii = " << ii << ", vHaloListB.size() = " << vHaloListB.size() << endl;
      cout << rp() + "g_nHaloesB = " << g_nHaloesB << ", vHaloBMap.size() = " << vHaloBMap.size() << endl;
      cout << rp() + "nCurrVal = " << nCurrVal << endl;
      cout << rp() + "g_ttSnap = " << g_ttSnap << endl;
      exit(777);
    }

  
  // Find out how many have status '0' -- these are new haloes
  int nNew = 0;
  for (ii = 0; ii < g_nHaloesB; ii++) {
    if (vHaloBMap.at(ii) == 0)
      nNew++;
  }

  std::vector<int> vNewHaloes(nNew, -1);
  int nCurrLoc = 0;

  // Write the result vector -- all new haloes
  
  for (int ii = 0; ii < g_nHaloesB; ii++) {
    if (vHaloBMap.at(ii) == 0)
      vNewHaloes.at(nCurrLoc++) = ii;
  }
  
  
  function_switch("find_new_haloes");
  return vNewHaloes;
}


// -----------------------------------------------------------------------------------
// --- This function writes the next merge list, based on the `starter' merge list ---
// --- and the info in vMergeTargets (generated in main). ----------------------------
// -----------------------------------------------------------------------------------


void fill_mergelist(const std::vector<int> &vMergeTargets,   // Merge targets by SUBHALO
		    const std::vector<int> &vMergeListA,     // Previous (starter) mergelist
		    std::vector<int> &vMergeListB,           // [0] New merge list
		    const std::vector<int> &vHaloListA,      // Starter halolist
		    const std::vector<int> &vMasterHaloLocListB,    // B-rev. halo list
		    std::vector<int> &vHaloListB) {

  int nHaloNumber = vMergeListA.size();
  //  vMergeListB.resize(vMergeListA.size(),-100);  // ???????!!!!!!

  for (int ii = 0; ii < nHaloNumber; ii++) {

    int carrier_index_a = vMergeListA.at(ii);

    // If galaxy ii does not exist anymore in snap B, not even under
    // another number (i.e., as merger result), then we can stop:
    if (carrier_index_a < 0) {
      vMergeListB.at(ii) = ML_DISRUPTED;
      continue;
    }    
    
    // If the galaxy DOES still exist (possibly under different number),
    // then locate its current carrier subhalo:
    int carrier_subhalo_a = vHaloListA.at(carrier_index_a);

    // Special situation: Galaxy has been skipped in snapshot B.
    // In this case, just copy the carrier index (galaxy still exists under this number)
    if (carrier_subhalo_a == BYPASSED) {
      vMergeListB.at(ii) = carrier_index_a;
      continue;
    }
    
    // Make sure that the carrier_subhalo_a value is sensible:
    if (carrier_subhalo_a < 0 or carrier_subhalo_a >= vMergeTargets.size()) {
      std::cout << rp() + "Unexpected value for CARRIER_SUBHALO_A = " << carrier_subhalo_a << std::endl;
      std::cout << rp() + "Galaxy = " << ii << ", carrier_index_a = " << carrier_index_a << std::endl;

      exit(777);
    }

    
    // Look up the corresponding SUBHALO in B:
    int carrier_subhalo_b = vMergeTargets.at(carrier_subhalo_a);

    // Need special treatment for the case that the galaxy was disrupted:
    if (carrier_subhalo_b == ML_DISRUPTED) {
      vMergeListB.at(ii) = ML_DISRUPTED;
      continue;
    }

    // If the galaxy is only by-passing this snapshot, just copy the merge-list entry from A:
    if (carrier_subhalo_b == BYPASSED) {
      vMergeListB.at(ii) = carrier_index_a;
      continue;
    }
    
  
    // Sanity check for sensible value of carrier_subhalo_b:
    if (carrier_subhalo_b < 0 or carrier_subhalo_b >= vMasterHaloLocListB.size()) {
      std::cout << rp() + "Unexpected value for CARRIER_SUBHALO_B = " << carrier_subhalo_b << std::endl;
      std::cout << rp() + "Galaxy = " << ii << ", carrier_index_a = " << carrier_index_a << ", carrier_subhalo_a = " << carrier_subhalo_a << std::endl;

      exit(777);
    }
    
    int carrier_index_b = vMasterHaloLocListB.at(carrier_subhalo_b);
    
    // Yet another sanity check:
    if (carrier_index_b < 0) {
      std::cout << rp() + "Unexpected value for CARRIER_INDEX_B = " << carrier_index_b << std::endl;
      std::cout << rp() + "Galaxy = " << ii << ", carrier_index_a = " << carrier_index_a << ", carrier_subhalo_a = " << carrier_subhalo_a << ", carrier_subhalo_b = " << carrier_subhalo_b << std::endl;
      
      exit(777);
    }

    // Put in correct identification of mergers in SubHaloIndex output:
    if (vHaloListA.at(ii) >= 0 && carrier_index_b != carrier_index_a) {

      // Some more sanity checks:
      if (carrier_index_a != ii) {
	std::cout << rp() + "Unexpected value for carrier_index_a = " << carrier_index_a << " (not equal to ii=" << ii << std::endl;
	exit(778);
      }

      // Check that vHaloListB is actually empty:
      if (vHaloListB.at(ii) >= 0) {
	std::cout << rp() + "Unexpected value for HALOLISTB = " << vHaloListB.at(carrier_index_a) << std::endl;
	std::cout << rp() + "Galaxy = " << ii << ", carrier_index_a=" << carrier_index_a << ", carrier_subhalo_a = " << carrier_subhalo_a << ", carrier_subhalo_b = " << carrier_subhalo_b << std::endl;
	exit(779);
      }

      vHaloListB.at(ii) = MERGED;
    }
    vMergeListB.at(ii) = carrier_index_b;
    
  } // ends loop through haloes
  
  return;

}
