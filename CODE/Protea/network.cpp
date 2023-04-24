// Functions for building and analysing the link network

#include <mpi.h>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <cstring>

#include "init.h"
#include "globals.h"
#include "Config.h"

#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/utilities_mpi.h"

#include "network.h"
#include "utilities_special.h"


// ****************************************
// Declaration of internally used functions
// ****************************************


Link activate_long_links(const Link &Longlinks,
			 const Link &ShortlinksAB,
			 const Link &ShortlinksBC,
			 const Result &SSP,
			 const Result &SS1);


// **********************************
// EXTERNAL Function implementations
// **********************************


// -------------------------------------------------------------------------------------------
// flag_contaminated_subhaloes(): Identify subhaloes with centering problems (e.g. mergers)
// -------------------------------------------------------------------------------------------

std::vector<int> flag_contaminated_subhaloes(const std::vector<int> &vHaloesB,   // B-halo for A-particles
				 const std::vector<long> &vOrderB,    // B-order for A-particles
				 const std::vector<int> &vTHOlist,  // Tasks-->Haloes(A)
				 const std::vector<long> &vFullPartOffsetA, // Halo-->Particles(A)
				 const Result &TracingResult,  // Linking result
				 const Link &LinksAB) {
  function_switch("flag_contaminated_subhaloes");

  using namespace std;

  int rc = 0;
  double dDummy = ElTime();
  double dStartTime = GetTime();

  // ------------------------------------
  // 1.) Send results to individual tasks
  // ------------------------------------

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
  vector<int> vTT_FlagList(0); // initialise as empty - push_back flagged haloes

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

  // vTT_Match and ..._Length now hold the relevant information for each task

  // -------------------------------------
  // 2.) Analyse particles on current task
  // -------------------------------------

  // Determine number of haloes on current task

  int nFirstTaskHalo = vTHOlist.at(mpirank());

  // Loop through haloes one-by-one
  for (int iihalo = 0; iihalo < nHaloesThisTask; iihalo++) {

    long nFirstPartCurr = vFullPartOffsetA.at(nFirstTaskHalo+iihalo)-vFullPartOffsetA.at(nFirstTaskHalo);
    long nNumPartCurr = vFullPartOffsetA.at(nFirstTaskHalo+iihalo+1)-vFullPartOffsetA.at(nFirstTaskHalo+iihalo);

    int nCurrMatch = vTT_Match.at(iihalo);
    int nCurrLength = vTT_Length.at(iihalo);

    // Loop through individual particles of current halo
    for (long iipart_halo = 0; iipart_halo < nNumPartCurr; iipart_halo++) {
      long iipart = nFirstPartCurr + iipart_halo;

      if (vOrderB.at(iipart) >= 5 || vHaloesB.at(iipart) < 0)
	continue;

      // Check whether the halo of this particle in snap B is the same as the 
      // descendant of the current halo in snap B

      // 'Flag' the B-halo hosting current particle if either of two conditions is met:
      // (a) Current galaxy not identified in B
      // (b) Current galaxy descendant is not the halo hosting current particle

      if (vHaloesB.at(iipart) != nCurrMatch || nCurrLength > 1) {
	vTT_FlagList.push_back(vHaloesB.at(iipart));
      }

    } // ends loop through particles in current halo
  } // ends loop through haloes on current task


  // ----------------------------------
  // 3.) Collect results on master task
  // ----------------------------------

  // Build full SnipOffset vector on task 0
  // NOTE: This CLEARS the data from the original tasks (last '1')
  collect_vector_mpi(vTT_FlagList, 0, 0, 1);

  // For clarity, re-name the (full) offset vector (significant only on task 0)
  vector<int> vFullFlagList;
  vFullFlagList.swap(vTT_FlagList);

  // Need to re-work into by-galaxy flag list
  vector<int> vSHFlag;

  if (mpirank() == 0) {

    vSHFlag.resize(g_nHaloesB);
    for (int iihalo = 0; iihalo < g_nHaloesB; iihalo++)
      vSHFlag.at(iihalo) = 0;
    
    for (int iiflag = 0; iiflag < vFullFlagList.size(); iiflag++)
      vSHFlag.at(vFullFlagList.at(iiflag)) = 1;
    
    

  // ---------------------------------
  // 4.) Also check for mass contamination, using Link data
  // -------------------------------------------------------

  // Loop through B-haloes
    for (int iihaloB = 0; iihaloB < g_nHaloesB; iihaloB++) {
      
      double dContFrac = 0;
      
      // Loop through links received by current B-halo
      for (int iilink_recv = LinksAB.ReceiverOffset.at(iihaloB);
	   iilink_recv < LinksAB.ReceiverOffset.at(iihaloB+1);
	   iilink_recv++) {
	
	int iilink = LinksAB.SortedByRecv.at(iilink_recv);
	int nHaloA = LinksAB.Sender.at(iilink);
	
	if (TracingResult.Length.at(nHaloA) > 1)
	  dContFrac += LinksAB.ReceiverFraction.at(iilink);
	
      } // ends loop through shortlinks received by current B-halo
      
      if (dContFrac > 0.33333)
	vSHFlag.at(iihaloB) += 2;
      
    } // ends loop through B-haloes
  } // ends master-only section
    
  function_switch("flag_contaminated_subhaloes");

  return vSHFlag;

}  // ends flag_contaminated_subhaloes()

// -------------------------------------------------------------------------------------------
// identify_mergers(): Determine which galaxy ``carries forward'' the original galaxy subhalo
// -------------------------------------------------------------------------------------------

void identify_mergers(const Link &LinksAB, 
		      const Result &SelSendB, 
		      std::vector<int> &vMergeTargets) {
  
  
  function_switch("identify_mergers");

  // Set up the output vector vMergeTargets
  vMergeTargets.clear();
  vMergeTargets.resize(g_nHaloesA);

  // Need to look through galaxies which have NO MATCH in SelSendB:
  // N.B.: SelSendB has 'g_nHaloesA' elements (initialized in main)
  for (int ii = 0; ii < g_nHaloesA; ii++) {
    
    int nCurrSSBMatch = SelSendB.Match.at(ii);

    // If the match is non-negative, the galaxy COULD be traced.
    if (nCurrSSBMatch >= 0)  {

      // Debug update: need to also check whether galaxy was short- or long-linked.
      // If it was long-linked, write BYPASSED here.
      // The fill_mergelist() function in write_result() will interpret this correctly
      // and just copy the mergelist entry from snap A.

      if (SelSendB.Length.at(ii) == 1)
	vMergeTargets.at(ii) = nCurrSSBMatch;
      else 
	vMergeTargets.at(ii) = BYPASSED; 
      
    }
    // If galaxy is no longer alive, need to check whether it has merged:
    else {
      
      // Need to check whether there are ANY shortlinks from this galaxy:
      int nShortLinksCurr = LinksAB.SenderOffset.at(ii+1) - LinksAB.SenderOffset.at(ii);
      
      if (nShortLinksCurr == 0) {
	
	// Ok. There's not much we can do here, because there is no shortlink.
	// I.e., this galaxy has been completely disrupted. Not sure why this would
	// happen, but if it's non-rare, someone should probably look into it...
	
	vMergeTargets.at(ii) = ML_DISRUPTED;
	continue;   // go to next subhalo now.
      }
      
      // At this point, we are guaranteed that there IS at least one shortlink.
      // We don't care about anything else than the rank-1 one, which is the
      // one pointed to by LinksAB.SenderOffset[ii]:

      int nMergeTarget = LinksAB.Receiver.at(LinksAB.SenderOffset.at(ii));
      vMergeTargets.at(ii) = nMergeTarget;
     
    } // ends section only if current galaxy has not been traced

  } // ends loop through individual galaxies

  function_switch("identify_mergers");
  return;
}


// ---------------------------------------
// Debugging function to output all links
// ---------------------------------------

void output_links(const Link &lLink,
		  std::string sFileName) {

  using namespace std;
  fstream file;

  char cmap[1024];
  strcpy(cmap, sFileName.c_str());
  
  file.open(cmap, ios_base::out);
  file << "Index | Sender | Receiver | Rank | Choice | SenderFraction | ReceiverFraction | NumPart | SortedByRecv" << endl;

  std::string sSendString, sRecvString, sRankString, sChoiceString, sSFString, sRFString, sNPString, sSBRString;
  for (int ii = 0; ii < lLink.Sender.size(); ii++) {

    if (ii < lLink.Sender.size())
      sSendString = to_string(static_cast<long long>(lLink.Sender.at(ii)));
    else
      sSendString = "---";

    if (ii < lLink.Receiver.size())
      sRecvString = to_string(static_cast<long long>(lLink.Receiver.at(ii)));
    else
      sRecvString = "---";

    if (ii < lLink.Rank.size())
      sRankString = to_string(static_cast<long long>(lLink.Rank.at(ii)));
    else
      sRankString = "---";

    if (ii < lLink.Choice.size())
      sChoiceString = to_string(static_cast<long long>(lLink.Choice.at(ii)));
    else
      sChoiceString = "---";

    if (ii < lLink.SenderFraction.size())
      sSFString = to_string(static_cast<long double>(lLink.SenderFraction.at(ii)));
    else
      sSFString = "---";

    if (ii < lLink.ReceiverFraction.size())
      sRFString = to_string(static_cast<long double>(lLink.ReceiverFraction.at(ii)));
    else
      sRFString = "---";

    if (ii < lLink.NumPart.size())
      sNPString = to_string(static_cast<long long>(lLink.NumPart.at(ii)));
    else
      sNPString = "---";

    if (ii < lLink.SortedByRecv.size())
      sSBRString = to_string(static_cast<long long>(lLink.SortedByRecv.at(ii)));
    else
      sSBRString = "---";
    

    file << ii << ":   " << sSendString + "  " + sRecvString + "  " + sRankString + "  " + sChoiceString + "  " + sSFString + "  " + sRFString + "  " + sNPString + "  " + sSBRString << endl;

  }  
  
  file << endl;
  file << endl;

  file << "Sender and receiver offset lists follow below: " << endl;
  file << "-----------------------------------------------" << endl;
  file << endl;
  file << "SenderOffset  |  ReceiverOffset" << endl;
  
  int nOffsetMax = max<int>(lLink.SenderOffset.size(), lLink.ReceiverOffset.size());

  for (int ii = 0; ii < nOffsetMax; ii++) {
    if (ii < lLink.SenderOffset.size())
      sSendString = to_string(static_cast<long long>(lLink.SenderOffset.at(ii)));
    else
      sSendString = "---";

    if (ii < lLink.ReceiverOffset.size())
      sRecvString = to_string(static_cast<long long>(lLink.ReceiverOffset.at(ii)));
    else
      sRecvString = "---";
    
    file << sSendString + "  " + sRecvString << std::endl;
  }

  file.close();

  return;
}


// -----------------------------------------------------
// build_links(): Build links from ID comparison result.
// -----------------------------------------------------

void build_links(const std::vector<int> &vTaskMatchHaloesAB,
		 const std::vector<long> &vFullPartOffsetA, 
		 const std::vector<int> &vTHOlistA,
		 std::vector<int> &vLinkSenderAB, 
		 std::vector<int> &vLinkReceiverAB, 
		 std::vector<int> &vLinkRankAB, 
		 std::vector<int> &vLinkNumPartAB,
		 std::vector<float> &vLinkSenderFractionAB,
		 int nCheck) {
  
  using namespace std;

  function_switch("build_links");
  
  double dDummy = ElTime();
  double dTime_One = 0, dTime_Two = 0, dTime_Three = 0;
  
  // Set up FULL histogram here (for all haloes on current task):
  
  int success;
  vector<int> vMinMax = minmax<int> (vTaskMatchHaloesAB, 0, success);

  int nHaloOffsetThisTask = vTHOlistA.at(mpirank());
  long nPartOffsetThisTask = vFullPartOffsetA.at(nHaloOffsetThisTask);
  int nNumHaloesThisTask = vTHOlistA.at(mpirank()+1)-nHaloOffsetThisTask;


  // Set up links
  vLinkSenderAB.clear();
  vLinkSenderAB.reserve(nNumHaloesThisTask * g_nLinkRatio);

  vLinkReceiverAB.clear();
  vLinkReceiverAB.reserve(nNumHaloesThisTask * g_nLinkRatio);

  vLinkRankAB.clear();
  vLinkRankAB.reserve(nNumHaloesThisTask * g_nLinkRatio);

  vLinkNumPartAB.clear();
  vLinkNumPartAB.reserve(nNumHaloesThisTask * g_nLinkRatio);

  vLinkSenderFractionAB.clear();
  vLinkSenderFractionAB.reserve(nNumHaloesThisTask * g_nLinkRatio);

  if (success == 0) {
    cout << rp() + "No particle matches at all in build_links() for task " << mpirank() << " (??)" << endl;
    cout << rp() + "   [vMinMax = " << vMinMax.at(0) << ", " << vMinMax.at(1) << "]" << endl;
  } else {
 
  int MinMatchHalo = vMinMax.at(0);
  vector<long> vHistogram(vMinMax.at(1)-vMinMax.at(0)+1,0);

  
#ifdef VERBOSE
  report<double> (ElTime(),                   "Histogram setup [sec.]: ");
  report<int> (MinMatchHalo,                  "Histogram minimum:      ");
  report<long> (vMinMax.at(1)-MinMatchHalo+1, "Histogram length:       ");
  report<int> (nNumHaloesThisTask,            "Number of haloes/task:  ");
#endif  

  
  int curr_LinkRatio = g_nLinkRatio; // In case it needs increasing later... (see below)

#ifdef DEBUG
  cout << rp() + "Now loop through haloes..." << endl;
  cout << rp() + "vTaskMatchHaloesAB.size() = " << vTaskMatchHaloesAB.size() << endl;
#endif

  // Now loop through all A-HALOES on this task, and build links in turn:
  for (size_t ii=0; ii<nNumHaloesThisTask; ii++) {
    
    double dStartTimeHalo = GetTime();
    
    long minval, maxval;
    int success = 0;
    
    // Offset of current A-halo in (hypothetical, full, particle) list: 
    long nFullOffsetThisHaloA = vFullPartOffsetA.at(ii+nHaloOffsetThisTask); 

    // Offset of current A-halo in (real, task-specific) particle list:
    long nTaskOffsetThisHaloA = nFullOffsetThisHaloA - nPartOffsetThisTask;

    // How many particles are there in the current A-halo?
    long nPartThisHaloA = vFullPartOffsetA.at(ii+nHaloOffsetThisTask+1) - nFullOffsetThisHaloA;


    // Loop through individual PARTICLES in current A-halo:
    for (long jj=0; jj < nPartThisHaloA; jj++) {

      long nCurrPosInTaskIDVec;
      int nCurrTargHalo;
      
      nCurrPosInTaskIDVec = jj + nTaskOffsetThisHaloA;
      nCurrTargHalo = vTaskMatchHaloesAB.at(nCurrPosInTaskIDVec);
      
      if (nCurrTargHalo >= 0) {
	vHistogram.at(nCurrTargHalo-MinMatchHalo) += 1;
      }
	
      if (success == 0) {  // If no particle so far had a valid halo attached
	success=1;
	minval=nCurrTargHalo;
	maxval=nCurrTargHalo;
      }
	
      if (nCurrTargHalo < minval)  // Update running min/max as/if appropriate:
	minval=nCurrTargHalo;
      if (nCurrTargHalo > maxval)
	maxval=nCurrTargHalo;
      
      
    } // ends loop through PARTICLES in current A-halo
    

    
    if (success==0) {
#ifdef VERBOSE
      cout << rp() + "No match at all for halo <" << ii << ">" << endl;
#endif 
      continue; // Done with this halo!
    }
     
    // Go through histogram entries and append link lists!

    // Set up a small "temporary link" area, for those we're building for current A-halo:
    vector<int> vLinkTemp_Receiver;
    vLinkTemp_Receiver.reserve(g_nLinkRatio*4);
    
    vector<long> vLinkTemp_NumPart;
    vLinkTemp_NumPart.reserve(g_nLinkRatio*4);
    
    // Loop through individual PARTICLES in current A-halo (again)
    // Although it is more intuitive to loop through the histogram,
    // there are typically many more (sub-)haloes per simulation than
    // particles per (sub-)halo. Therefore, it is much more 
    // efficient to loop through the particles.

    
    for (long jj=0; jj < nPartThisHaloA; jj++) {
      
      long nCurrPosInTaskIDVec = jj + nTaskOffsetThisHaloA;
      int nCurrTargHalo = vTaskMatchHaloesAB.at(nCurrPosInTaskIDVec);
      
      if (nCurrTargHalo < 0) 
	continue; // don't care about non-matched particles
      
      // Need to check that we haven't visited (and re-set!) this histogram entry already:
      long nPartCurrHistEntry = vHistogram.at(nCurrTargHalo-MinMatchHalo);
      
      if (nPartCurrHistEntry == 0)
	continue;

      // Shortcut in case we are enforcing a strict link limit (i.e. min. number of particles)
#ifdef STRICT_LINK_LIMIT
      if (nPartCurrHistEntry < STRICT_LINK_LIMIT) {
	vHistogram.at(nCurrTargHalo - MinMatchHalo) = 0; // Important to also do this here!!
	continue; }
#endif 	    
      
      // [+++++] There would/will need to be some additional code here
      // [+++++] if we want to NOT set the above strict option...

      // Ok - this is the first time we've come here...
      vLinkTemp_Receiver.push_back(nCurrTargHalo);
      vLinkTemp_NumPart.push_back(nPartCurrHistEntry);
     
      // And -important!- re-set the histogram entry:
      vHistogram.at(nCurrTargHalo-MinMatchHalo) = 0;
      
    } // ends loop through PARTICLES in current A-halo (for second time)
    
    // Ok - the temporary link vectors now contain all the data we need.
    // Have to SORT by NumPart!


#ifdef DEBUG
    
    //    cout << rp() + "Halo <" << ii << "> sends " << vLinkTemp_Receiver.size() << " links: " << endl;
    int nCheck = 0;
    for (int xx = 0; xx < vLinkTemp_Receiver.size(); xx++) {
      //  cout << rp() + "   --> " << vLinkTemp_Receiver.at(xx) << " [" << vLinkTemp_NumPart.at(xx) << "]" << endl;
      nCheck += vLinkTemp_NumPart.at(xx);
    }

    if (nCheck > nPartThisHaloA) {
      cout << rp() + "Inconsistent particle numbers in build_links()..." << endl;
      cout << rp() + "MinMatchHalo = " << MinMatchHalo << endl;
      exit(489);
    }

#endif
    
    int nNumLinks = vLinkTemp_Receiver.size();
    vector<long> vSortedLinkIndices = sort_indices_descending<long>(vLinkTemp_NumPart);

    // And now transfer links to full list, in sorted order:
    for (int kk = 0; kk < nNumLinks; kk++) {

      int nSortedLoc = vSortedLinkIndices.at(kk);
      vLinkSenderAB.push_back(ii+nHaloOffsetThisTask);
      vLinkReceiverAB.push_back(vLinkTemp_Receiver.at(nSortedLoc));
      vLinkRankAB.push_back(kk);
      vLinkNumPartAB.push_back(vLinkTemp_NumPart.at(nSortedLoc));
      vLinkSenderFractionAB.push_back(static_cast<float>(static_cast<double>(vLinkTemp_NumPart.at(nSortedLoc)) / static_cast<double>(nPartThisHaloA)));
      
    } // ends transfer of temporary links
  
    // Check if link limit is nearly reached:
    // (in principle, nothing bad would happen, but it may slow down due to frequent
    // re-allocation of the link vectors...)
    
    if (vLinkSenderAB.size() > ((nNumHaloesThisTask * curr_LinkRatio) - (g_nLinkRatio*4))) {
      
      curr_LinkRatio += 2;
      
      vLinkSenderAB.reserve(nNumHaloesThisTask * curr_LinkRatio);
      vLinkReceiverAB.reserve(nNumHaloesThisTask * curr_LinkRatio);
      vLinkRankAB.reserve(nNumHaloesThisTask * curr_LinkRatio);
      vLinkNumPartAB.reserve(nNumHaloesThisTask * curr_LinkRatio);
      vLinkSenderFractionAB.reserve(nNumHaloesThisTask * curr_LinkRatio);
      
    } // ends link expanding section
    
    
  } // ends loop though A-haloes on current task

  /** The following needs to be commented out at the moment because this seems not
      yet be supported by Intel compilers... **/

  /*
  vLinkSenderAB.shrink_to_fit();
  vLinkReceiverAB.shrink_to_fit();
  vLinkRankAB.shrink_to_fit();
  vLinNumPartAB.shrink_to_fit();
  vLinkSenderFractionAB.shrink_to_fit();
  */

  } // ends section only if there was at least one successfully matched particle

  // Last bit: Transfer all links to master task for further processing...

#ifdef VERBOSE
  if (mpirank() == 0)
    cout << rp() + "Finished building links, now collecting from all tasks..." << endl;
#endif

  collect_vector_mpi<int>(vLinkSenderAB, 0, 0, 0); // NOT offset lists, so last flag is 0 here!
  collect_vector_mpi<int>(vLinkReceiverAB, 0, 0, 0);
  collect_vector_mpi<int>(vLinkRankAB, 0, 0, 0);
  collect_vector_mpi<int>(vLinkNumPartAB, 0, 0, 0);
  collect_vector_mpi<float>(vLinkSenderFractionAB, 0, 0, 0);
  
  

  function_switch("build_links");
  return;

}


// **************************************
// Function to invert a link set
// (i.e., do the receiver-based analysis)
// **************************************

void invert_links(Link &Link,
		  const std::vector<long> &vFullPartOffsetList) {
  
  function_switch("invert_links");

  using std::endl;
  using std::cout;
  
  // Sort by target subhaloes
  std::vector<int> vTargetOffset, vRI;
  int nMin = -10, nMax = -10;

  Link.ReceiverOffset.clear();
  Link.ReceiverOffset.resize(vFullPartOffsetList.size(), 0);
  
  Link.SortedByRecv.clear();
  Link.SortedByRecv.resize(Link.Sender.size(),-77);

  Link.ReceiverFraction.clear();
  Link.ReceiverFraction.resize(Link.Sender.size(),-77);

  Link.Choice.clear();
  Link.Choice.resize(Link.Sender.size(),-77);

  if (Link.Sender.size() == 0) {
    function_switch("invert_links");
    return;
  }

  std::vector<int> vRecvLinkOffset;
  std::vector<int> vTargetHist = idlhist<int, int>(Link.Receiver, vRecvLinkOffset, vRI, nMin, nMax,1, 0);


#ifdef DEBUG
  std::cout << rp() + "Completed histogram: " << endl;
  std::cout << rp() + "   nMin = " << nMin << endl;
  std::cout << rp() + "   nMax = " << nMax << endl;
#endif

  // Coda bits
  for (size_t ii = nMax; ii < vFullPartOffsetList.size(); ii++)
    Link.ReceiverOffset.at(ii) = Link.Receiver.size();
  
  
  if (nMin < 0) {
    std::cout << rp() + "vRI[0] = " << vRI.front() << std::endl;
    exit(888);
  }
    
  // Now loop through receiver haloes:
  for (int ii = 0; ii < (nMax-nMin+1); ii++) {
    int nTargetHistCurr = vTargetHist.at(ii);
    Link.ReceiverOffset.at(ii+nMin+1) = Link.ReceiverOffset.at(ii+nMin)+nTargetHistCurr;
    
    if (nTargetHistCurr > 0) {
      
      long nPosOffset = vRecvLinkOffset.at(ii);
      std::vector<int> vLinkTemp_NumPart(nTargetHistCurr);
      for (int jj = 0; jj < nTargetHistCurr; jj++) 
	vLinkTemp_NumPart.at(jj) = Link.NumPart.at(vRI.at(nPosOffset+jj));
    
      
      /*
#ifdef DEBUG
      std::cout << rp() + "Halo <" << ii << ">: Filled in vLinkTemp_NumPart " << std::endl;
#endif DEBUG
      */

      // Sort links by mass:
      std::vector<long> vSortedLinkIndices = sort_indices_descending<int>(vLinkTemp_NumPart);
      long nPartThisHaloB = vFullPartOffsetList.at(ii+nMin+1) - vFullPartOffsetList.at(ii+nMin);;

      // And now fill in full list:
      for (int jj = 0; jj < vTargetHist.at(ii); jj++) {
	
	int nSortedLoc = vSortedLinkIndices.at(jj);
	int IndexInSendSort = vRI.at(nPosOffset+nSortedLoc);
	
#ifdef OLD_SORTEDBYRECV
	Link.SortedByRecv.at(IndexInSendSort) = nPosOffset+jj;
#else
	Link.SortedByRecv.at(nPosOffset+jj) = IndexInSendSort;
#endif	

	Link.Choice.at(IndexInSendSort) = jj;
	Link.ReceiverFraction.at(IndexInSendSort) = static_cast<float>(static_cast<double>(Link.NumPart.at(IndexInSendSort) / static_cast<double>(nPartThisHaloB)));
  
	if (Link.ReceiverFraction.at(IndexInSendSort) > 1.0) {
	  std::cout << rp() + "Big problem: RecvFrac > 1 at link " << IndexInSendSort << " (" << Link.Sender.at(IndexInSendSort) << " --> " << Link.Receiver.at(IndexInSendSort) << ", " << Link.NumPart.at(IndexInSendSort) << " vs. " << nPartThisHaloB << ")" << std::endl;
	  exit(1000);
	}

    } // ends loop through links TO current halo [ii+nMin]
    } // ends section only if current halo has any links to it
  } // ends loop THROUGH TARGET HALOES


  function_switch("invert_links");
  return;
}



// **********************************************
// (Wrapper) function to analyse the link network
// **********************************************


void evaluate_link_network(const Link &LinksAB, 
			   const Link &LinksBC, 
			   const Link &LinksAC, 
			   Result &SelSendB,                            // [0] Main result! 
			   std::vector<long> vFullPartOffsetList,
			   Result &SelRecOut) {
  
  function_switch("evaluate_link_network");

  static Result SelRecOld; // this needs to be passed through time iterations / function calls
  
  std::cout << rp() + "[AB " << LinksAB.Sender.size() << " -- BC " << LinksBC.Sender.size() << " -- AC " << LinksAC.Sender.size() << "]" << std::endl;

  // Initialize iteration

  Result SelSendA, SelRecOldOrig, SelSendB1, SelRecNew, SelSendAPrev, SelSendBPrev, SelSendTemp;
  initialize_result(SelSendA, g_nHaloesA);
  
  // If this is the FIRST snapshot, we also need to initialize SelRecOld:
  // (otherwise, this will have been passed through from last round)
  if (g_ttSnap == 0) {
    initialize_result(SelRecOld, g_nHaloesB);
  }


  // Make a copy of the original version of SelRecOld --> SelRecOldOrig:
  copy_result(SelRecOld, SelRecOldOrig);

  int ii = 0;
  // Now begin the iteration...
  for (ii = 0; ii < g_nMaxNetworkIter; ii++) {

    std::cout << std::endl;
    std::cout << rp() + "Started iteration " << ii << std::endl;
    std::cout << rp() + "Selecting short links..." << std::endl;

    copy_result(SelSendA, SelSendTemp);
    copy_result(SelRecOldOrig, SelRecOld);
  
    // Step A: Select the Short links
    select_links(LinksAB, SelSendTemp, SelRecOld, 1);
    
    Result SelRecOldWithShort;
    copy_result(SelRecOld, SelRecOldWithShort);
    copy_result(SelSendTemp, SelSendB);

#ifdef DEBUG
    std::cout << rp() + "SSB.Match[0] = " << SelSendB.Match.at(0) << std::endl;
#endif


    // If first iteration, save this as sel_send_b1, for future use:
    if (ii == 0)
      copy_result(SelSendB, SelSendB1);
    else {

      // if NOT first iter, check for changes compared to last round,
      // and exit if there are none!

      int nChanges = 0;
      for (int cc = 0; cc < g_nHaloesA; cc++) {
	if (SelSendB.Match.at(cc) != SelSendBPrev.Match.at(cc))
	  nChanges++;
      }

      if (nChanges == 0)
	break;

    } // ends section only for non-first iteration

    // Make copies for next round comparison:
    copy_result(SelSendA, SelSendAPrev);
    copy_result(SelSendB, SelSendBPrev);

    
    // STEP B: ACTIVATE LONG LINKS!
    std::cout << rp() + "Activate long links..." << std::endl;
    Link LonglinksRed = activate_long_links(LinksAC, LinksAB, LinksBC, SelSendB, SelSendB1);
    
#ifdef DEBUG
    std::cout << rp() + "After activate_long_links():" << std::endl;
    std::cout << rp() + "SSB.Match[0] = " << SelSendB.Match.at(0) << std::endl;
#endif

    int nLongLinksActive = LonglinksRed.Sender.size();
    std::cout << rp() + "(" << nLongLinksActive << " found active)" << std::endl;

    // Can stop the iteration if there are no active long links at any point:
    if (nLongLinksActive == 0)
      break;

    // Now invert long links...
    std::cout << rp() + "Invert long links..." << std::endl;
    invert_links(LonglinksRed, vFullPartOffsetList);

    // Set up SelSendA fresh:
    initialize_result(SelSendA, g_nHaloesA);
    initialize_result(SelRecNew, g_nHaloesC);

    // Now select longlinks:
    std::cout << rp() + "Select long links..." << std::endl;
    select_links(LonglinksRed, SelSendA, SelRecNew, 2);
    
    // Make comparison to previous SS-A:
    
    int nChanges = 0;
    for (int cc = 0; cc < g_nHaloesA; cc++) {
      if (SelSendA.Match.at(cc) != SelSendAPrev.Match.at(cc))
	nChanges++;
    }
    
    if (nChanges == 0)
      break;

        
  } // ends iteration loop
  
  if (ii == g_nMaxNetworkIter) {
    std::cout << rp() + "Link selection did NOT converge after " << g_nMaxNetworkIter << " iterations. Aborting..." << std::endl;
    exit(42);
  }
    
  std::cout << rp() + "Link selection required " << ii << " iterations." << std::endl;
  
  // Just some tidying-up:
  if (SelRecNew.Match.size() > 0)
    copy_result(SelRecNew, SelRecOld);
  else 
    initialize_result(SelRecOld, g_nHaloesC);
  
  // Generate a copy of SelRecOld for return to main:
  copy_result(SelRecOld, SelRecOut);

  function_switch("evaluate_link_network");
  return;

}

// -------------------------------------------------------------------------
// Convenience function to turn a set of link vectors into single structure:
// -------------------------------------------------------------------------

Link build_link_struct(std::vector<int> &vLinkSender,
		       std::vector<int> &vLinkReceiver,
		       std::vector<int> &vLinkRank,
		       std::vector<int> &vLinkNumPart,
		       std::vector<float> &vLinkSenderFraction,
		       int nHaloes,
		       int nVerb) 
{

#ifdef DEBUG
  nVerb = 1;
#endif
  
  if (nVerb == 1)
    function_switch("build_link_struct");
  
  Link lLink;
  
  lLink.Sender.swap(vLinkSender);
  lLink.Receiver.swap(vLinkReceiver);
  lLink.Rank.swap(vLinkRank);
  lLink.NumPart.swap(vLinkNumPart);
  lLink.SenderFraction.swap(vLinkSenderFraction);

  lLink.SenderOffset.resize(nHaloes+1);
  lLink.SenderOffset = make_offset(lLink.Sender, 0, nHaloes,nVerb); // final 0: silent
  
  if (nVerb == 1)
    function_switch("build_link_struct");
  return lLink;
}


// *******************************************
// Implementation of INTERNALLY USED functions
// *******************************************


void select_links(const Link &Links,
		  Result &ss,
		  Result &sr,
		  int length) {

  function_switch("select_links");

  std::cout << rp() + "[Number of links = " << Links.Sender.size() << "]" << std::endl;

  if (Links.Sender.size() == 0) {
    function_switch("select_links");
    return;
  }

  using std::endl;
  using std::cout;

  int nLinks = Links.Sender.size();

#ifdef EXHAUST_LINKS
  int tripout_counter = 0;
#endif

  // Find lowest choice in input (i.e. MAX choice!)
  int nLowestChoice = max<int>(Links.Choice);
  int nLowestChoiceUsed = nLowestChoice;

  std::cout << rp() + "[Lowest choice level is " << nLowestChoice << "]" << std::endl;

#ifdef LINK_MAX_CHOICE
  if (nLowestChoiceUsed > LINK_MAX_CHOICE)
    nLowestChoiceUsed = LINK_MAX_CHOICE;

  std::cout << rp() + "[Only using up to " << nLowestChoiceUsed << "]" << std::endl;
#endif

  // Loop through different choice levels, from top to bottom:
  for (int cc = 0; cc <= nLowestChoiceUsed; cc++) {

#ifdef DEBUG
    cout << rp() + "Choice level " << cc << "..." << endl;
#endif DEBUG

    double dCStartTime = GetTime();

    // Mark CURRENT links:
    std::vector<int> vActiveSenders;
    std::vector<int> vActiveLinks;

    vActiveSenders.reserve(Links.Sender.size());
    vActiveLinks.reserve(Links.Sender.size());
    
#ifdef DEBUG
    cout << rp() + "nLinks = " << nLinks << endl;
    cout << rp() + "Links.Sender.size() = " << Links.Sender.size() << endl;
    cout << rp() + "Links.Receiver.size() = " << Links.Receiver.size() << endl;
    cout << rp() + "Links.Choice.size() = " << Links.Choice.size() << endl;
    cout << rp() + "ss.Match.size() = " << ss.Match.size() << endl;
    cout << rp() + "sr.Match.size() = " << sr.Match.size() << endl;
#endif DEBUG

    std::vector<int> vSendLinkOffset, vSRI;
    int nMin = -10, nMax = -10;
    std::vector<int> vSenderHist;
    std::vector<int> vIndexSend;
    
    vIndexSend.reserve(Links.Sender.size());

    double dTimeEndSetup = GetTime();
    std::cout << rp() + "   <Setup took " << dTimeEndSetup - dCStartTime << " sec.>" << std::endl;

    // Loop through all links to filter out those of interest in current choice-iteration
    for (int rr = 0; rr < nLinks; rr++) {

      // Criteria: SENDING halo not yet occupied,
      //           RECEIVING halo not yet occupied,
      //           CURRENT CHOICE level
      
      if ((ss.Match.at(Links.Sender.at(rr)) < 0) and (sr.Match.at(Links.Receiver.at(rr)) < 0) and (Links.Choice.at(rr) == cc)) {

	// The following two clauses were added 26 Apr 2016, to limit tracing to sensible cases

#ifdef LINK_MIN_SENDFRAC
	if (Links.SenderFraction.at(rr) < Params.dMinSendFrac)
	  continue;
#endif

#ifdef LINK_MIN_RECVFRAC
	if (Links.ReceiverFraction.at(rr) < Params.dMinRecvFrac)
	  continue;
#endif

#ifdef LINK_MAX_RANK
	if (Links.Rank > LINK_MAX_RANK)
	  continue;
#endif

	// Added 28-APR-2016:
	// Limit selectability of lower-choice links to cases where their mass is close to choice-0
 
	if (cc > 0) {
	  int nCurrLinkRecv = Links.Receiver.at(rr);
	  int nIndCurrPrimaryLinkRecv = Links.SortedByRecv.at(Links.ReceiverOffset.at(nCurrLinkRecv));
	  
	  if (Links.ReceiverFraction.at(rr) < Params.dMinFracLowChoice * Links.ReceiverFraction.at(nIndCurrPrimaryLinkRecv))
	    continue;
	}

	vActiveLinks.push_back(rr);
	vActiveSenders.push_back(Links.Sender.at(rr));
      }
    }

    double dTimeEndFind = GetTime();
    std::cout << rp() + "   <Finding interesting ones took " << dTimeEndFind - dTimeEndSetup << " sec.>" << std::endl;
    
    
#ifdef DEBUG_SELECT_LINKS
    cout << rp() + "Found " << vActiveLinks.size() << " active links at current level..." << endl;
#endif DEBUG_SELECT_LINKS
    
    //     Can end process if there are no more links of current choice!
    if (vActiveSenders.size() == 0) {
      //  cout << rp() + "Breaking at cc = " << cc << endl;
#ifdef EXHAUST_LINKS
      
      if (++tripout_counter > EXHAUST_LINKS)
	break;
      else
	continue;
#else
      break;
#endif
    }

#ifdef EXHAUST_LINKS
    // Re-set the counter to zero if we HAVE found a nonzero number of interesting links
    tripout_counter = 0;
#endif
    
    vSenderHist = idlhist<int, int>(vActiveSenders, vSendLinkOffset, vSRI, nMin, nMax,1,0);
    
    // Select "successful" senders: These are AT THE OFFSET levels, because they will be the 
    // HIGHEST-RANK (first in list) with a given Sender.
    
    for (int rr = 0; rr < vSenderHist.size(); rr++)
      if (vSenderHist.at(rr) > 0)
	vIndexSend.push_back(rr);

    double dTimeEndSelect = GetTime();
    std::cout << rp() + "   <Selecting successful ones took " << dTimeEndSelect - dTimeEndFind << " sec.>" << std::endl;
    
    
    // Now fill in the SS/SR results:
    
    for (int mm = 0; mm < vIndexSend.size(); mm++) {
      
      int nThisLink = vActiveLinks.at(vSRI.at(vSendLinkOffset.at(vIndexSend.at(mm))));
      int nThisSender = vIndexSend.at(mm)+nMin;
      int nThisReceiver = Links.Receiver.at(nThisLink);

#ifdef DEBUG_SELECT_LINKS
      cout << rp() + "mm = " << mm << endl;
      cout << rp() + "nThisLink = " << nThisLink << endl;
      cout << rp() + "nThisSender = " << nThisSender << endl;
      cout << rp() + "nThisReceiver = " << nThisReceiver << endl;
#endif

      ss.Match.at(nThisSender)  = nThisReceiver;
      ss.Length.at(nThisSender) = length;
      ss.Rank.at(nThisSender) = Links.Rank.at(nThisLink);
      ss.Choice.at(nThisSender) = Links.Choice.at(nThisLink);
      ss.LinkInd.at(nThisSender) = nThisLink;
      ss.SenderFraction.at(nThisSender) = Links.SenderFraction.at(nThisLink);
      ss.ReceiverFraction.at(nThisSender) = Links.ReceiverFraction.at(nThisLink);
      
#ifdef DEBUG_SELECT_LINKS
      cout << rp() + "Filled in sender list..." << endl;
#endif 

      sr.Match.at(nThisReceiver) = nThisSender;
      sr.Length.at(nThisReceiver) = length;
      sr.Rank.at(nThisReceiver) = Links.Rank.at(nThisLink);
      sr.Choice.at(nThisReceiver) = Links.Choice.at(nThisLink);
      sr.LinkInd.at(nThisReceiver) = nThisLink;
      sr.SenderFraction.at(nThisReceiver) = Links.SenderFraction.at(nThisLink);
      sr.ReceiverFraction.at(nThisReceiver) = Links.ReceiverFraction.at(nThisLink);

    } // ends loop to fill in SS/SR 

    double dTimeEndFill = GetTime();
    std::cout << rp() + "   <Filling in lists took " << dTimeEndFill - dTimeEndSelect << " sec.>" << std::endl;
    
      
    double dCTime = GetTime() - dCStartTime;
    std::cout << rp() + "Choice iteration " << cc << " took " << dCTime << " sec." << std::endl;

  } // ends loop through CHOICE levels
  
  

  function_switch("select_links");
  return;
}



Link activate_long_links(const Link &Longlinks,
			 const Link &ShortlinksAB,
			 const Link &ShortlinksBC,
			 const Result &SSA,
			 const Result &SS1) {

  function_switch("activate_long_links");
  
  using std::cout;
  using std::endl;

  double dDummy = ElTime();

  enum TimeCat
  {
    AL_MISC,
    AL_CASE1,
    AL_CASE2A,
    AL_CASE2B,
    AL_CASE2C,
    AL_CASE2I,
    AL_CASE2I_SUSPLINK
  };


  std::vector<double> vTimeBreakDown(7,0);  

  int nLonglinks = Longlinks.Sender.size();
  int nHaloA = SSA.Match.size();

  Link Longlinks_Red;

  // Sanity check: Are there any input elements?
  if (nLonglinks == 0) {
    
    std::cout << rp() + "No longlinks in input! Returning..." << std::endl;
    Longlinks_Red.Sender.clear();
    function_switch("activate_long_links");    
    
    return Longlinks_Red;
  }
  
  // Alright... Make a marking vector to see which ones are 'active':
  std::vector<char> vActiveMap(nLonglinks, 0);

  // This game is played in a A-Halo by A-Halo way, so let's loop:
  for (int ii = 0; ii < nHaloA; ii++) {

#ifdef DEBUG
    std::cout << rp() + "Now processing halo " << ii << std::endl;
#endif

    int nLongCurr = Longlinks.SenderOffset.at(ii+1) - Longlinks.SenderOffset.at(ii);
    
    // First check: Does the current A-halo actually send any longlinks?
    // If not, we can stop this iteration right here:
    if (nLongCurr == 0)
      continue;

#ifdef DEBUG
    std::cout << rp() + "nLongCurr = " << nLongCurr << std::endl;
#endif
    
    // Ok, just checking... You *do* send at least one longlink. 
    // For convenience, store their offset, so we can retrieve them later:
    int nCurrLongOffset = Longlinks.SenderOffset.at(ii);
    
    
    // Now let's see how many *shortlinks* there are...
    int nShortCurr = ShortlinksAB.SenderOffset.at(ii+1)-ShortlinksAB.SenderOffset.at(ii);

#ifdef DEBUG
    std::cout << rp() + "nShortCurr = " << nShortCurr << std::endl;
#endif
    
    // There are two possible scenarios:
    // CASE I: There are NO shortlinks. In this case, it's kind-of easy...
    // CASE II: There are SOME shortlinks. In this case, it's harder...

    // -------- CASE I ---------

    if (nShortCurr == 0) {

      vTimeBreakDown.at(AL_MISC) += ElTime();

#ifdef DEBUG
      std::cout << rp() + "Executing CASE-I..." << std::endl;
#endif


      // We still need to compare the longlinks against SHORTLINKS to SAME TARGET
      // This is because the Longlinks will be selected in this snapshot iteration,
      // but the shortlinks only later, so we'd better be sure that there is 
      // not some more suitable progenitor from snapshot B!

      // Do this on Long-link by Long-link basis 
      // (Remember, these all come from same A-halo, so all go to different 
      // haloes in C by definition)

      for (int ll = 0; ll < nLongCurr; ll++) {

	int nCurrLongInd, nCurrTarget, nShortSame, nMassShortTemp, nMassLongTemp;

	try {

	// Determine index and target of current long link:
	nCurrLongInd = nCurrLongOffset + ll;
	nCurrTarget = Longlinks.Receiver.at(nCurrLongInd);

	// Determine number of SHORTLINKS received by same target (in C):
	nShortSame = ShortlinksBC.ReceiverOffset.at(nCurrTarget+1) - ShortlinksBC.ReceiverOffset.at(nCurrTarget);

	}
	catch (const std::out_of_range& oor)
	{
	  cout << endl;
	  cout << rp() + "Out of range occurred in Block 1!" << endl;
	  exit(777);
	}




	// If NO shortlink is received, then the case is clear:
	if (nShortSame == 0)

	  try {
	  vActiveMap.at(nCurrLongInd) = 1;
	  }
	  catch (const std::out_of_range& oor)
	    {
	      cout << endl;
	      cout << rp() + "Out of range occurred in Block 2!" << endl;
	      exit(777);
	    }
	
	
	else {
	  // If at least one short link is received, we compare masses.
	  // Specifically, look at mass of FIRST CHOICE (== most massive!)
	  // shortlink. If longlink beats this, take it! Otherwise, leave it!
	  
	  try {
	    nMassShortTemp = ShortlinksBC.NumPart.at(ShortlinksBC.SortedByRecv.at(ShortlinksBC.ReceiverOffset.at(nCurrTarget)));
	  }
	  catch (const std::out_of_range& oor)
	    {
	      cout << endl;
	      cout << rp() + "Out of range occurred in Block 3a!" << endl;
	      cout << rp() + "nCurrTarget = " << nCurrTarget << endl;
	      cout << rp() + "ShortlinksBC.ReceiverOffset.size() = " << ShortlinksBC.ReceiverOffset.size() << endl;
	      cout << rp() + "SLBC.ReceiverOffset[] = " << ShortlinksBC.ReceiverOffset.at(nCurrTarget) << endl;
	      print_vector<int>(ShortlinksBC.ReceiverOffset, "SLBC.RecOffset end", ShortlinksBC.ReceiverOffset.size()-11, ShortlinksBC.ReceiverOffset.size()-1);
	      cout << rp() + "ShortlinksBC.SortedByRecv.size() = " << ShortlinksBC.SortedByRecv.size() << endl;
	      cout << rp() + "SLBC.SortedByRecv[] = " << ShortlinksBC.SortedByRecv.at(ShortlinksBC.ReceiverOffset.at(nCurrTarget)) << endl;
	      cout << rp() + "SLBC.NumPart[] = " << ShortlinksBC.NumPart.at(ShortlinksBC.SortedByRecv.at(ShortlinksBC.ReceiverOffset.at(nCurrTarget))) << endl;

	      exit(777);
	    }
	  
	  try {
	  nMassLongTemp = Longlinks.NumPart.at(nCurrLongInd);
	  }
	  catch (const std::out_of_range& oor)
	    {
	      cout << endl;
	      cout << rp() + "Out of range occurred in Block 3b!" << endl;
	      exit(777);
	    }
	  

	  if (nMassLongTemp > nMassShortTemp) {
	    try {
	      vActiveMap.at(nCurrLongInd) = 1;
	    }
	    catch (const std::out_of_range& oor)
	      {
		cout << endl;
		cout << rp() + "Out of range occurred in Block 3b!" << endl;
		exit(777);
	      }
	  }
      
	} // ends section IF some shortlinks are received by same target as the longlink
      } // ends loop through longlinks from current A-halo (for Case-I)
      
      vTimeBreakDown.at(AL_CASE1) += ElTime();




    } else {// Ends Case-I section: If there is NO shortlink sent from same halo as longlink
    

      // ----------------- CASE II -------------------------------------
      // ------- Reminder: this means that the current A-halo also sends
      // ------- at least one short link. It's a bit harder to justify
      // ------- use of longlinks in this case, but it can be done:
      // --------------------------------------------------------------

#ifdef DEBUG
      std::cout << rp() + "Executing CASE-II..." << std::endl;
#endif

      
      // Check if any short links are selected in AB:
      int nShortSel = SSA.Match.at(ii), nShortSel1 = SS1.Match.at(ii);
      int nShortLength = SSA.Length.at(ii), nShortLength1 = SS1.Length.at(ii);
      
      // Set up a map for 'suspicious' short link targets:
      // The purpose of this will become clear in a bit...
      std::vector<char> vSuspMap(nShortCurr, 0);

#ifdef DEBUG
      std::cout << rp() + "nShortSel = " << nShortSel << ", nShortSel1 = " << nShortSel1 << std::endl;
      std::cout << rp() + "nShortLength = " << nShortLength << ", nShortLength1 = " << nShortLength1 << std::endl;

#endif
      
      if ((nShortSel >= 0) and (nShortLength == 1)) {
	
	// **--**-- CASE II-a: A SHORTLINK (Length=1!) has been selected already from A --**--**

	vTimeBreakDown.at(AL_MISC) += ElTime();

#ifdef DEBUG
	std::cout << rp() + "Executing CASE-IIa..." << std::endl;
#endif
	
	// Find the RANK of the selected shortlink. 
	// All links with HIGHER RANK (if they exist) are SUSPICIOUS

	int nSelRank = SSA.Rank.at(ii);

#ifdef DEBUG
	std::cout << rp() + "   nSelRank = " << nSelRank << std::endl;
#endif
	

	if (nSelRank > 0) {
	  for (int jj = 0; jj < nSelRank; jj++)
	    vSuspMap.at(jj) = 1;
	}
	
	vTimeBreakDown.at(AL_CASE2A) += ElTime();

      } else { // ends identification of CASE-IIa. We'll be back shortly...

	// Case-II-nonA:
	
	if ((nShortSel1 >= 0) and (nShortLength1 == 1)) {

	  // **--**-- CASE II-b: No shortlink selected in previous iteration, but 
	  //                     in the first!

	  vTimeBreakDown.at(AL_MISC) += ElTime();
	  
#ifdef DEBUG
	  std::cout << rp() + "Executing CASE-IIb..." << std::endl;
#endif
	  
	  
	  // Find the RANK of the selected shortlink. 
	  // All links with HIGHER RANK (if they exist) are SUSPICIOUS
	  
	  int nSelRank = SS1.Rank.at(ii);
	  if (nSelRank > 0) {
	    for (int jj = 0; jj < nSelRank; jj++)
	      vSuspMap.at(jj) = 1;
	  }
	  
	  // AND all those whose target is selected by another link in AB:
	  for (int jj = 0; jj < nShortCurr; jj++) {
	    int nCurrShortTarg = ShortlinksAB.Receiver.at(ShortlinksAB.SenderOffset.at(ii) + jj);
	    
	    // Now see if the selected target is matched by some selected link:
	    int nMatch = 0;
	    for (int ww = 0; ww < SSA.Match.size(); ww++) {
	      if ((SSA.Match.at(ww) == nCurrShortTarg) and (SSA.Length.at(ww) == 1))
		nMatch += 1;
	    }
	    
	    // Sanity check to make sure there is not more than one match!
	    if (nMatch > 1) {
	      std::cout << rp() + "Target selected multiply???" << std::endl;
	      std::cout << rp() + "(nMatch = " << nMatch << ")" << std::endl;
	      exit(742);
	    }

	    // Now check if nMatch is 0 or 1
	    // If 1 --> suspicious short link jj.

	    if (nMatch == 1)
	      vSuspMap.at(jj) = 1;
	    
	  } // ends loop through all short links
	      
	  vTimeBreakDown.at(AL_CASE2B) += ElTime();

	} else { // Ends CASE-IIb

	  // **--**-- CASE-IIc: None selected in PREV or 1st. 
	  // ==> ALL shortlinks are suspicious

	  vTimeBreakDown.at(AL_MISC) += ElTime();

#ifdef DEBUG
	std::cout << rp() + "Executing CASE-IIc..." << std::endl;
#endif

	for (int jj = 0; jj < vSuspMap.size(); jj++)
	  vSuspMap.at(jj) = 1;
	
	vTimeBreakDown.at(AL_CASE2C) += ElTime();

	} // ends CASE-IIc
      } // ends CASE-II-nonA 
      
      
      // ------- Now all three CASE-II subtypes are treated together -------
      // For this, we create a list of all suspicious shortlinks:
      // vIndexShortSusp

      vTimeBreakDown.at(AL_MISC) += ElTime();

      std::vector<int> vIndexShortSusp;
      for (int jj = 0; jj < vSuspMap.size(); jj++) {
	if (vSuspMap.at(jj) == 1)
	  vIndexShortSusp.push_back(jj);
      }

      int nSusp = vIndexShortSusp.size();

#ifdef DEBUG
      std::cout << rp() + "nSusp = " << nSusp << std::endl;
#endif

      
      // It's SIMPLE if there are no such suspicious shortlinks:
      if (nSusp == 0)
	continue;
      
      // Good. We now know that there ARE suspicious shortlinks...
      // Find the suspicious subhaloes in snapshot B:

      std::vector<int> vSuspSHI(nSusp);
      for (int jj = 0; jj < nSusp; jj++) 
	vSuspSHI.at(jj) = ShortlinksAB.Receiver.at(ShortlinksAB.SenderOffset.at(ii) + vIndexShortSusp.at(jj));
      
      // ----------------------------------------------------
      // Now check each LONG-LINK in turn for "I-1" and "I-2"
      // (These are different reasons for being invalid)
      // ----------------------------------------------------

      // First, build up a list of all BC-Shortlinks coming from a 
      // suspicious B halo:
      
      // std::vector<char> vSuspLinkMap(ShortlinksBC.Sender.size(), 0);
      std::vector<int> vIndexSuspLink;

      for (int jj = 0; jj < nSusp; jj++) {
	int nCurrFirstSusp = ShortlinksBC.SenderOffset.at(vSuspSHI.at(jj));
	int nCurrLastSusp = ShortlinksBC.SenderOffset.at(vSuspSHI.at(jj)+1)-1; 
	  
	for (int kk = nCurrFirstSusp; kk <= nCurrLastSusp; kk++) {
	  vIndexSuspLink.push_back(kk);
	}
      }
      
      /*
      double dStartTemp = GetTime();
      
      std::vector<int> vIndexSuspLink;
      for (int jj = 0; jj < vSuspLinkMap.size(); jj++) {
	if (vSuspLinkMap.at(jj) == 1)
	  vIndexSuspLink.push_back(jj); 
      }
      
      vTimeBreakDown.at(AL_CASE2I_SUSPLINK) += (GetTime()-dStartTemp);
      */

      // Now, make a list of the (current) LONGLINKS, to 
      // mark out which are INVALID:

      std::vector<char> vInvalidMap(nLongCurr, 0);
      
      // Go through each long link (from current A-halo),
      // and check whether it connects to the same 
      // C-halo as a suspicious B-halo

      for (int jj = 0; jj < nLongCurr; jj++) {

	// Determine the index and target of current longlink:
	int nCurrLongInd = nCurrLongOffset + jj;
	int nCurrTarget = Longlinks.Receiver.at(nCurrLongInd);

	// Determine the number of shortlinks received by same target:
	int nShortSame = ShortlinksBC.ReceiverOffset.at(nCurrTarget+1) - ShortlinksBC.ReceiverOffset.at(nCurrTarget);

	// If there ARE NO such shortlinks:
	// Current target cannot be 'tunnelled' through a suspicious halo in B
	// There may be (contrived?) scenarios where we would still want to 
	// include these, but for simplicity this is excluded here. 

	if (nShortSame == 0)
	  vInvalidMap.at(jj) = 4;
	else {

	  // I2 test: Is this target's FIRST CHOICE shortlink
	  //          not sent by a suspicious subhalo?
	  
	  int nFCSender = ShortlinksBC.Sender.at(ShortlinksBC.SortedByRecv.at(ShortlinksBC.ReceiverOffset.at(nCurrTarget)));
	  char cI2 = 0;
	  for (int kk = 0; kk < vSuspSHI.size(); kk++) {
	    if (vSuspSHI.at(kk) == nFCSender)
	      cI2 = 1;
	  }
	  
	  if (cI2 == 0)
	    vInvalidMap.at(jj) = 2;

	  
	  // I1 test: Is this target connected to a susp
	  //          via primary link?
	  
	  char cI1 = 0;
	  for (int kk = 0; kk < vIndexSuspLink.size(); kk++) {
	    if ((vIndexSuspLink.at(kk) == nCurrTarget) and (ShortlinksBC.Rank.at(vIndexSuspLink.at(kk)) == 0))
	      cI1 = 1;
	  }
	  
	  if (cI1 == 0)
	    vInvalidMap.at(jj) = 1;
	  
	} // ends section only if there are shortlinks to current target
      } // ends loop over long-links

      
      // Final bit: I3 test. Mark out any long links that are
      //                     of lower rank than highest rank I2

      std::vector<int> vIndexI2;
      for (int jj = 0; jj < vInvalidMap.size(); jj++) {
	if (vInvalidMap.at(jj) == 2)
	  vIndexI2.push_back(jj);
      }

      // Only need to do this if there ARE some I2 long-links:
      if (vIndexI2.size() > 0) {
	
	std::vector<int> vTempRank(vIndexI2.size());
	for (int jj = 0; jj < vIndexI2.size(); jj++) 
	  vTempRank.at(jj) = Longlinks.Rank.at(nCurrLongOffset+vIndexI2.at(jj));
	
	int nTopI2Rank = min<int>(vTempRank);
	
	for (int jj = nTopI2Rank+1; jj < nLongCurr; jj++)
	  vInvalidMap.at(jj) = 3;
	  
      }

#ifdef DEBUG
      print_vector<char>(vInvalidMap, "vInvalidMap");
#endif

      // Now we can FINALLY select the active links:
      
      for (int jj = 0; jj < nLongCurr; jj++) {
	if (vInvalidMap.at(jj) == 0)
	  vActiveMap.at(nCurrLongOffset + jj) == 1;
      }

      vTimeBreakDown.at(AL_CASE2I) += ElTime();
		
    } // ends Case-II (!!!)

    /*
#ifdef DEBUG
    std::cout << rp() + "Finished first halo, end for now..." << std::endl;
    exit(1000);
#endif
    */

  } // ends loop through A-haloes 

  
  // Find total number of active long links:
  int nActive = 0;
  for (int ii = 0; ii < vActiveMap.size(); ii++) {
    if (vActiveMap.at(ii) == 1)
      nActive++;
  }

  // Can stop if there are no active links:
  if (nActive == 0) {
    
    std::cout << rp() + "No active longlinks found! Returning..." << std::endl;
    Longlinks_Red.Sender.clear();
    function_switch("activate_long_links");    
    
    return Longlinks_Red;
    
  }
    
  // Now build the reduced Longlink list:
  Longlinks_Red.Sender.resize(nActive,-1);
  Longlinks_Red.Receiver.resize(nActive,-1);
  Longlinks_Red.Rank.resize(nActive, -1);
  Longlinks_Red.NumPart.resize(nActive, -1);
  Longlinks_Red.SenderFraction.resize(nActive, -1);

  int nCurrLocRed = 0;
  for (int ii = 0; ii < vActiveMap.size(); ii++) {
    if (vActiveMap.at(ii) == 1) {

      Longlinks_Red.Sender.at(nCurrLocRed) = Longlinks.Sender.at(ii);
      Longlinks_Red.Receiver.at(nCurrLocRed) = Longlinks.Receiver.at(ii);
      Longlinks_Red.Rank.at(nCurrLocRed) = Longlinks.Rank.at(ii);
      Longlinks_Red.NumPart.at(nCurrLocRed) = Longlinks.NumPart.at(ii);
      Longlinks_Red.SenderFraction.at(nCurrLocRed) = Longlinks.SenderFraction.at(ii);
      
      nCurrLocRed++; // This is kind-of important...
						  
    }
  }

  // Build new offset list:
  Longlinks_Red.SenderOffset = make_offset<int>(Longlinks_Red.Sender, 0, static_cast<int>(Longlinks.SenderOffset.size())-1);

  vTimeBreakDown.at(AL_MISC) += ElTime();
  
  double dTimeTotal = 0;
  for (int xx = 0; xx < 6; xx++)
    dTimeTotal += vTimeBreakDown.at(xx);

  std::cout << std::endl;
  std::cout << std::setfill('-') << std::setw(60)<<"-" << std::endl;
  std::cout << std::setfill(' ') << std::setw(40) << rp() + "Total time taken by activate_long_links: " << std::setprecision(4) << dTimeTotal << " sec." << std::endl;
  std::cout << std::setprecision(4) << std::left << std::setw(15) << rp() + "Case1 :  " << std::setw(15) << vTimeBreakDown.at(AL_CASE1) << " sec. (" << vTimeBreakDown.at(AL_CASE1) / dTimeTotal * 100.0 << "%)" << std::endl;
  std::cout << std::setprecision(4) << std::left << std::setw(15) << rp() + "Case2a:  " << std::setw(15) << vTimeBreakDown.at(AL_CASE2A) << " sec. (" << vTimeBreakDown.at(AL_CASE2A) / dTimeTotal * 100.0 << "%)" << std::endl;
  std::cout << std::setprecision(4) << std::setw(15) << rp() + "Case2b:  " << std::setw(15) << vTimeBreakDown.at(AL_CASE2B) << " sec. (" << vTimeBreakDown.at(AL_CASE2B) / dTimeTotal * 100.0 << "%)" << std::endl;
  std::cout << std::setprecision(4) << std::setw(15) << rp() + "Case2c:  " << std::setw(15) << vTimeBreakDown.at(AL_CASE2C) << " sec. (" << vTimeBreakDown.at(AL_CASE2C) / dTimeTotal * 100.0 << "%)" << std::endl;
  std::cout << std::setprecision(4) << std::setw(15) << rp() + "Case2I:  " << std::setw(15) << vTimeBreakDown.at(AL_CASE2I) << " sec. (" << vTimeBreakDown.at(AL_CASE2I) / dTimeTotal * 100.0 << "%)" << std::endl;
  std::cout << std::setprecision(4) << std::setw(15) << rp() + "Misc  :  " << std::setw(15) << vTimeBreakDown.at(AL_MISC) << " sec. (" << vTimeBreakDown.at(AL_MISC) / dTimeTotal * 100.0 << "%)" << std::endl;
  std::cout << std::setfill('-') << std::setw(60)<<"-" << std::endl;
  std::cout << std::endl;

  function_switch("activate_long_links");
  return Longlinks_Red;

}
