#include <mpi.h>

//#include "init.h"
//#include "globals.h"


#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/utilities_mpi.h"


// **************************************************************************************
// Main function: search_katamaran()
// Performs "Katamaran-style" comparison between two sorted vectors to identify matches
// Returns number of successfully matched entries
// **************************************************************************************

// N.B.: the "sort-vectors" are IDL-style index arrays into the respective vectors.
// I.e.: sortvec[0] contains the index of the smallest (usually) entry of the vector, 
// sortvec[1] of the next-smallest, and so on. 
// To loop through vector in sorted order, access vector[sortvec[ii]].

// N.B. II: The returned "match-vector" [ii] contains the index *in the (hypothetical) full target
//          ID list* of particle ii in the source ID vector. This is why PartOffset_B needs to be 
//          provided -- otherwise position in this full vector would be unknown.

// N.B. III: Update 26 Apr 2016 -- Added option to return tasks where matches are found.
//                                 Also added option to make returned match vector LOCAL, i.e.
//                                 to switch off the behaviour specified in N.B. II above.

template <typename T>
long search_katamaran(const std::vector<unsigned long> &vTaskIDs_A,       // Source IDs [original order]
		      const std::vector<long> &vSortedIndices_A,  // Sort-vector for source IDs
		      
		      std::vector<unsigned long> &vTaskIDs_B,       // Target IDs [original order]
		      std::vector<long> &vSortedIndices_B,  // Sort-vector for target IDs

		      std::vector<T> &vMatchInds, // [O] Match-vector (see above) 
		      std::vector<int> &vMatchTasks, // [O] Match tasks (may be dummy) 
		      long PartOffset_B,            // Offset of B part ID list
		      int nFlagReturnTasks,      // If 1, match tasks are returned
		      int nFlagLocalIndices)     // If 1, indices returned are LOCAL, not global

{

  function_switch("search_katamaran");
  using namespace std;

  long nMatches = 0;
  
  double dDummy = ElTime();
  double dStartTime = GetTime();

  if (mpirank() == 0)
    cout << rp() + "Beginning search_katamaran()..." << endl;
  
#ifdef VERBOSE
  if (mpirank() == 0) {
    cout << rp() + "ID A0 = " << vTaskIDs_A.at(0) << endl;
    cout << rp() + "ID B0 = " << vTaskIDs_B.at(0) << endl;
  }
#endif
  
  long size_a = vTaskIDs_A.size();
  long size_b = vTaskIDs_B.size();

#ifdef VERBOSE
  cout << rp() + "Vector A (search) has " << size_a << " elements." << endl;
  cout << rp() + "Vector B (match)  has " << size_b << " elements." << endl;
  cout << endl;

  cout << rp() + "Inds A (search) has " << vSortedIndices_A.size() << " elements." << endl;
  cout << rp() + "Inds B (match)  has " << vSortedIndices_B.size() << " elements." << endl;
#endif
  

  // =========================================
  // Now, loop through B-passing iterations...
  // =========================================

  vMatchInds.clear();
  vMatchInds.resize(vTaskIDs_A.size(),-1); // Initialise match vector to -1 [no match]

  if (nFlagReturnTasks == 1) {
    vMatchTasks.clear();
    vMatchTasks.resize(vTaskIDs_A.size(),-1); // Likewise, initialise to -1 [no match]
  }
    
#ifdef VERBOSE
  cout << rp() + "Reached barrier at beginning of Katamaran run. Waiting..." << endl;
#endif

  // Put barrier [should be there -- strange behaviour can happen otherwise...]
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Status stats[8];
  MPI_Request reqs[8];
  
  for (int jj=0;jj<numtasks();jj++)
    {
      if (mpirank() == 0)
	print_n_times(jj, numtasks(), 10, "Katamaran run");

#ifdef REPORT_TIME
      double dStartIter = GetTime();
#endif 
      
#ifdef VERBOSE
      cout << rp() + "Starting Katamaran run " << jj << "..." << endl;
#endif
      
      int stoploop = 0;
      long ind_a = 0, ind_b = 0; // Index in SEARCH and TARGET vector, respectively

      // Which (TARGET) block this task is currently working on
      // N.B.: blocks get passed upwards, so task receives successively LOWER blocks      
      int currblock = mpirank()-jj; 
      if (currblock < 0)
	currblock += numtasks();
      
      // Now set up katamaran:

      // Find size of B vector. Important to update this for EVERY iteration!
      // Needs to be an explicit variable for MPI-communication
      size_b = static_cast<long>(vTaskIDs_B.size()); 
      
      if (size_a > 0 and size_b > 0)
	{
	  long sortLocA = vSortedIndices_A.at(ind_a);
	  long val_a = vTaskIDs_A.at(sortLocA);
	  
	  long sortLocB = vSortedIndices_B.at(ind_b);
	  long val_b = vTaskIDs_B.at(sortLocB);
	  
	  
#ifdef VERBOSE
	  cout << rp() + "Starting val_a = " << val_a << endl;
	  cout << rp() + "Starting val_b = " << val_b << endl;
#endif
      
      

#ifdef VERBOSE
      cout << rp() + "   Beginning search loop (size_b = " << size_b << ", size_a = " << size_a << ")" << endl;
      cout << rp() + "   Size of vTaskIDs_A = " << vTaskIDs_A.size() << endl;
      cout << rp() + "   Size of vTaskIDs_B = " << vTaskIDs_B.size() << endl;
      cout << rp() + "   Size of vSortedIndices_A = " << vSortedIndices_A.size() << endl;
      cout << rp() + "   Size of vSortedIndices_B = " << vSortedIndices_B.size() << endl;
#endif

      // Now do the actual katamaran search:
      do {
	
	// Source value (still) below target value? Next source index.
	if (val_a < val_b) {
	  ind_a++;
	  if (ind_a >= size_a)
	    break;
	  sortLocA = vSortedIndices_A[ind_a]; 
	  val_a = vTaskIDs_A[sortLocA];
	}

	// Source value above target? Next target index.
	else if (val_a > val_b) {
	  ind_b++;
	  if (ind_b >= size_b)
	    break;
	  sortLocB = vSortedIndices_B[ind_b];
	  val_b = vTaskIDs_B[sortLocB];
	}

	// Match!?! Great! Update both.
	else if (val_a == val_b)
	  {
	    if (nFlagLocalIndices == 0)
	      vMatchInds.at(sortLocA) = static_cast<T>(sortLocB + PartOffset_B);
	    else
	      vMatchInds.at(sortLocA) = static_cast<T>(sortLocB);
	    
	    if (nFlagReturnTasks == 1)
	      vMatchTasks.at(sortLocA) = currblock;

	    ind_a++;
	    ind_b++;
	    if (ind_b >= size_b || ind_a >= size_a)
	       break;
	    sortLocA = vSortedIndices_A[ind_a];
	    val_a = vTaskIDs_A[sortLocA];
	    
	    sortLocB = vSortedIndices_B[ind_b];
	    val_b = vTaskIDs_B[sortLocB];
	    
	    nMatches++;
	  }

	else 
	  cout << "How can this ever be reached??!!??" << endl;
	
      } while (1);

#ifdef REPORT_TIME
      double dMidIter = GetTime();
#endif

      // -------------------------------------------------------------
      // -------------- We've ended the search loop now --------------
      // ------- Need to pass on the search vector to next task!! ----
      // -------------------------------------------------------------

	} // ends section only if both A and B vector are non-zero size
      
#ifdef VERBOSE
      cout << rp() + "Ended search loop " << jj << ", preparing to exchange data..." << endl;
#endif
      
      // Variables for next block to be processed
      long size_b_next = 0;
      long PartOffset_B_next = 0;

      // Work out tasks to exchange B-data with now:
      int rectask = (mpirank()+1) % numtasks();    // Increases, so never negative
      int sendtask = (mpirank()-1);
      if (sendtask < 0)
	sendtask += numtasks();
      
      
      // -------- Send data ---------

      // Metadata (size of currently hosted vector, and its offset in the virtual full list)
      MPI_Isend(&size_b, 1, MPI_LONG, rectask, 1000, MPI_COMM_WORLD, &reqs[0]);
      MPI_Isend(&PartOffset_B, 1, MPI_LONG, rectask, 1001, MPI_COMM_WORLD, &reqs[1]);

      // Actual data (IDs and sort-vector)
      if (size_b > 0) {
	MPI_Isend(&vTaskIDs_B.front(), size_b, MPI_LONG, rectask, 1002, MPI_COMM_WORLD, &reqs[2]);
	MPI_Isend(&vSortedIndices_B.front(), size_b, MPI_LONG, rectask, 1003, MPI_COMM_WORLD, &reqs[3]);
      }

#ifdef VERBOSE
      cout << rp() + "    Sent my ID and sort list..." << endl;
#endif
      

      // ------- Receive data ----------

      // Metadata first:
      MPI_Irecv(&size_b_next, 1, MPI_LONG, sendtask, 1000, MPI_COMM_WORLD, &reqs[4]);
      MPI_Irecv(&PartOffset_B_next, 1, MPI_LONG, sendtask, 1001, MPI_COMM_WORLD, &reqs[5]);

      MPI_Wait(&reqs[4], &stats[4]);   // Need to know metadata before receiving actual data
      MPI_Wait(&reqs[5], &stats[5]);
      
#ifdef VERBOSE
      cout << rp() + "   Received new list number [" << size_b_next << "] and PartOffset_B [" << PartOffset_B_next << "]" << endl;
#endif

      vector<long> vTaskIDs_B_new(size_b_next);
      vector<long> vSortedIndices_B_new(size_b_next);
      
#ifdef VERBOSE
      cout << rp() + "   Waiting to receive " << size_b_next << " data elememts... " << endl;
#endif
      
      if (size_b_next > 0) {
	MPI_Irecv(&vTaskIDs_B_new.front(), size_b_next, MPI_LONG, sendtask, 1002, MPI_COMM_WORLD, &reqs[6]);
	MPI_Irecv(&vSortedIndices_B_new.front(), size_b_next, MPI_LONG, sendtask, 1003, MPI_COMM_WORLD, &reqs[7]);
      }

      if (size_b > 0) {
	MPI_Wait(&reqs[2], &stats[2]); // wait until we are sure that the old data has sent
	MPI_Wait(&reqs[3], &stats[3]); // wait until we are sure that the old data has sent
      }
      
      if (size_b_next > 0) {
	MPI_Wait(&reqs[6], &stats[6]); // Also wait until data is actually received (!!)
	MPI_Wait(&reqs[7], &stats[7]); // Also wait until data is actually received (!!)
      }

#ifdef VERBOSE
      cout << rp() + "   ...data reception complete." << endl;
#endif

      // Now move data from temporary to permanent vector:
      vTaskIDs_B.resize(size_b_next);
      vSortedIndices_B.resize(size_b_next);
      
#ifdef VERBOSE
      cout << rp() + "   Finished resizing B-vectors to size " << size_b_next << endl;
      cout << rp() + "   Populating the new vTaskIDs_B vector..." << endl;
#endif

      for (long kk=0; kk<size_b_next; kk++) {
       	vTaskIDs_B.at(kk)=vTaskIDs_B_new.at(kk);
      	vSortedIndices_B.at(kk)=vSortedIndices_B_new.at(kk); }

#ifdef VERBOSE
      cout << rp() + "De-allocating the intermediate buffer TasksB_next..." << endl;
#endif 
      
      vTaskIDs_B_new.clear();
      vSortedIndices_B_new.clear();
      
      MPI_Wait(&reqs[1], &stats[1]); // Make sure old Offset_B has sent
      PartOffset_B = PartOffset_B_next;
      
#ifdef VERBOSE
      cout << rp() + "   My run " << jj << " is finished, waiting for other tasks..." << endl;
#endif
      
#ifdef REPORT_TIME
      double dEndIter = GetTime();
#endif
      
#ifdef BARRIER_AT_KATAMARAN_LOOPEND
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      
#ifdef REPORT_TIME
      report<double>(dMidIter-dStartIter, "Search part: ");
      report<double>(dEndIter-dMidIter,   "Send part  : ");
      report<double>(GetTime()-dEndIter,  "Waiting    : ");
      report<double>(dAllocTime, "Allocation:");
#endif
      
#ifdef VERBOSE
      if (mpirank() == 0)
	cout << rp() + "   Finished comparison iteration " << jj << " (" << ElTime() << " sec.) " << endl;
#endif      

    } // ends loop through MATCH vectors

#ifdef VERBOSE
  if (mpirank() == 0)
    cout << rp() + "Particle tracing finished -- took " << GetTime() - dStartTime << " sec. " << endl;
#endif  

  function_switch("search_katamaran");
  return nMatches;
  
}

