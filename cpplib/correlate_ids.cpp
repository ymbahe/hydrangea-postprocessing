// correlate_ids.cpp
// This file contains routines to correlate IDs between two input vectors
// The FUNDAMENTAL assumption is that the IDs are relatively dense, i.e. that
// no crazily high ID values occur. Otherwise, proceed at your own risk...

#include <mpi.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

//#include "/u/ybahe/ANALYSIS/Protea-H/Config.h"
//#include "/u/ybahe/ANALYSIS/Protea-H/globals.h"

#include "/u/ybahe/cpplib/correlate_ids.h"
#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/io_hdf5.h"



#ifdef OLD_BUILD_MEDIATOR_LIST
// **************************************************************************************
// Main function: build_mediator_list()
// This constructs the (distributed) translation table from the input ID list
// For convenience, it also returns a list specifying which task each ID comes from
// **************************************************************************************

void build_mediator_list(const std::vector<long> &vTaskIDs,         // Source IDs on this task
			 
			 std::vector<long> &vML_Indices,       // [O] Transtable chunk
			 std::vector<int> &vML_OrigTasks,     // [O] Task of origin
			 
			 long PartOffset)            // Offset of this task's ID list chunk

{
  function_switch("build_mediator_list");
  using namespace std;

  double dDummy = ElTime();
  double dStartTime = GetTime();

  //
  // The communication program distributed_exchange() needs THREE pieces of information
  // (ie. three input vectors, all of equal length):
  //
  // (i)   The 'data' to go into the to-be-built list
  // (ii)  Which chunk each piece of data will be moved to
  // (iii) Which position in its chunk the data will be moved to
  // 
  // As output vectors, distributed_exchange() needs two vectors:
  // (a) The to-be-built list
  // (b) The task-source list (building this is optional)
  //
  // Here, the three input vectors are:
  // (i)    Index position in fiducial full list of a particular ID --> vForMLIndices
  // (ii)   ID value / chunk_size --> vForMLTask
  // (iii)  ID value % chunk_size --> vForMLPosition
  //

  int chunk_size = mediator_chunk_size();
  int nTaskIDs = vTaskIDs.size();

  vector<long> vForMLIndices(nTaskIDs);
  vector<int> vForMLTask(nTaskIDs);
  vector<int> vForMLPosition(nTaskIDs);

  for (int ii = 0; ii < nTaskIDs; ii++) {
    vForMLIndices.at(ii) = static_cast<long>(ii) + PartOffset;
    vForMLTask.at(ii) = static_cast<int>(vTaskIDs.at(ii) / chunk_size);
    vForMLPosition.at(ii) = static_cast<int>(vTaskIDs.at(ii) % chunk_size);
  }

  // Prepare the 'output' vectors - must be sized & initialized
  vML_Indices.clear();
  vML_Indices.resize(chunk_size, -1);
  vML_OrigTasks.clear();
  vML_OrigTasks.resize(chunk_size, -1);

  // Now call the exchange function - last '1' means that 
  // we do want the original tasks kept track of
  // (because this will be required later)

  distributed_exchange(vForMLIndices, vForMLTask, vForMLPosition, 
		       vML_Indices, vML_OrigTasks, 1);
  
  function_switch("build_mediator_list");  
  return;

}
#endif


// **************************************************************************************
// Main function: correlate_ids()
// This exploits the (already constructed) mediator lists of TWO DIFFERENT snapshots
// It builds a new list saying where in snapshot B each particle from snapshot A lies.
// **************************************************************************************

long correlate_ids(const std::vector<long> &vML_IndicesA,     // Indices in snapshot A
		   const std::vector<int> &vML_OrigTasksA,    // Tasks in snapshot A
		   const std::vector<long> &vPartOffsetA,     // Particle offsets in A
		   const std::vector<long> &vML_IndicesB,     // Indices in snapshot B
		   std::vector<long> &vFullIndexInB,         // [O] Indices in B matched to A
		   int FlagInputIsLocal) {  // def=0, if 1: vML_IndicesA is local, not global. Nature of vML_IndicesB is irrelevant as it's just passed on, so outtype = intype. 
  
  function_switch("correlate_ids");
  using namespace std;
  
  double dDummy = ElTime();
  double dStartTime = GetTime();
  
  // N.B.: vFullIndexInB must already be sized correctly, 
  //       i.e. to same length as vTaskIDs

  vector<int> vTaskInB_Dummy;

  // For exchange, we need the LOCAL A-indices.
  // If the input is global (FlagInputIsLocal == 0), we need to convert
  // it by subtracting its target-task's (vML_OrigTasksA) offset.
  std::vector<long> vDestsInA = localize_indices(vML_IndicesA, vML_OrigTasksA, vPartOffsetA, FlagInputIsLocal);

  // Do actual correlation by re-purposing distributed_exchange function.
  // Effectively, it uses the IDs to send the corresponding index in B to 
  // the the index in A.
    
  distributed_exchange(vML_IndicesB, vML_OrigTasksA, vDestsInA,
		       vFullIndexInB, vTaskInB_Dummy, 0);
  
  // Final bit: determine how many particles could be matched:
  // i.e., how many have vFullIndexInB >= 0

  long nMatches = 0;
  for (int ii=0; ii < vFullIndexInB.size(); ii++) {
    if (vFullIndexInB.at(ii) >= 0)
      nMatches++;
  } 

  function_switch("correlate_ids");
  
  return nMatches;
} 


#ifdef OLD_DISTRIBUTED_EXCHANGE
// *****************************************
// Helper function: distributed_exchange()
// It exchanges particle info between tasks
// *****************************************

void distributed_exchange(const std::vector<long> &vData,     // Data to be exchanged 
			  const std::vector<int> &vDestTask,  // Destination task
			  const std::vector<int> &vDestInd,   // Destination index
			  std::vector<long> &vOutList,        // [O] List to be built
			  std::vector<int> &vSourceTasks,     // [O] Where data is from
			  int FlagConstructSourceTasks) {     // If !=1, skip source tasks


  using namespace std;
  int rc = 0;

  function_switch("distributed_exchange");
  
  // 1.) Create an IDL-style histogram of the destination tasks
  // 2.) Transpose exchange counts between tasks with MPI_Alltoall
  // 3.) Fill in output from self
  // 4.) Loop through separations and exchange info with other tasks
  

  // 1.) Create an IDL-style histogram of the destination tasks
  int nMin = 0, nMax = numtasks()-1;
  vector<int> vTaskOffset, vTaskRI;
  vector<int> vTaskCountsSend = idlhist<int, int>(vDestTask, 
						  vTaskOffset, 
						  vTaskRI, 
						  nMin, nMax, 0, 0);

  // 2.) Transpose exchange counts between tasks with MPI_Alltoall
  vector<int> vTaskCountsRecv(numtasks());
  rc = MPI_Alltoall(&vTaskCountsSend.front(), 1, MPI_INT, 
		    &vTaskCountsRecv.front(), 1, MPI_INT,
		    MPI_COMM_WORLD);

  // 3.) Fill in output from self
  int NumXCurr = vTaskCountsSend.at(mpirank());
  if (NumXCurr > 0) {
    int nCurrTaskOffset = vTaskOffset.at(mpirank());

    for (int ii = 0; ii < NumXCurr; ii++) {
      int nCurrParticle = vTaskRI.at(ii+nCurrTaskOffset);
      long nCurrData = vData.at(nCurrParticle);
      int nCurrDestInd = vDestInd.at(nCurrParticle);

      vOutList.at(nCurrDestInd) = nCurrData;
      if (FlagConstructSourceTasks == 1)
	vSourceTasks.at(nCurrDestInd) = mpirank();
    }
  } // Ends section to fill own info into output list
  
  
  // 4.) Loop through separations and exchange info with other tasks
  for (int iitask = 1; iitask < numtasks(); iitask++) {

    int nSendToTask = mpirank() - iitask;
    if (nSendToTask < 0)
      nSendToTask += numtasks();

    int nRecvFromTask = mpirank() + iitask;
    if (nRecvFromTask >= numtasks())
      nRecvFromTask -= numtasks();

    MPI_Request reqs[4];
    MPI_Status stats[4];
    
    // (a) --> SEND data to other task
    int nCurrSend = vTaskCountsSend.at(nSendToTask);

    if (nCurrSend > 0) {

      int nCurrTaskOffset = vTaskOffset.at(nSendToTask);
      vector<int> vSendBuf_Loc(nCurrSend);
      vector<long> vSendBuf_Data(nCurrSend);
      
      // Make temporary arrays holding data to be sent:
      // Locations in other task, and data
      
      for (int ii = 0; ii < nCurrSend; ii++) {
	int nCurrParticle = vTaskRI.at(nCurrTaskOffset + ii);
	int nCurrXLoc = vDestInd.at(nCurrParticle); 
	long nCurrData = vData.at(nCurrParticle);
	vSendBuf_Loc.at(ii) = nCurrXLoc;
	vSendBuf_Data.at(ii) = nCurrData;
      }
      
      rc = MPI_Isend(&vSendBuf_Loc.front(), nCurrSend, MPI_INT, 
		     nSendToTask, 7775, MPI_COMM_WORLD, &reqs[0]);       

      rc = MPI_Isend(&vSendBuf_Data.front(), nCurrSend, MPI_LONG, 
		     nSendToTask, 7776, MPI_COMM_WORLD, &reqs[1]);       
    
      // It might be possible to optimize this by not waiting here,
      // but for now this seems the safest thing to do...
      MPI_Wait(&reqs[0], &stats[0]);     
      MPI_Wait(&reqs[1], &stats[1]);     
      
    } // ends section to SEND data to other task 

    // (b) --> RECEIVE data from other task
    int nCurrRecv = vTaskCountsRecv.at(nRecvFromTask);

    if (nCurrRecv > 0) {
      
      vector<int> vRecvBuf_Loc(nCurrRecv);
      vector<long> vRecvBuf_Data(nCurrRecv);
      
      rc = MPI_Irecv(&vRecvBuf_Loc.front(), nCurrRecv, MPI_INT, 
		     nRecvFromTask, 7775, MPI_COMM_WORLD, &reqs[2]);

      rc = MPI_Irecv(&vRecvBuf_Data.front(), nCurrRecv, MPI_LONG, 
		     nRecvFromTask, 7776, MPI_COMM_WORLD, &reqs[3]);
      
      // Need to make sure data is received before processing it...
      MPI_Wait(&reqs[2], &stats[2]);     
      MPI_Wait(&reqs[3], &stats[3]);     

      for (int ii = 0; ii < nCurrRecv; ii++) {
	
	int nCurrDestInd = vRecvBuf_Loc.at(ii);
	long nCurrData = vRecvBuf_Data.at(ii);
	
	vOutList.at(nCurrDestInd) = nCurrData;
	if (FlagConstructSourceTasks == 1)
	  vSourceTasks.at(nCurrDestInd) = nRecvFromTask;
	
      }
    } // ends section to RECEIVE data from other task
    
  } // ends loop through other tasks to exchange data
  
  // We are done!
  
  function_switch("distributed_exchange");  
  return;
  
}
#endif			  


// HELPER function: calculate/provide mediator chunk size
// this is only calculated ONCE in the program, and afterwards
// just returned


int mediator_chunk_size(std::string sSnepDir,
			std::string sRunType,
			long nParts) {

  static int first_time = 1;
  static int mediator_chunk_size = -1;

  if (first_time == 1)
    {
      long glob_max_id; //= g_nMaxPartNumber;   //find_global_maximum(vTaskIDs);

      // Long-term, this should probably be replaced with some more
      // fancy function that actually determines the maximum ID...
      // For now, just work with the built-in standard of 1e9.

      if (nParts > 0)
	glob_max_id = nParts;

      else 
	{

	  long nPartT1 = read_hdf5_attribute_long(sSnepDir, "Header/NumPart_Total", 'g', 1);
	  
	  if (sRunType.compare("Hydro") == 0)
	    glob_max_id = (nPartT1+1)*2;
	  else if (sRunType.compare("DM") == 0)
	    glob_max_id = (nPartT1+1);
	  else
	    std::cout << "Unexpected RunType encountered: '" << sRunType << "'..." << std::endl;
	}

      mediator_chunk_size = static_cast<int>(glob_max_id / numtasks() + 1);

      first_time = 0;
    }

  return mediator_chunk_size;

}

#ifdef OLD_MEDIATOR_CHUNK_SIZE
// HELPER function: calculate/provide mediator chunk size
// this is only calculated ONCE in the program, and afterwards
// just returned

int mediator_chunk_size(void) {

  static int first_time = 1;
  static int mediator_chunk_size = -1;

  if (first_time == 1)
    {
      long glob_max_id = g_nMaxPartNumber;   //find_global_maximum(vTaskIDs);

      // Long-term, this should probably be replaced with some more
      // fancy function that actually determines the maximum ID...
      // For now, just work with the built-in standard of 1e9.

      mediator_chunk_size = static_cast<int>(glob_max_id / numtasks() + 1);

      first_time = 0;
    }

  return mediator_chunk_size;

}
#endif


void pull_to_multipush(const std::vector<int> &vReqTask, 
		       const std::vector<long> &vReqIndex,
		       std::vector<int> &vPushTask,
		       std::vector<long> &vPushInd,
		       std::vector<long> &vPushOffset,
		       long nNumElem)
{
  
  // 1.) Create self-addressed envelopes
  
  long nReqs = vReqIndex.size();
   
  std::vector<long> vLocalInds(nReqs);
  for (long ii = 0; ii < nReqs; ii++)
    vLocalInds.at(ii) = ii;
  
  // 2.) Send these to ReqTask/Index.
  //     N.B.: in contrast to distributed_exchange, multiRecvPush clears 
  //           the output vectors and builds them from scratch
  
  distributed_multiRecvPush(vLocalInds, vReqTask, vReqIndex, vPushInd, vPushTask, vPushOffset, nNumElem, 1);
  
  return;
}
