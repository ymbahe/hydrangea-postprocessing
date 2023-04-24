
#include "/u/ybahe/cpplib/utilities.h"


// **************************************************************************************
// Main function: build_mediator_list()
// This constructs the (distributed) translation table from the input ID list
// For convenience, it also returns a list specifying which task each ID comes from
// **************************************************************************************

template <typename T>
void build_mediator_list(const std::vector<unsigned long> &vTaskIDs,         // Source IDs on this task
			 
			 std::vector<T> &vML_Indices,       // [O] Transtable chunk
			 std::vector<int> &vML_OrigTasks,     // [O] Task of origin
			 
			 std::string sSnepshotDir,  // Dir from which to read max ID
			 std::string sRunType,      // RunType of sim

			 long PartOffset,            // Offset of this task's ID list chunk
			 int FlagIndexType, // 0:full[=default], 1:local
			 long nParts)       // default=0, pass on to mediator_chunk_size()
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

  // This makes the mediator index LOCAL
  if (FlagIndexType == 1)
    PartOffset = 0;
    
  int chunk_size = mediator_chunk_size(sSnepshotDir, sRunType, nParts);
  
  // Sanity check
  if (chunk_size < 0) {
    std::cout << "Chunk size specified as " << chunk_size << ", but must be positive!" << std::endl;
    exit(777);
  }

  if (mpirank() == 0)
    std::cout << rp() + "Using ML chunk size of " << chunk_size << "..." << std::endl;
  

  int nTaskIDs = vTaskIDs.size();

  vector<T> vForMLIndices(nTaskIDs);
  vector<int> vForMLTask(nTaskIDs);
  vector<long> vForMLPosition(nTaskIDs);

  for (int ii = 0; ii < nTaskIDs; ii++) {
    vForMLIndices.at(ii) = static_cast<T>(ii) + PartOffset;
    vForMLTask.at(ii) = static_cast<int>(vTaskIDs.at(ii) / chunk_size);
    vForMLPosition.at(ii) = static_cast<long>(vTaskIDs.at(ii) % chunk_size);
  }

  // Prepare the 'output' vectors - must be sized & initialized
  vML_Indices.clear();
  vML_Indices.resize(chunk_size, -1);
  vML_OrigTasks.clear();
  vML_OrigTasks.resize(chunk_size, -1);

  // Now call the exchange function - last '1' means that 
  // we do want the original tasks kept track of
  // (because this will be required later)

  distributed_exchange<T>(vForMLIndices, vForMLTask, vForMLPosition, 
			  vML_Indices, vML_OrigTasks, 1);
  
  function_switch("build_mediator_list");  
  return;

}

// *****************************************
// Helper function: distributed_exchange()
// It exchanges particle info between tasks
// *****************************************

template <typename T>
void distributed_exchange(const std::vector<T> &vData,     // Data to be exchanged 
			  const std::vector<int> &vDestTask,  // Destination task
			  const std::vector<long> &vDestInd,   // Destination index
			  std::vector<T> &vOutList,        // [O] List to be built
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
      T nCurrData = vData.at(nCurrParticle);
      long nCurrDestInd = vDestInd.at(nCurrParticle);

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

    int nCurrSend = vTaskCountsSend.at(nSendToTask);
    vector<long> vSendBuf_Loc(nCurrSend);
    vector<T> vSendBuf_Data(nCurrSend);

    // Make temporary arrays holding data to be sent:
    // Locations in other task, and data

    int nCurrTaskOffset = vTaskOffset.at(nSendToTask);
    for (int ii = 0; ii < nCurrSend; ii++) {
      int nCurrParticle = vTaskRI.at(nCurrTaskOffset + ii);
      long nCurrXLoc = vDestInd.at(nCurrParticle); 
      T nCurrData = vData.at(nCurrParticle);
      vSendBuf_Loc.at(ii) = nCurrXLoc;
      vSendBuf_Data.at(ii) = nCurrData;
    }


    int nCurrRecv = vTaskCountsRecv.at(nRecvFromTask);
    vector<long> vRecvBuf_Loc(nCurrRecv);
    vector<T> vRecvBuf_Data(nCurrRecv);


    MPI_Sendrecv(&vSendBuf_Loc.front(), nCurrSend, MPI_LONG, nSendToTask, 7775,
		 &vRecvBuf_Loc.front(), nCurrRecv, MPI_LONG, nRecvFromTask, 7775,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&vSendBuf_Data.front(), nCurrSend*sizeof(T), MPI_BYTE, nSendToTask, 7776,
		 &vRecvBuf_Data.front(), nCurrRecv*sizeof(T), MPI_BYTE, nRecvFromTask, 7776,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    
    for (int ii = 0; ii < nCurrRecv; ii++) {
	
      long nCurrDestInd = vRecvBuf_Loc.at(ii);
      T nCurrData = vRecvBuf_Data.at(ii);
	
      vOutList.at(nCurrDestInd) = nCurrData;
      if (FlagConstructSourceTasks == 1)
	vSourceTasks.at(nCurrDestInd) = nRecvFromTask;
      
    }

  } // ends loop through other tasks to exchange data
  
  // We are done!
  
  function_switch("distributed_exchange");  
  return;
  
}


template <typename T>
std::vector<T> localize_indices(std::vector<T> vIndices,
	       	                std::vector<int> vTasks,
  				std::vector<long> vOffsets,
				int FlagIndicesAreLocal)
{
		
  size_t chunk_size = vIndices.size();
  std::vector<T> vLocalIndices(chunk_size, -1);
  
  for (size_t ii = 0; ii < chunk_size; ii++) {
    if (vTasks.at(ii) < 0)
      continue;
    
    T nLocalInd = vIndices.at(ii);
    if (FlagIndicesAreLocal == 0)
      nLocalInd -= vOffsets.at(vTasks.at(ii)); 
	
    if (nLocalInd < 0) {
      std::cout << "Inconsistent particle destination encountered in localize_indices." << std::endl;
      exit(77); 
    }
    
    vLocalIndices.at(ii) = nLocalInd;
  }

  return vLocalIndices;

}


template <typename T>
std::vector<T> globalize_indices(std::vector<T> vIndices,
				 std::vector<int> vTasks,
				 std::vector<long> vOffsets,
				 int FlagIndicesAreLocal)
{
		
  size_t chunk_size = vIndices.size();
  std::vector<T> vGlobalIndices(chunk_size, -1);
  
  for (size_t ii = 0; ii < chunk_size; ii++) {
    if (vTasks.at(ii) < 0)
      continue;
    
    T nGlobalInd = vIndices.at(ii);
    if (FlagIndicesAreLocal == 1)
      nGlobalInd += vOffsets.at(vTasks.at(ii)); 
    
    vGlobalIndices.at(ii) = nGlobalInd;
  }

  return vGlobalIndices;

}


template <typename T>
void unpack_global_indices(const std::vector<T> &vGlobalIndices,
			   const std::vector<long> &vTPOlist,
			   std::vector<int> &vTasks,           // [O] tasks
			   std::vector<T> &vLocalIndices) // [0] loc. inds

{

  std::vector<long> vSorter = sort_indices(vGlobalIndices);
  vLocalIndices.resize(vGlobalIndices.size());
  vTasks.resize(vGlobalIndices.size());

  size_t ii_task = 0;
  size_t ii_ind = 0;

  while(1) {

    if (ii_task >= vTPOlist.size()-1)
      break;
    if (ii_ind >= vGlobalIndices.size())
      break;

    // Next part is necessary to deal with possibility of -1 inputs
    // (indicating 'no match, anywhere')
    if (vGlobalIndices.at(vSorter.at(ii_ind)) < vTPOlist.at(ii_task)) {
      vTasks.at(vSorter.at(ii_ind)) = -1;
      ii_ind++;
      continue;
    }

    if (vGlobalIndices.at(vSorter.at(ii_ind)) >= vTPOlist.at(ii_task+1)) {
      ii_task++;
      continue;
    }

    vTasks.at(vSorter.at(ii_ind)) = ii_task;
    vLocalIndices.at(vSorter.at(ii_ind)) = vGlobalIndices.at(vSorter.at(ii_ind)) - vTPOlist.at(ii_task);
    ii_ind++;
    
  }

  return;

}


template <typename T> 
void distributed_lookup(const std::vector<unsigned long> &vTT_IDs, // IDs to be matched
			const std::vector<int> &vML_tasks,         // Source tasks
			const std::vector<T> &vML_indices,         // (Local) inds @ source 
			std::vector<int> &vTT_MatchTasks,     // [O] Matched tasks
			std::vector<T> &vTT_MatchInds)        // [O] Matched inds


{
  function_switch("distributed_lookup");
  
  // Plan for this program:
  // 1.) Translate input ID into task+index (='address') on mediator list
  // 2.) Send index array ('self-addressed envelope') to these addresses
  // 3.) Reply to queries from (2) by sending vML_Indices and vML_Tasks back

  // --------------------------------------------------------------------

  size_t nIDs = vTT_IDs.size();
  size_t nIDs_Ref = vML_indices.size();

  // 1.) Translate ID --> ML task + ML index
  std::vector<long> vTT_ID_ML_indices;
  std::vector<int> vTT_ID_ML_tasks;
  decompose_IDs<T>(vTT_IDs, vTT_ID_ML_tasks, vTT_ID_ML_indices);

  // 2.) Create and send 'self-addressed envelope' to addresses just found:
  std::vector<long> vLocalInds(nIDs);
  for (long ii = 0; ii < nIDs; ii++)
    vLocalInds.at(ii) = ii;

  std::vector<long> vReply_inds;  // automatically sized inside
  std::vector<int> vReply_tasks;  // multiRecvPush
  std::vector<long> vReply_offsets;
  distributed_multiRecvPush(vLocalInds, vTT_ID_ML_tasks, vTT_ID_ML_indices, vReply_inds, vReply_tasks, vReply_offsets, nIDs_Ref, 1);

  // 3.) Now send local ML indices/tasks to just-built 'reply addresses':
  
  std::vector<int> vSourceTasksCheck(nIDs, -1); 
  distributed_multiSendPush(vML_indices, vReply_tasks, vReply_inds, vReply_offsets, vTT_MatchInds, vSourceTasksCheck, 1);
  distributed_multiSendPush(vML_tasks, vReply_tasks, vReply_inds, vReply_offsets, vTT_MatchTasks, vSourceTasksCheck, 1);

  // And we're done!

  // Final consistency check:
  for (size_t ii = 0; ii < nIDs; ii++) {
    if (vSourceTasksCheck.at(ii) != vTT_ID_ML_tasks.at(ii)) {
      std::cout << "In function `" << __func__ << "', file " << __FILE__ << ", line " << __LINE__ << ": source task check failed for element " << ii << " (expected " << vTT_ID_ML_tasks.at(ii) << ", but obtained " << vSourceTasksCheck.at(ii) << ")" << std::endl;
      exit(10571);
    }
      
  }
  
  function_switch("distributed_lookup");
  return;  
}


template <typename T>
void distributed_pull(const std::vector<T> &vData,  // data to be exchanged
		      const std::vector<int> &vReqTask, // index to pull from
		      const std::vector<long> &vReqIndex, // task to pull from
		      std::vector<T> &vOutData)  // [O] output data
{
  function_switch("distributed_pull");

  // 1.) Translate pull to push instructions:
  std::vector<int> vPushTask;
  std::vector<long> vPushInd, vPushOffset;
  pull_to_multipush(vReqTask, vReqIndex, vPushTask, vPushInd, vPushOffset, vData.size());
  
  // 2.) Send data to requestors:
  
  std::vector<int> vCheckTask(nReqs, -1);
  distributed_multiSendPush(vData, vPushTask, vPushInd, vPushOffset, vOutData, vCheckTask, 1);

  // Final bit: Verify that vCheckTask is identical to vReqTask
  for (long ii = 0; ii < nReqs; ii++)
    if (vCheckTask.at(ii) != vReqTask.at(ii))
      {
	std::cout << "Error detected in distributed_pull - inconsistent tasks" << std::endl;
	exit(9185);
      }

  function_switch("distributed_pull");
  return;
}


template <typename T>						      
void decompose_IDs(const std::vector<unsigned long> &vIDs,
		   std::vector<int> &vML_task,
		   std::vector<T> &vML_index)

{
  size_t nTaskIDs = vIDs.size();
  
  vML_index.clear();
  vML_index.resize(nTaskIDs);
  vML_task.clear();
  vML_task.resize(nTaskIDs);

  // If we get here and mediator_chunk_size has not been initialized, we're 
  // in trouble, so may as well use it:
  long chunk_size = mediator_chunk_size("dummy", "Dummy", 0);

  for (size_t ii = 0; ii < nTaskIDs; ii++) 
    {
      vML_task.at(ii) = static_cast<int>(vIDs.at(ii) / chunk_size);
      vML_index.at(ii) = static_cast<T>(vIDs.at(ii) % chunk_size);
    }
  
  return;

}


template <typename T>
void distributed_multiRecvPush(const std::vector<T> &vData,     // Data to be exchanged 
			       const std::vector<int> &vDestTask,  // Destination task
			       const std::vector<long> &vDestInd,   // Destination index
			       std::vector<T> &vOutList,        // [O] List to be built
			       std::vector<int> &vSourceTasks,     // [O] Where data is from
			       std::vector<long> &vOffset,        // [O] Offset into above
			       long nNumTT, // num for offset list 
			       int FlagConstructSourceTasks) {     // If !=1, skip source tasks


  // This is a modified/generalised version of 'distributed_exchange'.
  // It deals with cases where multiple sources push data to the same receiver,
  // by building TWO output lists: 
  // (i)  the actual 'received data' list, which contains all submissions in 
  //      (compact) order
  // (ii) an offset list translating indices in the (hypothetical) local list 
  //      into this received data list. 
  // As with distributed_exchange, there is also the option to record the source
  // tasks along with the data


  using namespace std;
  int rc = 0;

  function_switch("distributed_multiRecvPush");
  
  // 1.) Create an IDL-style histogram of the destination tasks
  // 2.) Transpose exchange counts between tasks with MPI_Alltoall
  // 3.) Fill in output from self
  // 4.) Loop through separations and exchange info with other tasks


  // 1.) Create an IDL-style histogram of the destination tasks
  int nMin = 0, nMax = numtasks()-1;
  vector<long> vTaskOffset, vTaskRI;
  vector<long> vTaskCountsSend = idlhist<long, int>(vDestTask, 
						    vTaskOffset, 
						    vTaskRI, 
						    nMin, nMax, 0, 0);

  // 2.) Transpose exchange counts between tasks with MPI_Alltoall
  vector<long> vTaskCountsRecv(numtasks());
  rc = MPI_Alltoall(&vTaskCountsSend.front(), 1, MPI_LONG, 
		    &vTaskCountsRecv.front(), 1, MPI_LONG,
		    MPI_COMM_WORLD);

  // 2a.) And calculate total number of elements received on this task
  long nTT_TotRecv = 0;
  for (int ii = 0; ii < numtasks(); ii++)
    nTT_TotRecv += vTaskCountsRecv.at(ii);

  // 2b.) Set up temporary holding vectors:
  std::vector<T> vTT_TempRecv(nTT_TotRecv);
  std::vector<int> vTT_TempSourceTasks(nTT_TotRecv);
  std::vector<long> vTT_TempDestInd(nTT_TotRecv);

  long nIndInTemp = 0; // for filling temporary holding list

  // 3.) Fill in output from self
  long NumXCurr = vTaskCountsSend.at(mpirank());
  if (NumXCurr > 0) {
    long nCurrTaskOffset = vTaskOffset.at(mpirank());

    for (long ii = 0; ii < NumXCurr; ii++) {
      long nCurrParticle = vTaskRI.at(ii+nCurrTaskOffset);
      T nCurrData = vData.at(nCurrParticle);
      long nCurrDestInd = vDestInd.at(nCurrParticle);

      vTT_TempRecv.at(nIndInTemp) = nCurrData;
      vTT_TempSourceTasks.at(nIndInTemp) = mpirank();
      vTT_TempDestInd.at(nIndInTemp) = nCurrDestInd;
      nIndInTemp++;

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

    int nCurrSend = vTaskCountsSend.at(nSendToTask);
    vector<long> vSendBuf_Loc(nCurrSend);
    vector<T> vSendBuf_Data(nCurrSend);

    // Make temporary arrays holding data to be sent:
    // Locations in other task, and data

    int nCurrTaskOffset = vTaskOffset.at(nSendToTask);
    for (int ii = 0; ii < nCurrSend; ii++) {
      int nCurrParticle = vTaskRI.at(nCurrTaskOffset + ii);
      long nCurrXLoc = vDestInd.at(nCurrParticle); 
      T nCurrData = vData.at(nCurrParticle);
      vSendBuf_Loc.at(ii) = nCurrXLoc;
      vSendBuf_Data.at(ii) = nCurrData;
    }


    int nCurrRecv = vTaskCountsRecv.at(nRecvFromTask);
    vector<long> vRecvBuf_Loc(nCurrRecv);
    vector<T> vRecvBuf_Data(nCurrRecv);


    MPI_Sendrecv(&vSendBuf_Loc.front(), nCurrSend, MPI_LONG, nSendToTask, 7775,
		 &vRecvBuf_Loc.front(), nCurrRecv, MPI_LONG, nRecvFromTask, 7775,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&vSendBuf_Data.front(), nCurrSend*sizeof(T), MPI_BYTE, nSendToTask, 7776,
		 &vRecvBuf_Data.front(), nCurrRecv*sizeof(T), MPI_BYTE, nRecvFromTask, 7776,
		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    // Now incorporate just-received info into temporary list
    for (int ii = 0; ii < nCurrRecv; ii++) {
	
      long nCurrDestInd = vRecvBuf_Loc.at(ii);
      T nCurrData = vRecvBuf_Data.at(ii);

      vTT_TempRecv.at(nIndInTemp) = nCurrData;
      vTT_TempSourceTasks.at(nIndInTemp) = nRecvFromTask;
      vTT_TempDestInd.at(nIndInTemp) = nCurrDestInd;
      nIndInTemp++;
      
    }

  } // ends loop through other tasks to exchange data
  
  // 4.) Final bit (specific to multiRecv-version):
  //     need to 'unscramble' the temporary list(s) into final output

  // Check that things have gone ok so far:
  if (nIndInTemp != nTT_TotRecv) {
    std::cout << "In function `" << __func__ << "', file " << __FILE__ << ", line " << __LINE__ << ": Received " << nIndInTemp << " elements, but expected " << nTT_TotRecv << std::endl;
    exit(718501);
  } 

  // Set up output lists (always from scratch here!)
  vOutList.clear();
  vSourceTasks.clear();
  vOffset.clear();

  vOutList.resize(nTT_TotRecv);
  vSourceTasks.resize(nTT_TotRecv);
  
  // Generate sorter and offset list:
  std::vector<long> vIndRI;
  int nDummy = 0;
  int nNumTTInt = static_cast<int>(nNumTT);
  std::vector<long> vNumRecvPerInd = idlhist(vTT_TempDestInd,
					     vOffset, 
					     vIndRI, 
					     nDummy, nNumTTInt, 0, 0);
 
  // Transcribe data and source info according to sorter (vIndRI):
  for (long ii = 0; ii < nTT_TotRecv; ii++) {
    long nCurrRI = vIndRI.at(ii);
    vOutList.at(ii) = vTT_TempRecv.at(nCurrRI);
    vSourceTasks.at(ii) = vTT_TempSourceTasks.at(nCurrRI);
  }

  // We are done!
  
  function_switch("distributed_multiRecvPush");  
  return;
  
}

template <typename T>
void distributed_multiSendPush(const std::vector<T> &vData,  // data to send from
			       const std::vector<int> &vSendTask, // list of receivers
 			       const std::vector<long> &vSendIndex, // list of destination indices local to each receiver
			       const std::vector<long> &vSendOffset, // offset per element into above lists
			       std::vector<T> &vOutData, // [O] output data list
			       std::vector<int> &vSourceTask, // [O] source tasks
			       int nFlagBuildSourceTasks) // 1 -> build source task list

{
  // Wrapper around distributed_exchange to deal with case of one data element
  // being sent to multiple receivers (but each receiver gets only one packet).

  // 1.) Build temporary data vector corresponding to vSendTask/...index

  long nNumToSend = vSendIndex.size();
  long nNumSourceElems = vData.size();
  std::vector<T> vTempData(nNumToSend);

  for (long ii = 0; ii < nNumSourceElems; ii++) {
    for (long jj = vSendOffset.at(ii); jj < vSendOffset.at(ii+1); jj++)
      vTempData.at(jj) = vData.at(ii);
  }
   
  // 2.) Send just-created temporary list with distributed_exchange, as usual
  distributed_exchange(vTempData, vSendTask, vSendIndex, vOutData, vSourceTask, nFlagBuildSourceTasks); 

  // And done!

  return;
}
