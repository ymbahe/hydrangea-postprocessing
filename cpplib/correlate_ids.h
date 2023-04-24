// build_mediator_list(): 
// Function to construct 'inverse ID' list
// from distributed ID array. 

#ifndef CORRELATE_IDS_H
#define CORRELATE_IDS_H

template<typename T>
void build_mediator_list(const std::vector<unsigned long> &vTaskIDs,    // Source IDs on this task
			 std::vector<T> &vML_Indices,          // [O] Transtable chunk
			 std::vector<int> &vML_OrigTasks,      // [O] Task of origin
			 std::string sSnepshotDir,             // To determine max ID
			 std::string sRunType,
			 long PartOffset,                      // Offset of this task's ID list chunk
			 int FlagIndexType = 0,  // 0=full, 1=local
			 long nParts = 0);       // If > 0, use pre-defined number of IDs

// correlate_ids():
// Function to match IDs between two snapshots, using pre-built
// mediator lists as input.
// It returns the number of particles in snap A that could be matched

long correlate_ids(const std::vector<long> &vML_IndicesA,     // Indices in snapshot A
		   const std::vector<int> &vML_OrigTasksA,    // Tasks in snapshot A
		   const std::vector<long> &vPartOffsetA,     // Particle offsets in A
		   const std::vector<long> &vML_IndicesB,     // Indices in snapshot B
		   std::vector<long> &vFullIndexInB,         // [O] Indices in B matched to A
		   int FlagInputIsLocal = 0); // // def=0, if 1: vML_IndicesA is local, not global. Nature of vML_IndicesB is irrelevant as it's just passed on, so outtype = intype. 

// mediator_chunk_size(): 
// Convenience function to calculate (on first call)
// and later return the mediator list chunk size
  
int mediator_chunk_size(std::string sSnepDir,
			std::string sRunType,
			long nParts = 0);


// distributed_exchange():
// Function to exchange data between tasks.
// It takes three inputs, 'data', 'tasks', and 'inds', 
// and moves 'data' to task 'task' at index 'inds'.

template <typename T>
void distributed_exchange(const std::vector<T> &vData,     // Data to be exchanged 
			  const std::vector<int> &vDestTask,  // Destination task
			  const std::vector<long> &vDestInd,   // Destination index
			  std::vector<T> &vOutList,        // [O] List to be built
			  std::vector<int> &vSourceTasks,     // [O] Where data is from
			  int FlagConstructSourceTasks=0);    // If !=1, skip source tasks

template <typename T>
std::vector<T> localize_indices(std::vector<T> vIndices,
	       	                std::vector<int> vTasks,
  				std::vector<long> vOffsets,
				int FlagIndicesAreLocal);


template <typename T>
std::vector<T> globalize_indices(std::vector<T> vIndices,
				 std::vector<int> vTasks,
				 std::vector<long> vOffsets,
				 int FlagIndicesAreLocal);


template <typename T>
void unpack_global_indices(const std::vector<T> &vGlobalIndices,
			   const std::vector<long> &vTPOlist,
			   std::vector<int> &vTasks,           // [O] tasks
			   std::vector<T> &vLocalIndices); // [0] loc. inds

template <typename T>						      
void decompose_IDs(const std::vector<unsigned long> &vIDs,
		   std::vector<int> &vML_task,
		   std::vector<T> &vML_index);
		  

template <typename T>
void distributed_pull(const std::vector<T> &vData,  // data to be exchanged
		      const std::vector<int> &vReqTask, // index to pull from
		      const std::vector<long> &vReqIndex, // task to pull from
		      std::vector<T> &vOutData);  // [O] output data


template <typename T> 
void distributed_lookup(const std::vector<unsigned long> &vTT_IDs, // IDs to be matched
			const std::vector<int> &vML_tasks,         // Source tasks
			const std::vector<T> &vML_indices,         // (Local) inds @ source 
			std::vector<int> &vTT_MatchTasks,     // [O] Matched tasks
			std::vector<T> &vTT_MatchInds);        // [O] Matched inds


template <typename T>
void distributed_multiRecvPush(const std::vector<T> &vData,     // Data to be exchanged 
			       const std::vector<int> &vDestTask,  // Destination task
			       const std::vector<long> &vDestInd,   // Destination index
			       std::vector<T> &vOutList,        // [O] List to be built
			       std::vector<int> &vSourceTasks,     // [O] Where data is from
			       std::vector<long> &vOffset,        // [O] Offset into above
			       long nNumTT, // num for offset list 
			       int FlagConstructSourceTasks);     // If !=1, skip source tasks

template <typename T>
void distributed_multiSendPush(const std::vector<T> &vData,  // data to send from
			       const std::vector<int> &vSendTask, // list of receivers
 			       const std::vector<long> &vSendIndex, // list of destination indices local to each receiver
			       const std::vector<long> &vSendOffset, // offset per element into above lists
			       std::vector<T> &vOutData, // [O] output data list
			       std::vector<int> &vSourceTask, // [O] source tasks
			       int nFlagBuildSourceTasks); // 1 -> build source task list

void pull_to_multipush(const std::vector<int> &vReqTask, 
		       const std::vector<long> &vReqIndex,
		       std::vector<int> &vPushTask,
		       std::vector<long> &vPushInd,
		       std::vector<long> &vPushOffset,
		       long nNumElem);


#include "correlate_ids.tpp"

#endif
