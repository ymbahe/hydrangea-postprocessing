#ifndef SEARCH_KATAMARAN_H
#define SEARCH_KATAMARAN_H

template <typename T>
long search_katamaran(const std::vector<unsigned long> &vTaskIDs_A,       // Source IDs [original order]
		      const std::vector<long> &vSortedIndices_A,  // Sort-vector for source IDs
		      
		      std::vector<unsigned long> &vTaskIDs_B,       // Target IDs [original order]
		      std::vector<long> &vSortedIndices_B,  // Sort-vector for target IDs

		      std::vector<T> &vMatchInds, // [O] Match-vector (see above) 
		      std::vector<int> &vMatchTasks, // [O] Match tasks (may be dummy) 
		      long PartOffset_B,            // Offset of B part ID list
		      int nFlagReturnTasks = 0,      // If 1, match tasks are returned
		      int nFlagLocalIndices = 0);    // If 1, indices returned are LOCAL, not global



std::vector<int> index_to_halo(std::vector<long> &vIndex,                 // Index list
			       const std::vector<long> &vOffset,        // HPOlist
			       std::vector<long> &vOrder);       // Order in remote halo

#include "search_katamaran.tpp"

#endif
