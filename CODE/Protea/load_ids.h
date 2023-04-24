#ifndef LOAD_IDS_H
#define LOAD_IDS_H

std::vector<long> 
make_tpolist_parallel(std::string sFileName,                   // Input (subdir) file name
		      int &nHaloIni,                           // First halo to be used
		      int &nHaloFin,                           // Last halo to be used
		      std::vector<int> &vTHOlist,              // [O] Haloes across tasks
		      std::vector<long> &vFullPartOffset,  // [O] Offsets of all used haloes
		      std::vector<long> &vFullPartLength);  // [O] Lengths of all used haloes



void reject_baryons(std::vector<unsigned long> &vThisTaskIDs,          // ID list for current task (changed)
		    std::vector<long> &vFullPartOffset,       // Overall offset list (changed)
		    std::vector<long> &vFullPartLength,       // Overall length list (erased)
		    std::vector<long> &vTPOlist,              // Distrib. of IDs across tasks (ch.)
		    const std::vector<int> &vTHOlist);        // Distrib. of Haloes across tasks


#ifdef OLD_LOADIDS
// To identify which files need to be read
template<typename T>
void identify_relevant_files(const std::vector<T> &vFHOlist,             // Halo distr. across files 
			     T nHaloIni,                         // First interesting halo
			     T nHaloFin,                         // Last interesting halo
			     std::vector<int> &vRelevantFiles,      // [O] Indices of rel. files
			     std::vector<T> &vFirstUsedInFile,    // [O] First useful halo / file 
			     std::vector<T> &vNumUsedInFile,     // [O] N(useful haloes) / file
			     std::vector<T> &vFileOffsetInThisTask, // [O] FO of file in task 
			     int nVerb = 1); 
#endif

// To distribute haloes across tasks
void build_tpolist(const std::vector<long> &vFullPartOffset, 
		   const std::vector<long> &vFullPartLength, 
		   int nHaloIni,
		   std::vector<long> &vTPOlist,
		   std::vector<int> &vTHOlist);

#ifdef OLD_LOADIDS
#include "load_ids.tpp"
#endif

#endif
