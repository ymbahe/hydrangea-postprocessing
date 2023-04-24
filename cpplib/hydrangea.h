#ifndef HYDRANGEA_H
#define HYDRANGEA_H

#include "init_hydrangea.h"

/* Function to expand the scalar particle type code into a vector of particle types */
std::vector<int> expand_parttype_code(int nCode, 
				      char nPartTypeNum = 6);
 

std::string eagle_filename(std::string BaseDir,
			   Snepshot Snepshot,
			   int nType = 0,
			   int nRank = -1);

std::string eagle_filename(std::string BaseDir,
			   int nSnapshot,
			   int nType = 0,
			   int nRank = -1);


template <typename T>
void eagleread(std::string sPrimaryFileName,    // Full path to primary file of data product
	       std::string sDataSetName,        // Full (internal) dataset name
	       std::vector<T> &vOutput,      // [O] The output vector

	       std::vector<long> &vTDOlist,        // [I/O] TDOlist - can be empty on input

	       int nBroadcastFlag,              // 0: no comm., 1: assemble on root, 2: +distr.
	       
	       std::vector<long> &vFDOlist,        // [I/O] FDOlist - can be empty on input
	       int nDimension,                  // For rank-2 datasets, which dimension to read
	       std::string sFileNumAttribute = "Header/NumFilesPerSnapshot",   // Attrib. saying how many files there are
	       std::string sElemNumAttribute = "Header/Nids", // Elem. per file
	       char cFileNumAttributeType = 'g',     // Type of ^ (g[roup], [d]ataset)
	       char cElemNumAttributeType = 'g',
	       int nElemNumAttributeIndex = 0,
	       int FlagAstroConv = 0);



// ------------------------
// INTERNAL-only functions:
// ------------------------

int check_fdo_list(std::vector<long> &vFDOlist, 
		   std::string sPrimaryFileName,
		   std::string sDataSetName,
		   std::string sFileNumAttribute, 
		   char nFileNameAttributeType,
		   std::string sElemNumAttribute,
		   char nElemNumAttributeType,
		   int nElemNumAttributeIndex);

int check_tdo_list(std::vector<long> &vTDOlist,
		   long nDataTotal);


void identify_relevant_files (const std::vector<long> &vFDOlist,
			      long nFirstDataThisTask,
			      long nLastDataThisTask,
			      
			      std::vector<int> &vFilesThisTask, 
			      std::vector<long> &vFirstUsedInFile, 
			      std::vector<long> &vFileOffsetInThisTask,
			      
			      int nSomeFlagThatIDontUnderstandYet);


int test_list(std::vector<long> &vList,
	      int nRefLength,
	      int FlagCriterion);        // 0: return 1 if EQUAL to nRefLength (1: unequal)


std::string change_file_sequence_nr(std::string sPrimaryFileName, 
				    int nFileNumber);
  
std::vector<Snepshot> read_snepshot_list(std::string sFileName,
					 int nEntries = 0,
					 int nHasHeader = 0);

std::vector<int> read_snapshot_list(std::string sFileName,
				    int nEntries = 0);


#include "hydrangea.tpp"

#endif

