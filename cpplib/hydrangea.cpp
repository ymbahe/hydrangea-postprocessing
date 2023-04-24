// File containing Eagle / Hydrangea specific functions

#include <mpi.h>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include "glob.h"
#include <fstream>
#include <sstream>

#include "init_hydrangea.h"
#include "utilities.h"
#include "utilities_mpi.h"
#include "io_hdf5.h"
#include "hydrangea.h"


// -------------------------------------------------------------
// Function to form the (principal) filename of an Eagle output
// ------------------------------------------------------------

std::string eagle_filename(std::string sBaseDir, 
			   Snepshot Snepshot,
			   int nType,
			   int nRankDummy) {
  
  using namespace std;
  
  int nRank = mpirank();

  string sFileName;
  if (nRank <= 0) {
  
    string sSnap = to_string(Snepshot.Num, 3);
    string sPreDir, sPreFile;
    
    if (Snepshot.Type == SNEPSHOT_SNAP) {  // SNAPshot
      if (nType == EAGLE_SNAPSHOT) { // snapshot
	sPreDir = "snapshot"; 
	sPreFile = "snap"; }
      else if (nType == EAGLE_SUBDIR) { // subdir
	sPreDir = "groups";
	sPreFile = "eagle_subfind_tab"; }
      else if (nType == EAGLE_SUBPART) { // subpartdir
	sPreDir = "particledata";
	sPreFile = "eagle_subfind_particles"; } 
    }
    
    else if (Snepshot.Type == SNEPSHOT_SNIP) {  //  SNIPshot 
      if (nType != 0) {
	cout << "Snipshots don't have subfind..." << endl;
	cout << "You should probably call this function with nType = EAGLE_SNAPSHOT (=" << EAGLE_SNAPSHOT << ")!" << endl;
	exit(7777);
      }
      sPreDir = "snipshot";
      sPreFile = "snip";
    }
    
    std::string dirstring = sBaseDir + "/data/" + sPreDir + "_" + sSnap + "_";
    std::string sBaseName = complete_path(dirstring);
    
    vector<string> vPathParts = split_string(sBaseName, '/');
    vector<string> vFilenameParts = split_string(vPathParts.back(), '_');
    string sZString = vFilenameParts.back();
    
    sFileName = dirstring + sZString + "/" + sPreFile + "_" + sSnap + "_" + sZString + ".0.hdf5";
    
  }
  
  if (nRank >= 0)
    broadcast_string(sFileName, 0);
  
  return sFileName;
  
}


// -------------------------------------------------------------
// Overloaded convenience function - for Snapshot-only use
// ------------------------------------------------------------


std::string eagle_filename(std::string sBaseDir, 
			   int nSnapshot,
			   int nType,
			   int nRank) {
  
  using namespace std;

  string sFileName;
  if (nRank <= 0) {
  
    string sSnap = to_string(nSnapshot, 3);
    string sPreDir, sPreFile;
    
    if (nType == EAGLE_SNAPSHOT) { // snapshot
      sPreDir = "snapshot"; 
      sPreFile = "snap"; }
    else if (nType == EAGLE_SUBDIR) { // subdir
      sPreDir = "groups";
      sPreFile = "eagle_subfind_tab"; }
    else if (nType == EAGLE_SUBPART) { // subpartdir
      sPreDir = "particledata";
      sPreFile = "eagle_subfind_particles"; } 
    
    std::string dirstring = sBaseDir + "/data/" + sPreDir + "_" + sSnap + "_";
    std::string sBaseName = complete_path(dirstring);
    
    vector<string> vPathParts = split_string(sBaseName, '/');
    vector<string> vFilenameParts = split_string(vPathParts.back(), '_');
    string sZString = vFilenameParts.back();
    
    sFileName = dirstring + sZString + "/" + sPreFile + "_" + sSnap + "_" + sZString + ".0.hdf5";
    
  }
  
  if (nRank >= 0)
    broadcast_string(sFileName, 0);
  
  return sFileName;
  
}

 
// ---------------------------------------------------------
// Function to check (and if neccessary create) an FDO list
// ---------------------------------------------------------


int check_fdo_list(std::vector<long> &vFDOlist, 
		   std::string sPrimaryFileName,
		   std::string sDataSetName,
		   std::string sFileNumAttribute, 
		   char cFileNameAttributeType,
		   std::string sElemNumAttribute,
		   char cElemNumAttributeType,
		   int nElemNumAttributeIndex) 
{

  int FlagRecompute = test_list(vFDOlist, 0, 0);

  // ---------- Now main part: construct new simple FDO list if required ------------

  if (FlagRecompute == 1) {

    vFDOlist.clear();

    // Build your brand-new FDO list in five easy(ish) steps:
    //
    // (a) Load the total number of files in data product (on root)
    // (b) Spread this (evenly) across tasks
    // (c) Each task reads length of dataset in 'its' files into (local) offset
    // (d) Collect full offset list on root
    // (e) Broadcast result to all tasks
    
    // (a) + (b)  Create and distribute TFOlist for this job
    
    std::vector<int> vTFOlist;  /* List dividing up files across tasks for analysis */

    if (mpirank() == 0) {
      long nFiles = read_hdf5_attribute_long(sPrimaryFileName, sFileNumAttribute, cFileNameAttributeType);
      vTFOlist = split_list<int> (nFiles, numtasks(), 0);
           
    } else 
      vTFOlist.resize(numtasks()+1,0);
    
    
    MPI_Bcast(&vTFOlist.front(), vTFOlist.size(), MPI_INT, 0, MPI_COMM_WORLD); 

    // (c) Each task reads the length of datasets in 'its' files

    
    int nTT_Files = vTFOlist.at(mpirank()+1) - vTFOlist.at(mpirank());
    
    vFDOlist.resize(nTT_Files+1, 0);
    for (int iifile = 0; iifile < nTT_Files; iifile++) {
      int nCurrFile = vTFOlist.at(mpirank()) + iifile;
      
      std::string sFileNameCurr = change_file_sequence_nr(sPrimaryFileName, nCurrFile);

#ifdef EAGLEREAD_DIRECT_MEASURE
      std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileNameCurr, sDataSetName);
      vFDOlist.at(iifile+1) = vFDOlist.at(iifile) + vDataSetDims.front();
#else
      long nElemThisFile = read_hdf5_attribute_long(sFileNameCurr, sElemNumAttribute, cElemNumAttributeType, nElemNumAttributeIndex);
      vFDOlist.at(iifile+1) = vFDOlist.at(iifile) + nElemThisFile;
#endif

    }

    // (d) + (e)  Collect full offset vector on root and distribute to all other tasks


    collect_vector_mpi<long> (vFDOlist, 0, 1, 2);
    
  } // ends section if FDOlist had to be recomputed
   
  return FlagRecompute;

}


// ------------------------------------------------------
// Function to check, and if necessary build, a TDO list
// ------------------------------------------------------

int check_tdo_list(std::vector<long> &vTDOlist,
		   long nDataTotal) {

  int FlagRecompute = test_list(vTDOlist, numtasks()+1, 1);

  // ---------- Main part: Build (simple) list if required ----------
  
  if (FlagRecompute == 1)
    vTDOlist = split_list<long> (nDataTotal, numtasks(), 0);

  // Yes, it really is that simple.
  
  return FlagRecompute;
  
}


// --------------------------------------------------------------
// Function to test whether an input vector has a certain length
// (also checks that the lists are the same on all MPI tasks)
// --------------------------------------------------------------

int test_list(std::vector<long> &vList,
	      int nRefLength,
	      int nFlagCriterion) {
  
  int nSize = vList.size();
  int FlagRecompute = 0;
  
  if (nFlagCriterion == 0) {
    if (nSize == nRefLength)
      FlagRecompute = 1; }
  else {
    if (nSize != nRefLength)
      FlagRecompute = 1; }

  if (FlagRecompute == 0) {
    
    
    /* sanity check: Are there any differences between input tasks' lists? */
    int nCheckRes = check_for_consistency<int> (nSize, 1);
    if (nCheckRes > 0) {
      if (mpirank() == 0)
	std::cout << rp() + "Inconsistencies in vTDOlists detected (on task " << nCheckRes << " -- recomputing..." << std::endl;
      
      FlagRecompute = 1;
    }
  }

  return FlagRecompute;

}
  


void identify_relevant_files(const std::vector<long> &vFDOlist,  // Data distr. across files 
			     long nDataIni,                      // First interesting data
			     long nDataFin,                      // Last interesting data
			     std::vector<int> &vRelevantFiles,      // [O] Indices of rel. files
			     std::vector<long> &vFirstUsedInFile,   // [O] First useful data/file 
			     std::vector<long> &vFileOffsetInThisTask, // [O] Offs. of file in task  
			     int nVerb)  { 
  
#ifdef DEBUG
  nVerb = 1;
#endif

  if (nVerb == 1)
    function_switch("identify_relevant_files");

  using namespace std;
  
  int nNumFiles = vFDOlist.size()-1;
  int FirstFile = -1, LastFile = -1, success_start = 0, success_end = 0;

  // Sanity check: end after beginning?
  // If not, we can end right here.

  if (nDataFin < nDataIni) {

    vRelevantFiles.clear();
    vFirstUsedInFile.clear();
    vFileOffsetInThisTask.clear();
    vFileOffsetInThisTask.resize(1,0);

    if (nVerb == 1)
      function_switch("identify_relevant_files");

    return;
  }

  
  // Loop through files to identify relevant range:
  for (int iifile = 0; iifile < nNumFiles; iifile++) {
    if (vFDOlist.at(iifile+1) > nDataIni && success_start == 0) {  // nDataIni in current file
      success_start = 1;
      FirstFile = iifile; 
    }

    if (vFDOlist.at(iifile+1) > nDataFin && success_end == 0) {   // nDataFin in current file
      success_end = 1;
      LastFile = iifile; }
  }

#ifdef VERBOSE
  cout << rp() + "FirstFile = " << FirstFile << endl;
  cout << rp() + "LastFile  = " << LastFile << endl;
#endif

  if (success_start == 0 || success_end == 0) {
    cout << rp() + "File identification at least partly unsuccessful! (success_start = " << success_start << ", success_end = " << success_end << endl;
    cout << "nDataIni = " << nDataIni << endl;
    cout << "nDataFin = " << nDataFin << endl;
    exit(42); 
  }

  
  // Non-output values
  int nRelevantFiles = LastFile-FirstFile+1;
  vector<long> vLastUsedInFile(nRelevantFiles); 
  
  vRelevantFiles.resize(nRelevantFiles);
  vFirstUsedInFile.resize(nRelevantFiles);
  vFileOffsetInThisTask.resize(nRelevantFiles+1);
  vFileOffsetInThisTask.front() = 0;
  
  // Loop through relevant files:
  for (int kkfile = 0; kkfile < nRelevantFiles; kkfile++) {
    vRelevantFiles.at(kkfile) = FirstFile + kkfile;

    // The next two are defaults (possibly overwritten later):
    // vFirstUsedInFile: FIRST halo on file kk+FirstFile that is to be INCLUDED 
    // vLastUsedInFile:  LAST halo on file kk+FirstFile that is to be INCLUDED
    
    vFirstUsedInFile.at(kkfile) = 0; 
    vLastUsedInFile.at(kkfile) = vFDOlist.at(FirstFile + kkfile + 1) - vFDOlist.at(FirstFile + kkfile) - 1;
    
  } 
  
  // First file is special - may not be interested in first few data entries:
  vFirstUsedInFile.front() = nDataIni - vFDOlist.at(FirstFile);
  
  // Last file is also special - LAST entry may be non-standard:
  vLastUsedInFile.back() = nDataFin - vFDOlist.at(LastFile);
                                                            
  
  // Now the length of each is simple:
  for (int kkfile = 0; kkfile < nRelevantFiles; kkfile++)
    vFileOffsetInThisTask.at(kkfile+1) = vFileOffsetInThisTask.at(kkfile) + (vLastUsedInFile.at(kkfile) - vFirstUsedInFile.at(kkfile) + 1);

  if (nVerb == 1)
    function_switch("identify_relevant_files");

  return;

}


std::string change_file_sequence_nr(std::string sPrimaryFileName, 
				    int nFileNumber) {

  // Split file name by '.'
  std::vector<std::string> vFNParts = split_string(sPrimaryFileName, '.');
  
  // Change the SECOND-LAST element (XXXXX.0.hdf5)
  //                   v-------------------^
  vFNParts.at(vFNParts.size()-2) = to_string(nFileNumber);
  
  std::string sModString = join_strings(vFNParts, '.');
  
  return sModString;
}


std::vector<Snepshot> read_snepshot_list(std::string sFileName,
					 int nEntries,
					 int nHasHeader) {

  std::vector<Snepshot> vSnepList;
  if (nEntries > 0)
    vSnepList.resize(nEntries);

  std::ifstream file;
  size_t nFileNameLength = sFileName.size();
  char *cFileName = new char[nFileNameLength+1];
  strcpy(cFileName, sFileName.c_str());
  file.open(cFileName);
  delete[] cFileName;

  std::string line;
  std::string sInd, sType, sNum, sRootInd, sAexp;

  std::cout << rp() + "Reading snepshot list from file '" << sFileName << "'..." << std::endl;

  int nCount = 0;
  while (std::getline(file, line))
    {
      nCount++;

      if (nHasHeader && nCount == 1)
	continue;
      
      if (line.size() == 0)
	continue;
      
      if (line.at(0) == '#')
	continue; 
      
      std::istringstream iss(line);
      if (!(iss >> sInd >> sRootInd >> sAexp >> sType >> sNum)) {
	std::cout << "Erroneous format of line " << nCount << "!" << std::endl;
	exit(42);
      }

      Snepshot CurrSnep;
      CurrSnep.Index = string_to_num<int>(sInd);
      if (sType.compare("snap") == 0)
	CurrSnep.Type = 0;
      else if (sType.compare("snip") == 0)
	CurrSnep.Type = 1;
      else {
	std::cout << "Unexpected snepshot type encountered in line " << nCount << ": '" << sType << "'" << std::endl;
	exit(333);
      }
      
      CurrSnep.Num = string_to_num<int>(sNum);
      CurrSnep.RootIndex = string_to_num<int>(sRootInd);

      if (nEntries > 0)
	vSnepList.at(nCount-1) = CurrSnep;
      else
	vSnepList.push_back(CurrSnep);
      

    } // ends loop over input lines


  // Check to make sure we've read correct number of lines
  if (nEntries > 0) {
    if (nCount != nEntries) {
      std::cout << "Only found " << nCount << " input lines in snepshot list, but expected " << nEntries << "!" << std::endl;
      exit(555);
    }
  }

  return vSnepList;
}


// ----------------------------------------------------------------
// Read in a (one-column) ASCII int list from specified place
// ----------------------------------------------------------------

std::vector<int> read_snapshot_list(std::string sFileName,
				    int nEntries) {

  std::vector<int> vSnapList;
  if (nEntries > 0)
    vSnapList.resize(nEntries);
  
  std::ifstream file;
  size_t nFileNameLength = sFileName.size();
  char *cFileName = new char[nFileNameLength+1];
  strcpy(cFileName, sFileName.c_str());
  file.open(cFileName);
  delete[] cFileName;

  std::string line;
  std::string sNum;

  std::cout << rp() + "Reading snapshot list from file '" << sFileName << "'..." << std::endl;

  int nCount = 0;
  while (std::getline(file, line))
    {
      nCount++;
      
      if (line.size() == 0)
	continue;
      
      if (line.at(0) == '#')
	continue; 
      
      std::istringstream iss(line);
      if (!(iss >> sNum)) {
	std::cout << "Erroneous format of line " << nCount << "!" << std::endl;
	exit(42);
      }

      int CurrSnap;
      CurrSnap = string_to_num<int>(sNum);
      
      if (nEntries > 0)
	vSnapList.at(nCount-1) = CurrSnap;
      else
	vSnapList.push_back(CurrSnap);
      

    } // ends loop over input lines

  // Check to make sure we've read correct number of lines
  if (nEntries > 0) {
    if (nCount != nEntries) {
      std::cout << "Only found " << nCount << " input lines in snapshot list, but expected " << nEntries << "!" << std::endl;
      exit(555);
    }
  }

  return vSnapList;
}

std::vector<int> expand_parttype_code(int nCode, char cPartTypeNum) 
{
  std::vector<int> vTypeVec;
  for (char iitype = 0; iitype < cPartTypeNum; iitype++)
    if (nCode & 1<<iitype)
      vTypeVec.push_back(static_cast<int>(iitype));
  
  if (mpirank() == 0) {
    std::cout << rp() + "Selected to process " << vTypeVec.size() << " particle types:" << std::endl;
    print_vector<int> (vTypeVec, "");
  }
  
  return vTypeVec;
}

