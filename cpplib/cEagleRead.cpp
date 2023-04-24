#include <mpi.h>
#include <iostream>
#include <vector>
#include "utilities.h"
#include "io_hdf5.h"

extern "C"
{
  int ceagleread_files(int nFiles,
		       void* data,
		       char* filename,
		       char* dsetname,
		       char* elemNumAttribute,
		       int nElemNumAttributeIndex);
}


int test_list(std::vector<long> &vList,
	      int nRefLength,
	      int nFlagCriterion);

std::string change_file_sequence_nr(std::string sPrimaryFileName, 
				    int nFileNumber);

int check_fdo_list(std::vector<long> &vFDOlist, 
		   std::string sPrimaryFileName,
		   std::string sDataSetName,
		   std::string sFileNumAttribute, 
		   char cFileNameAttributeType,
		   std::string sElemNumAttribute,
		   char cElemNumAttributeType,
		   int nElemNumAttributeIndex); 



int ceagleread_files(int nFiles, 
		     void* data,
		     char* filename,
		     char* dsetname,
		     char* elemNumAttribute,
		     int nElemNumAttributeIndex)
		     
{

  std::cout << "Performing ceagleread_files with nFiles=" << nFiles << std::endl;
  std::cout << "Reading from base file " << filename << std::endl;

  std::vector<long> vFDOlist;
  std::string sPrimaryFileName = filename;
  std::string sDataSetName = dsetname;
  std::string sFileNumAttribute = "Header/NumFilesPerSnapshot";
  char cFileNumAttributeType = 'g';
  char cElemNumAttributeType = 'g';
  std::string sElemNumAttribute = elemNumAttribute;

  if(check_fdo_list(vFDOlist, sPrimaryFileName, sDataSetName, sFileNumAttribute, cFileNumAttributeType, sElemNumAttribute, cElemNumAttributeType, nElemNumAttributeIndex) == 1)
    std::cout << "FDO list was created successfully." << std::endl;


  return 0;

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
    long nFiles = read_hdf5_attribute_long(sPrimaryFileName, sFileNumAttribute, cFileNameAttributeType);
    vTFOlist = split_list<int> (nFiles, 1, 0);
           
    // (c)
    int nTT_Files = vTFOlist.at(1) - vTFOlist.at(0);
    
    vFDOlist.resize(nTT_Files+1, 0);
    for(int iifile = 0; iifile < nTT_Files; iifile++) {
      int nCurrFile = vTFOlist.at(mpirank()) + iifile;
      
      std::string sFileNameCurr = change_file_sequence_nr(sPrimaryFileName, nCurrFile);
      long nElemThisFile = read_hdf5_attribute_long(sFileNameCurr, sElemNumAttribute, cElemNumAttributeType, nElemNumAttributeIndex);
      vFDOlist.at(iifile+1) = vFDOlist.at(iifile) + nElemThisFile;

    }
  }
  return FlagRecompute;

}


// --------------------------------------------------------------
// Function to test whether an input vector has a certain length
// --------------------------------------------------------------

int test_list(std::vector<long> &vList,
	      int nRefLength,
	      int nFlagCriterion) 
{
  
  int nSize = vList.size();
  int FlagRecompute = 0;
  
  if (nFlagCriterion == 0) {
    if (nSize == nRefLength)
      FlagRecompute = 1; }
  else {
    if (nSize != nRefLength)
      FlagRecompute = 1; }

  return FlagRecompute;

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
