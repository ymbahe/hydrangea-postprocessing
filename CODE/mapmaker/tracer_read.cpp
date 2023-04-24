// tracer_read.cpp
// Functions to read the tracer ID list

#include <mpi.h>
#include "Config.h"

#include <iostream>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <cstdlib>

#include "globals.h"

#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/utilities_mpi.h"
#include "/u/ybahe/cpplib/io_hdf5.h"
#include "/u/ybahe/cpplib/correlate_ids.h"

#include "tracer_read.h"


void make_tracer_tpolist(std::vector<int> &vGPOlistT, 
			 std::vector<int> &vTPOlistT, 
			 std::vector<int> &vTGOlistT) {

  function_switch("make_tracer_tpolist");

#ifdef DEBUG
  std::cout << rp() + "Input filename = '" << Params.cFileNameTracing << "'." << std::endl;
#endif

  // Steps for this:
  // Load GPOlist from file -- broadcast - make other lists

  if (g_Snepshot.Type == 1) {
    std::cout << "You are trying to load tracing particles although we are not currently at a snapshot. Terminating." << std::endl;
    exit(77);
  }

  std::string sSnap = to_string(g_Snepshot.Num, 3);

  if (mpirank() == 0) {
    std::string sDataSetName = "Snapshot_"  + sSnap + "/SnipshotIDs/Offset";
    
    std::vector<long> vOffsetSize = hdf5_measure_dataset(Params.cFileNameTracing, sDataSetName);
    if (vOffsetSize.size() != 1) {
      std::cout << "Unexpected dimensions of Offset dataset - " << vOffsetSize.size() << std::endl;
      exit(78);
    }
    
    read_hdf5_data(Params.cFileNameTracing, sDataSetName, vGPOlistT); 
  }

  broadcast_vector<int> (vGPOlistT, 0);
  
  // May as well build offset lists on each task independently
  chunk_split_list<int> (vGPOlistT, numtasks(), vTPOlistT, vTGOlistT, 0);
  
  function_switch("make_tracer_tpolist");
  return;
}




void read_tracer_ids(long nStart, long nLength, std::vector<unsigned long> &vTT_IDsT) {
  
  
  function_switch("read_tracer_ids");
  using namespace std;
  
  std::string sSnap = to_string(g_Snepshot.Num, 3);
  
  read_hdf5_data(Params.cFileNameTracing, "Snapshot_" + sSnap + "/SnipshotIDs/IDs", 
		 vTT_IDsT, nStart, nLength);
  
  function_switch("read_tracer_ids");
  return;
}


void create_output_file(const std::vector<Snepshot> &vSnepshotList) {
  
  using namespace std;
  
  cout << rp() + "Setting up output file '" << Params.cFileNameOut << "'" << endl;

  hdf5_create_file(Params.cFileNameOut);
  
  std::vector<int> vSnepDims(1);
  int nSneps = Params.nSnepFin - Params.nSnepIni + 1;
  vSnepDims.front() = nSneps;
  
  // Make 'meta' datasets
  
  std::vector<int> vSnepshotListType(nSneps), vSnepshotListNum(nSneps);
  std::vector<int> vSnapshotInds, vSnipshotInds;
  
  for (int ii = 0; ii < nSneps; ii++) {
    Snepshot SnepCurr = vSnepshotList.at(ii);
    vSnepshotListType.at(ii) = SnepCurr.Type;
    vSnepshotListNum.at(ii) = SnepCurr.Num;
    if (vSnepshotListType.at(ii) == 0)
      vSnapshotInds.push_back(ii);
    else if (vSnepshotListType.at(ii) == 1)
      vSnipshotInds.push_back(ii);
    else {
      std::cout << "Unexpected snepshot type encountered: " << vSnepshotListType.at(ii) << " on index " << ii << ". Terminating..." << std::endl;
      exit(444);
    }
  }
  
  hdf5_create_dataset(Params.cFileNameOut, "SnepshotType", vSnepDims, "int", "Is this snepshot a snapshot [0] or snipshot [1]?");
  write_hdf5_data(Params.cFileNameOut, "SnepshotType", vSnepshotListType);
  
  hdf5_create_dataset(Params.cFileNameOut, "SnepshotNum", vSnepDims, "int", "Snap/snipshot number of this snepshot");
  write_hdf5_data(Params.cFileNameOut, "SnepshotNum", vSnepshotListNum);
  
  std::vector<int> vSnapIndDims(1);
  vSnapIndDims.front() = vSnapshotInds.size();
  std::cout << rp() + "Found " << vSnapshotInds.size() << " snapshots in current snepshot list..." << std::endl;
  hdf5_create_dataset(Params.cFileNameOut, "SnapshotIndex", vSnapIndDims, "int", "Snepshot indices of snApshots");
  write_hdf5_data(Params.cFileNameOut, "SnapshotIndex", vSnapshotInds);

  std::vector<int> vSnipIndDims(1);
  vSnipIndDims.front() = vSnipshotInds.size();
  std::cout << rp() + "Found " << vSnipshotInds.size() << " snipshots in current snepshot list..." << std::endl;
  hdf5_create_dataset(Params.cFileNameOut, "SnipshotIndex", vSnipIndDims, "int", "Snepshot indices of snIpshots");
  write_hdf5_data(Params.cFileNameOut, "SnipshotIndex", vSnipshotInds);
  
  return;
}


void create_snepshot_datasets(int nLength) 
{
  
  using namespace std;
  cout << rp() + "Setting up datasets for snepshot " << g_ttSnep << "..." << std::endl;
  
  std::string sSnep = to_string(g_ttSnep, 4);
  std::string sGroupName = "Snepshot_" + sSnep;
  hdf5_create_group(Params.cFileNameOut, sGroupName);

  // Vector specifying dimensions of to-be-created datasets
  std::vector<int> vDims(2);
  vDims.at(0) = nLength;
  vDims.at(1) = 3;

  double aexp_curr = read_hdf5_attribute_double(g_SnepshotDir, "Header/ExpansionFactor", 'g');
  double hubble_curr = read_hdf5_attribute_double(g_SnepshotDir, "Header/HubbleParam", 'g');
  

  hdf5_create_dataset(Params.cFileNameOut, sGroupName + "/Coordinates", vDims, "double", "Approximate centre of mass of the galaxy in this snepshot", -1000000);
  hdf5_write_attribute(Params.cFileNameOut, sGroupName + "/Coordinates", "aexp-factor", aexp_curr);
  hdf5_write_attribute(Params.cFileNameOut, sGroupName + "/Coordinates", "h-factor", 1.0/hubble_curr);
  
  hdf5_create_dataset(Params.cFileNameOut, sGroupName + "/Velocity", vDims, "float", "Approximate peculiar velocity of the galaxy in this snepshot", -1000000);
  hdf5_write_attribute(Params.cFileNameOut, sGroupName + "/Velocity", "aexp-factor", 1.0);
  hdf5_write_attribute(Params.cFileNameOut, sGroupName + "/Velocity", "h-factor", 1.0);

  
  return;
}




#ifdef ALLTHEREMAININGJUNK

// ******************************************************
// Function to read data of type LONG from an HDF5 file

void read_hdf5_data(std::string sFileName,            // HDF5 file name
		    std::string DataSetName,          // Data set to read
		    long pos_start,                    // first entry to read...
		    long pos_length,                   // ... and how many in total
		    std::vector<int> vOutput) {           // [O] output

  using namespace std;
  using namespace H5;

  vOutput.clear();

#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data");
  
  cout << rp() + "sFileName  = " << sFileName << endl;
  cout << rp() + "pos_start  = " << pos_start << endl;
  cout << rp() + "pos_length = " << pos_length << endl;
#endif  

  H5File file (sFileName, H5F_ACC_RDONLY);

  DataSet dataset = file.openDataSet(DataSetName);
  DataSpace dataspace = dataset.getSpace();

  if (pos_start == -1 && pos_length == -1) {
    hsize_t dims[1];
    int ndims = dataspace.getSimpleExtentDims(dims, NULL);
    pos_start = 0;
    pos_length = dims[0];
  }
  
  hsize_t count[1];
  hsize_t start[1]; 
  
  count[0] = pos_length;
  start[0] = pos_start;
  dataspace.selectHyperslab(H5S_SELECT_SET, count, start); 
  
  hsize_t dim[1];
  dim[0] = pos_length;
  
  DataSpace memspace(1, dim);

  vOutput.resize(pos_length);
  dataset.read(&vOutput.front(), PredType::NATIVE_INT, memspace, dataspace);
  
  dataset.close();
  file.close();

#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data");
#endif

  return;
}


void read_hdf5_data(std::string sFileName,            // HDF5 file name
		    std::string DataSetName,          // Data set to read
		    long pos_start,                    // first entry to read...
		    long pos_length,                   // ... and how many in total
		    std::vector<long> vOutput) {           // [O] output

  using namespace std;
  using namespace H5;

  vOutput.clear();

#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data");
  
  cout << rp() + "sFileName  = " << sFileName << endl;
  cout << rp() + "pos_start  = " << pos_start << endl;
  cout << rp() + "pos_length = " << pos_length << endl;
#endif  

  H5File file (sFileName, H5F_ACC_RDONLY);

  DataSet dataset = file.openDataSet(DataSetName);
  DataSpace dataspace = dataset.getSpace();

  if (pos_start == -1 && pos_length == -1) {
    hsize_t dims[1];
    int ndims = dataspace.getSimpleExtentDims(dims, NULL);
    pos_start = 0;
    pos_length = dims[0];
  }
  
  hsize_t count[1];
  hsize_t start[1]; 
  
  count[0] = pos_length;
  start[0] = pos_start;
  dataspace.selectHyperslab(H5S_SELECT_SET, count, start); 
  
  hsize_t dim[1];
  dim[0] = pos_length;
  
  DataSpace memspace(1, dim);

  vOutput.resize(pos_length);
  dataset.read(&vOutput.front(), PredType::NATIVE_LONG, memspace, dataspace);
  
  dataset.close();
  file.close();

#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data");
#endif

  return;
}



// *******************************************************************************
// Function to distribute galaxies across tasks

void build_tpolist(const std::vector<int> &vFullPartOffset,  // Offset list (assumed dense) 
		   std::vector<int> &vTPOlist,     // [O] Task --> particles
		   std::vector<int> &vTGOlist,     // [O] Task --> galaxies
		   int nGalIni = 0) {              // Can specify starting from != 0

  function_switch("build_tpolist");
  using namespace std;
  
  int nSpan = vFullPartOffset.size()-1;
  int nPartTot = vFullPartOffset.back()-vFullPartOffset.front();

#ifdef VERBOSE
  cout << rp() + "nSpan = " << nSpan << endl;
  cout << rp() + "vFullPartOffset[0]   = " << vFullPartOffset.front() << endl;
#endif 
  
  int nDesPartCore = static_cast<int>(static_cast<double>(nPartTot) / numtasks());
  int nCurrDesPartCore = nDesPartCore;
  
  cout << rp() + "There are " << nPartTot << " particles in total." << endl;
  cout << rp() + "Aiming for " << nDesPartCore << " particles per core" << endl;  
  
  int currcounter = 0;
  int currtask = 0;

  vTPOlist.resize(numtasks()+1);
  vTGOlist.resize(numtasks()+1);
  
  // Initialise the FIRST element (easy):
  vTPOlist.front() = vFullPartOffset.front();
  vTGOlist.front() = nGalIni;

#ifdef DEBUG
  std::cout << rp() + "Sizes of vFullPartOffset / Length: " << vFullPartOffset.size() << ", " << vFullPartLength.size() << std::endl;
  std::cout << rp() + "nSpan = " << nSpan << std::endl;
#endif

  
  // Now we need to loop through the halo list...
  for (int iihalo = 0 ; iihalo < nSpan; iihalo++)
    {
      long CurrHaloLength = vFullPartOffset.at(iihalo+1)-vFullPartOffset.at(iihalo);
      currcounter += CurrHaloLength;

      if (currcounter >= nCurrDesPartCore and currtask < (numtasks()-1))
	{
	  vTPOlist.at(currtask+1) = vFullPartOffset.at(iihalo+1);
	  vTGOlist.at(currtask+1) = nGalIni+iihalo+1;
	  
	  currcounter = 0;
	  currtask++;

#ifdef DEBUG
	  std::cout << rp() + "currtask = " << currtask << std::endl;
#endif

	  nCurrDesPartCore = static_cast<int>(static_cast<double>(vFullPartOffset.at(nSpan)-vFullPartOffset.at(iihalo+1))/(numtasks()-currtask));

	} // ends section if we've reached the end of current task's allocation
    } // ends loop through haloes
  
  // Add codae
  vTPOlist.at(numtasks())=vFullPartOffset[nSpan];
  vTGOlist.at(numtasks())=nGalIni+nSpan;

  function_switch("build_tpolist");
  return; 

}


#endif

void verify_id_range(std::vector<unsigned long> vIDs) {

  long nIDs = vIDs.size();
  unsigned long nMaxID = numtasks()*static_cast<unsigned long>(mediator_chunk_size(g_SnepshotDir, std::string(Params.cRunType)))-1;

  for (long ii = 0; ii < nIDs; ii++) {
    if (vIDs.at(ii) > nMaxID) {
      std::cout << rp() + "ID " << ii << " [=" << vIDs.at(ii) << "] is unexpectedly large. Please investigate." << std::endl;
      exit(999);
    }
  }   
 
  return;
}


void verify_id_range(std::vector<long> vIDs) {

  long nIDs = vIDs.size();
  long nMaxID = numtasks()*static_cast<long>(mediator_chunk_size(g_SnepshotDir, std::string(Params.cRunType)))-1;

  for (long ii = 0; ii < nIDs; ii++) {
    if (vIDs.at(ii) > nMaxID) {
      std::cout << rp() + "ID " << ii << " [=" << vIDs.at(ii) << "] is unexpectedly large. Please investigate." << std::endl;
      exit(999);
    }
  }   
  
  return;
}
