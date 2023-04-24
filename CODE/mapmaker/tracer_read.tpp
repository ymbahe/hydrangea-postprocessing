#include "globals.h"

#include "/u/ybahe/cpplib/correlate_ids.h"
#include "/u/ybahe/cpplib/utilities_mpi.h"
#include "/u/ybahe/cpplib/hydrangea.h"

template <typename T>
void find_gal_averages(std::string sDataSetName, 
		       int nDim,
		       std::vector<long> &vTPOlistS,
		       std::vector<long> &vFPOlistS,
		       const std::vector<int> &vDestTasks,
		       const std::vector<int> &vDestInds,
		       const std::vector<int> &vGPOlist,
		       const std::vector<int> &vTGOlist,
		       int FlagAstroConv) {

  // 1.) Read data
  std::vector<T> vTT_Data;
  eagleread<T>(std::string(g_SnepshotDir), sDataSetName, vTT_Data, vTPOlistS, 0, vFPOlistS, nDim, "blabla", "blablabla", 'g', 'g', 0, FlagAstroConv);


  // 2.) Re-distribute the data

  // Need to know how many (tracer) particles current task will get:
  int nTT_NumPart = vGPOlist.at(vTGOlist.at(mpirank()+1)) - vGPOlist.at(vTGOlist.at(mpirank()));

  std::vector<T> vGalData(nTT_NumPart, -1);
  std::vector<int> vSourceTasks; // Dummy
  distributed_exchange<T>(vTT_Data, vDestTasks, vDestInds, vGalData, vSourceTasks, 0);
  vTT_Data.clear();

  // 3.) Loop through galaxies on current task
  
  int nTT_GalOffset = vTGOlist.at(mpirank());
  int nTT_PartOffset = vGPOlist.at(nTT_GalOffset);
  int nTT_NumGal = vTGOlist.at(mpirank()+1)-nTT_GalOffset;

  std::vector<T> vOut(nTT_NumGal, -10000);

  for (int iigal = 0; iigal < nTT_NumGal; iigal++) {

    int nCurrGal = nTT_GalOffset + iigal;
    int nPartCurrGal = vGPOlist.at(nCurrGal+1)-vGPOlist.at(nCurrGal);
    
    // Check to make sure galaxy has more than 0 particles!
    // Otherwise, there is nothing we can do to trace it.
    if (nPartCurrGal == 0) {
      //std::cout << rp() + "Galaxy " << nCurrGal << " has 0 tracer particles..." << std::endl;
      continue;
    }


    T runSum = 0;
    for (int iipart = vGPOlist.at(nCurrGal)-nTT_PartOffset; iipart < (vGPOlist.at(nCurrGal+1)-nTT_PartOffset); iipart++)
      runSum += vGalData.at(iipart);
    
    vOut.at(iigal) = runSum / nPartCurrGal;

    if (mpirank() == 0 && iigal < 10)
      std::cout << rp() + "Galaxy " << iigal << "[" << iigal + nTT_GalOffset << "] --> " << vOut.at(iigal) << std::endl;
    
  } // ends loop over galaxies

  // 4.) Write output

  // Modified 1 Dec so that only task-0 writes.
  
  collect_vector_mpi<T>(vOut, 0, 0, 1);  // Vec - dest - FlagOff - FlagClear 

  if (mpirank() == 0) {

    std::string sSnep = to_string(g_ttSnep, 4);

    std::vector<std::string> vNameParts = split_string(sDataSetName, '/');
    std::string sQuant = vNameParts.back();
    
    if (mpirank() == 0)
      std::cout << rp() + "Determined dataset name as '" << sQuant << "', writing..." << std::endl;
    
    write_hdf5_data(Params.cFileNameOut, "Snepshot_" + to_string(g_ttSnep, 4) + "/" + sQuant, vOut, 0, nDim);
  }

  return;

}
