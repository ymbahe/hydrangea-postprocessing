// Special functions for mapmaker code

#include "mpi.h"
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "globals.h"

#include "mapmaker_tools.h"
#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/io_hdf5.h"


// -------------------------------
// Determine the best-fit PH level
// -------------------------------

int find_ph_level(std::vector<double> vPartBounds,     // Bounding box of particles
		  std::vector<double> &vCellMins,      // [O] Cell lower corner
		  std::vector<int> &vNumCells)         // [O] Number of cells per dimension
  
// NB: The 'cell bounding box' is the actual box containing the 
// selected PH cells. This is in general (somewhat) larger than the 
// particle bounding box, because cells cannot start and end at arbitrary positions

{

  // Battle plan:
  //
  // 1.) Calculate volume of particle bounding box
  // 2.) Convert this to (ideal, float) number of cells
  // 3.) Find nearest possible integer number of cells
  // 4.) Compute cell bounding box

  function_switch("find_ph_level");
  
  // 1.) Compute bounding box volume (DeltaX * DeltaY * DeltaZ)
  
  double dPartBoxVolume = (vPartBounds.at(3)-vPartBounds.at(0)) * (vPartBounds.at(4)-vPartBounds.at(1)) * (vPartBounds.at(5) - vPartBounds.at(2));
  
  // 2.) Find ideal number of cells
  
  long nPartTot = read_hdf5_attribute_long(g_SnepshotDir, "Header/NumPart_Total", 'g', g_type);
  double dCellVol = dPartBoxVolume * static_cast<double> (Params.nDesNumPerCell) / nPartTot;

  if (dCellVol < 0.000000001) {
    std::cout << "dCellVol=" << dCellVol << ", was adjusted to 0.000000001" << std::endl;
    dCellVol = 0.000000001;
  }

  double dIdealPH = log2(g_dBoxSize) - 1.0/3.0 * log2(dCellVol);
  
  if (mpirank() == 0)
    std::cout << rp() + "Ideal PH level = " << dIdealPH << std::endl;
  
  // 3.) Find bounding integer indices, and which one is closer to the ideal in volume
  int nPHLow = static_cast<int> (dIdealPH);
  int nPHHigh = nPHLow+1;

  double dVCellQuantLow = pow(g_dBoxSize, 3) * pow(2.0, -3*nPHLow);
  double dVCellQuantHigh = pow(g_dBoxSize, 3) * pow(2.0, -3*nPHHigh);

  double dAbsVolDiffLow = abs(dVCellQuantLow-dCellVol);
  double dAbsVolDiffHigh = abs(dVCellQuantHigh-dCellVol);
  double dVolRatio;

  int nPHLevel;
  if (dAbsVolDiffLow < dAbsVolDiffHigh) {
    nPHLevel = nPHLow;
    dVolRatio = dVCellQuantLow/dCellVol;
  }
  else {
    nPHLevel = nPHHigh;
    dVolRatio = dVCellQuantHigh/dCellVol;
  }

  if (mpirank() == 0) {
    std::cout << rp() + "Best-fit PH level determined as " << nPHLevel << "..." << std::endl;
    std::cout << rp() + "  (cell volume of " << dVolRatio*100 << " per cent of ideal)" << std::endl;
  }

  // 4.) Apply constraints, and find final PH level

  int nPHLevelOrig = nPHLevel;

  int nFlagMaxCellSize = 0, nFlagMinNumOfCells = 0;
  int nFlagMaxPHLevel = 0, nFlagMaxNumOfCells = 0;

  // -- a) 'Right-hand' constraints (to reduce cell size)
  
  double dCellVolume, dCellSize;
  long nNumOfCells;

  while(1) {
    
    update_cell_info(nPHLevel, vPartBounds, vCellMins, dCellVolume, dCellSize, vNumCells, nNumOfCells);
    
    if (dCellSize > Params.dMaxCellSize) {
      nPHLevel++;
      nFlagMaxCellSize++;
      continue;
    }
    else if (nNumOfCells < Params.nMinNumOfCells) {
      nPHLevel++;
      nFlagMinNumOfCells++;
      continue;
    }
    else break;

  }
  
  
  // -- b) 'Left-hand' constraints (to increase cell size - these are more important, so 
  //       are evaluated last (and can therefore override the right-hand constraints.

  while(1) {
    
    update_cell_info(nPHLevel, vPartBounds, vCellMins, dCellVolume, dCellSize, vNumCells, nNumOfCells);
    
    if (nPHLevel > Params.nMaxPHLevel) {
      nPHLevel = Params.nMaxPHLevel;
      nFlagMaxPHLevel++;
      continue;
    }
    else if (nNumOfCells > Params.nMaxNumOfCells) {
      nPHLevel--;
      nFlagMaxNumOfCells++;
      continue;
    }
    else break;
  }
  
  // Woohoo. Almost done.
  if (mpirank() == 0) {
    if (nFlagMaxNumOfCells+nFlagMinNumOfCells+nFlagMaxPHLevel+nFlagMaxCellSize == 0)
      std::cout << rp() + "PH level [" << nPHLevel << "] unchanged by constraints." << std::endl;
    else {
      
      if (nFlagMaxNumOfCells > 0)
      std::cout << rp() + "PH level increased " << nFlagMaxNumOfCells << " time(s) to satisfy maximum number of cells (=" << Params.nMaxNumOfCells << ", now " << nNumOfCells << ")" << std::endl;
      if (nFlagMaxPHLevel > 0)
      std::cout << rp() + "PH level increased to satisfy maximum PH level (=" << Params.nMaxPHLevel << ", now " << nPHLevel << ")" << std::endl;
      if (nFlagMinNumOfCells > 0)
      std::cout << rp() + "PH level decreased " << nFlagMinNumOfCells << " time(s) to satisfy minimum number of cells (=" << Params.nMinNumOfCells << ", now " << nNumOfCells << ")" << std::endl;
      if (nFlagMaxCellSize > 0)
      std::cout << rp() + "PH level decreased " << nFlagMaxCellSize << " time(s) to satisfy maximum cell size (=" << Params.dMaxCellSize << ", now " << dCellSize << ")" << std::endl;
      
      std::cout << rp() + "PH level adjusted to " << nPHLevel << " due to boundary constraints." << std::endl;
    }
  }


  function_switch("find_ph_level");
  
  return nPHLevel;
}


void update_cell_info(int nPHLevel,
		      std::vector<double> vPartBounds,
		      std::vector<double> &vCellMins,
		      double &dCellVolume,
		      double &dCellSize,
		      std::vector<int> &vNumCells,
		      long &nNumOfCells)
{

  // 1.) Calculate basic info (cell size/volume)
  dCellSize = g_dBoxSize / ipow_long(2,nPHLevel); 
  dCellVolume = pow(dCellSize, 3);

  // 2.) Calculate the 'minimum-corner' of celled region
  vCellMins.resize(3);
  std::vector<int> vCellOffsets(3);
  for (int iidim = 0; iidim < 3; iidim++) {
    vCellOffsets.at(iidim) = static_cast<int>(vPartBounds.at(iidim)/dCellSize);
    vCellMins.at(iidim) = vCellOffsets.at(iidim)*dCellSize;
  }
  
  // 3.) Calculate number of cells in each dimension
  vNumCells.resize(3);
  nNumOfCells = 1;
  for (int iidim = 0; iidim < 3; iidim++) {
    vNumCells.at(iidim) = static_cast<int>(ceil(vPartBounds.at(iidim+3)/dCellSize))-vCellOffsets.at(iidim)+1;
    nNumOfCells *= static_cast<long>(vNumCells.at(iidim));
  }

  return;

}

		      
int load_input_parameters(int argc, 
			  char* argv[], 
			  std::string &sParamFile) {

  
  using namespace std;

  if (argc > 1) {
    sParamFile = argv[1];
    std::cout << rp() + "Using supplied parameter file '" << sParamFile << "'" << std::endl;
  }

  return argc-1;
}	


int read_parameter_file(std::string sParamFile, RunParams &runParams) {

  using namespace std;

  int nParams = 0;

  ifstream file;
  char tab2[1024];
  strcpy(tab2, sParamFile.c_str());
  file.open(tab2);

  std::string line;
  string param_name, param_val;

  cout << rp() + "Reading parameter file '" << sParamFile << "':" << endl;

  int nCount = 0;
  while (std::getline(file, line))
    {
      nCount++;
#ifdef VERBOSE
      cout << "Line " << nCount << ": " << line << endl;
#endif

      if (line.size() == 0)
	continue;

      if (line.at(0) == '#') {
#ifdef VERBOSE
	cout << "Skipping line " << nCount << endl; 
#endif
	continue; 
      }

      std::istringstream iss(line);
      if (!(iss >> param_name >> param_val)) {
	std::cout << "Erroneous format of line " << nCount << "!" << endl;
	exit(42);
      }

      


#ifdef VERBOSE
      cout << param_name << " is " << param_val << endl;
#endif
      nParams++;

      if(param_name == "StartSnep") 
	runParams.nSnepIni = string_to_num<int>(param_val);
      else if (param_name == "EndSnep")
	runParams.nSnepFin = string_to_num<int>(param_val);
      
      else if (param_name == "SimBaseDir")
	strcpy(runParams.cSimBaseDir, param_val.c_str());
      else if (param_name == "OutputFileName")
	strcpy(runParams.cOutputFileName, param_val.c_str());
      else if (param_name == "SnepshotListName")
	strcpy(runParams.cSnepshotListName, param_val.c_str());
      else if (param_name == "SnepshotListLoc")
	runParams.nSnepshotListLoc = string_to_num<int>(param_val);

      else if (param_name == "NumOfSnepshots")
	runParams.nSnepshotListEntries = string_to_num<int>(param_val);
      else if (param_name == "OutputDirectory")
	strcpy(runParams.cOutputDir, param_val.c_str());
      
      else if (param_name == "MapPartTypes")
	runParams.nTypeCode = string_to_num<int>(param_val);
      else if (param_name == "DesNumPerCell")
	runParams.nDesNumPerCell = string_to_num<int>(param_val);


      else if (param_name == "MaxCellSize")
	runParams.dMaxCellSize = string_to_num<double>(param_val);
      else if (param_name == "MinNumOfCells")
	runParams.nMinNumOfCells = string_to_num<int>(param_val);
      else if (param_name == "MaxNumOfCells")
	runParams.nMaxNumOfCells = string_to_num<int>(param_val);
      else if (param_name == "MaxPeanoHilbertLevel")
	runParams.nMaxPHLevel = string_to_num<int>(param_val);

      else if (param_name == "SimLabel")
	strcpy(runParams.cSimLabel, param_val.c_str());
      else if (param_name == "CreateSneplist")
	runParams.nFlagCreateSnepList = string_to_num<int>(param_val);

      else if (param_name == "Redo")
	runParams.nRedo = string_to_num<int>(param_val);

      else
	cout << "Unknown parameter " << param_name << endl;
      
      } // ends loop to read
      
      
  if (std::string(runParams.cOutputDir).compare("SimBaseDir") == 0)
    strcpy(runParams.cOutputDir, runParams.cSimBaseDir);

  if (runParams.nSnepshotListLoc == 0)
    strcpy(runParams.cSnepshotListName, (std::string(runParams.cSimBaseDir) + "/" + std::string(runParams.cSnepshotListName)).c_str());
  else
    strcpy(runParams.cSnepshotListName, (std::string(runParams.cOutputDir) + "/" + std::string(runParams.cSnepshotListName)).c_str());

  cout << rp() + "   ...read " << nParams << " parameters." << endl;
  
  cout << endl;
  cout << "---------------------------" << endl;

  cout << "  Settings: " << endl;
  cout << "    StartSnep            = " << runParams.nSnepIni << endl;
  cout << "    EndSnep              = " << runParams.nSnepFin << endl;
  cout << "    SimBaseDir           = " << runParams.cSimBaseDir << endl;
  cout << "    OutputFileName       = " << runParams.cOutputFileName << endl;
  cout << "    SnepshotListName     = " << runParams.cSnepshotListName << endl;
  cout << "    OutputDir            = " << runParams.cOutputDir << endl;

  cout << std::endl;

  cout << "    MapPartType          = " << runParams.nTypeCode << endl;
  cout << "    DesNumPerCell        = " << runParams.nDesNumPerCell << endl;
  cout << "    MaxCellSize          = " << runParams.dMaxCellSize << endl;
  cout << "    MinNumOfCells        = " << runParams.nMinNumOfCells << endl;
  cout << "    MinNumOfCells        = " << runParams.nMinNumOfCells << endl;
  cout << "    MaxNumOfCells        = " << runParams.nMaxNumOfCells << endl;
  cout << "    MaxPeanoHilbertLevel = " << runParams.nMaxPHLevel << endl;

  cout << std::endl;

  cout << "    SimLabel             = " << runParams.cSimLabel << endl;
  cout << "    CreateSnepList       = " << runParams.nFlagCreateSnepList << endl;

  cout << "---------------------------" << endl;
  cout << endl;

  
  return nParams;
      
}
  

void print_config_flags() {
  
  std::cout << std::endl;
  std::cout << "Code was compiled with following options:" << std::endl;
  std::cout << "-----------------------------------------" << std::endl;

#ifdef VERBOSE
  std::cout << "VERBOSE" << std::endl;
#endif

#ifdef DEBUG
  std::cout << "DEBUG" << std::endl;
#endif

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << std::endl;

  return;

}
