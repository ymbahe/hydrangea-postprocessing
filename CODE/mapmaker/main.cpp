// *****************************************************************
// ***************                  ********************************
// ***************     MAPMAKER     ********************************
// ***************                  ********************************
// *****************************************************************


// -------------------------------------------------------------------------------
// ---- Program to create particle maps to enable easy loading of subregions -----
// ---------------------------   Started 4 Dec 2015 ------------------------------
// -------------------------------------------------------------------------------


// General modification-switch file:
#include "Config.h" 

// Standard libraries:
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <cstring>
#include <stdlib.h>


#include "/u/ybahe/cpplib/init_hydrangea.h" 
#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/utilities_mpi.h"
#include "/u/ybahe/cpplib/hydrangea.h"
#include "/u/ybahe/cpplib/io_hdf5.h"
#include "/u/ybahe/cpplib/correlate_ids.h"

// Small helpers and global vars:
#include "init.h"
#include "globals_declare.h"


// Headers for sub-files:
// (will go here as program is created)

#include "mapmaker_tools.h"


// **************************
// MAPMAKER's MAIN FUNCTION
// **************************


int main(int argc, char *argv[])
{
  using namespace std;

  // ---- MPI initialisation ----
  
 
  MPI_Init(&argc, &argv);
  rp("[" + to_string(mpirank()) + "]: ");


  // ---- Welcome messages -------
  
  if (mpirank() == 0) {
     
    cout << endl;
    cout << endl;
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;
    cout << endl;
    cout << rp() + "Welcome to MAPMAKER (" << numtasks() << " cores)!" << endl;
    cout << endl;
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;
    cout << endl;
  }

  // ----------------------------
  // Program-specific setup here:
  // ----------------------------

  if (sizeof(int) != 4 || sizeof(long) != 8) {
    cout << "Sizes are not standard on this machine. Exiting..." << endl;
    exit(74); }
  
  double dDummy = ElTime(), dProgStartTime = GetTime();
  rt(); /* Initialize cumulative run time counter */
  
  // Default parameter file name - can be overridden from command line
  string sParamFile = "sniplocate.param";  


  // ============================================================
  // First: Deal with input parameters (snapshot and halo ranges)
  // ============================================================
    
  
  // ----- Read in the run parameters from file  ---

  if (mpirank() == 0) {
    load_input_parameters(argc, argv, sParamFile);
    int nFileParams = read_parameter_file(sParamFile, Params);  


  }
  broadcast_struct<RunParams> (Params);
  

  // ------ Read in snepshot list ------

  std::vector<Snepshot> vSnepshotList; 
  int nSnepshots;

  if (mpirank() == 0) {

    if (file_test(Params.cSnepshotListName) == 0) {
      if (Params.nFlagCreateSnepList == 1) {
      	std::cout << rp() + "Creating snepshot list..." << std::endl;
	std::string syscall = "/u/ybahe/anaconda3/bin/python3.4 /u/ybahe/ANALYSIS/make_full_sneplist.py " + std::string(Params.cSimBaseDir);
	system(syscall.c_str());
      } else {
	std::cout << "The snepshot list '" << Params.cSimBaseDir << "' does not exist, and you have specified not to create one automatically. Please manually run make_full_sneplist.py and get back afterwards." << std::endl;
	exit(222);
      }
    }
    
    std::vector<Snepshot> vFullSnepList = read_snepshot_list(Params.cSnepshotListName, Params.nSnepshotListEntries);
    
    if (Params.nSnepFin == -1)
      Params.nSnepFin = vFullSnepList.size()-1;

    nSnepshots = Params.nSnepFin - Params.nSnepIni + 1;

    vSnepshotList.resize(nSnepshots);

    for (int iisnep = 0; iisnep < nSnepshots; iisnep++)
      vSnepshotList.at(iisnep) = vFullSnepList.at(Params.nSnepIni+iisnep);
  }
  
  // It would be a good idea to communicate this list to all tasks...
  broadcast_vector<Snepshot> (vSnepshotList, 0);
  nSnepshots = vSnepshotList.size();

  // ==========================================================
  // Declare variables to be used across snepshots here, so they 
  // are not re-set at the beginning of each snapshot iteration!
  // ===========================================================

  // Not sure there are many/any of them here in mapmaker...
  std::string sFirstSnepshotDir = eagle_filename(string(Params.cSimBaseDir), vSnepshotList.front(), 0, mpirank());
  g_dBoxSize = read_hdf5_attribute_double(sFirstSnepshotDir, "Header/BoxSize");

  // Find particle types
  std::vector<int> vTypeVec = expand_parttype_code(Params.nTypeCode);
      
  if (vTypeVec.size() == 0) {
    if (mpirank() == 0) {
    std::cout << "There are no particle types to process." << std::endl;
    }    

    MPI_Finalize;
    return 0;
  }

    
  // ------------------------------------------------------------
  // ------------------ Now begin the main loop -----------------
  // ------------------ (over different snepshots) --------------
  // ------------------------------------------------------------


  for (int tt = 0; tt < nSnepshots; tt++) { 
    
    double dItStartTime = GetTime();
    
    // make global copies for ease of access from functions
    g_Snepshot = vSnepshotList.at(tt);
    g_ttSnep = tt;  

    // ---------------------------------------------------
    // And now set up variables specific to this iteration
    // ---------------------------------------------------
    
    g_SnepshotDir = eagle_filename(string(Params.cSimBaseDir), g_Snepshot, 0, mpirank());
    
    double dZRed = read_hdf5_attribute_double(g_SnepshotDir, "Header/Redshift");
    double dSimTime = read_hdf5_attribute_double(g_SnepshotDir, "Header/Time");
    

    if (mpirank() == 0) {
      std::cout << std::endl;
      std::cout << "***********************************************" << std::endl;
      std::cout << "*** Start time step " << tt+1 << " / " << nSnepshots << " *** " << rt() << std::endl;
      std::cout << "******* Redshift = " << dZRed << " *******" << std::endl;
   std::cout << "***********************************************" << std::endl;
      std::cout << std::endl;
      std::cout << rp() + "Snepshot type = " << g_Snepshot.Type << ", Snepshot number = " << g_Snepshot.Num << std::endl;
    }

    

    // Some overview:
    // 1.) Load particle coordinates (distributed over all tasks)
    // 2.) Determine particle bounding box (globally)
    // 3.) Determine PH level and cell bounding box
    // XX 4.) Determine cell sequence (sorted by Peano-Hilbert number) << skipped
    // 5.) Make local particle offset list (into cell sequence numbers)
    // 6.) Combine local offset lists into global one [on rank==0]
    // 7.) Write output for current snepshot --> in SEPARATE file in .../snAIpshotXXX/ dir
    
    
    // Preamble: set up output file for current snepshot
    //           (also copy relevant header attributes)
    
    std::string sCurrDir = file_to_dir(g_SnepshotDir);
    std::vector<std::string> vsCurrParts = split_string(g_SnepshotDir, '/');
    int nCurrParts = vsCurrParts.size();
    std::string sOutDirCoda = join_strings(vsCurrParts, '/', nCurrParts-3, nCurrParts-2); 
    std::string sCurrOutDir = std::string(Params.cOutputDir) + "/" + sOutDirCoda;
    std::string sCurrOutFile = sCurrOutDir + "/" + std::string(Params.cOutputFileName);

    int nFileExists = 0;

    if (mpirank() == 0) {
      if (std::ifstream(sCurrOutFile.c_str()))
	{
	  std::cout << "Map file already exists" << std::endl;
	  /*nFileExists = 1;*/
	}
    }
    MPI_Bcast(&nFileExists, 1, MPI_INT, 0, MPI_COMM_WORLD);     

    if (nFileExists == 1 && Params.nRedo == 0) {
      if (mpirank() == 0)
	std::cout << "Skipping this snep..." << std::endl; 
      continue;
    }
    
    if (mpirank() == 0) {
      std::string syscall = "mkdir -p " + sCurrOutDir;
      system(syscall.c_str());
      hdf5_create_file(sCurrOutFile);

      long nNumFilesPerSnapshot = read_hdf5_attribute_long(g_SnepshotDir, "Header/NumFilesPerSnapshot");
      std::vector<long> vNumPartTotal = read_hdf5_attribute_vlong(g_SnepshotDir, "Header/NumPart_Total");
      
      hdf5_create_group(sCurrOutFile, "Header");
      hdf5_write_attribute(sCurrOutFile, "Header", "BoxSize", g_dBoxSize, 'g');
      hdf5_write_attribute(sCurrOutFile, "Header", "NumFilesPerSnapshot", nNumFilesPerSnapshot, 'g');
      hdf5_write_attribute_array(sCurrOutFile, "Header", "NumPart_Total", vNumPartTotal, 'g');
      hdf5_write_attribute(sCurrOutFile, "Header", "Redshift", dZRed, 'g');
      hdf5_write_attribute(sCurrOutFile, "Header", "SnepshotType", g_Snepshot.Type, 'g');
      hdf5_write_attribute(sCurrOutFile, "Header", "SnepshotNumber", g_Snepshot.Num, 'g');
      hdf5_write_attribute(sCurrOutFile, "Header", "Time", dSimTime, 'g');
      hdf5_write_attribute(sCurrOutFile, "Header", "SimLabel", std::string(Params.cSimLabel), 'g');
    }

    
    for (char iitype = 0; iitype < vTypeVec.size(); iitype++) {
      
      if (mpirank() == 0) {
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << rp() + "Processing particle type " << vTypeVec.at(iitype) << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      }

      g_type = vTypeVec.at(iitype); // for ease of access from functions
      
      std::string g_sTypeName = "PartType" + to_string(vTypeVec.at(iitype));
      

      // =================================
      // PART I: LOAD PARTICLE COORDINATES
      // =================================
      
      std::vector<long> vTPOlist, vFPOlist;
      std::vector<double> vTT_x, vTT_y, vTT_z;
      
      // NB: We specify 'junktest' for the FileLookup variable to ensure it is using 
      //     the FPO/TPOlists constructed from the x reading.
      
      eagleread<double> (g_SnepshotDir, g_sTypeName + "/Coordinates", vTT_x, vTPOlist, 0, vFPOlist, 0, "Header/NumFilesPerSnapshot", "Header/NumPart_ThisFile", 'g', 'g', g_type, 0);
      eagleread<double> (g_SnepshotDir, g_sTypeName + "/Coordinates", vTT_y, vTPOlist, 0, vFPOlist, 1, "junktest");
      eagleread<double> (g_SnepshotDir, g_sTypeName + "/Coordinates", vTT_z, vTPOlist, 0, vFPOlist, 2, "junktest");

      // Special section: if there are no particles, end here!
      if (vFPOlist.back() == 0) {
	if (mpirank() == 0)
	  std::cout << rp() + "No particles of " << g_sTypeName << " in current snepshot " << tt << std::endl;
	continue;
      }

      // PART II: Determine global bounding box
      
      dDummy = ElTime();
      if (mpirank() == 0)
	std::cout << rp() + "Find global bounding box of particles..." << std::endl;
      
      std::vector<double> minmax_x = find_global_minmax<double>(vTT_x);
      std::vector<double> minmax_y = find_global_minmax<double>(vTT_y);
      std::vector<double> minmax_z = find_global_minmax<double>(vTT_z);

      report<double> (ElTime(), "Determining global bounding box took [sec.]");


      if (mpirank() == 0) {
	std::cout << rp() + "Region to be mapped was identified as follows:" << std::endl;
	std::cout << rp() + "   x: " << minmax_x.front() << " --> " << minmax_x.back() << std::endl;
	std::cout << rp() + "   y: " << minmax_y.front() << " --> " << minmax_y.back() << std::endl;
	std::cout << rp() + "   z: " << minmax_z.front() << " --> " << minmax_z.back() << std::endl;

      }

      std::vector<double> vPartBounds(6);
      vPartBounds.at(0) = minmax_x.front();
      vPartBounds.at(1) = minmax_y.front();
      vPartBounds.at(2) = minmax_z.front();
      vPartBounds.at(3) = minmax_x.back();
      vPartBounds.at(4) = minmax_y.back();
      vPartBounds.at(5) = minmax_z.back();

      minmax_x.clear();
      minmax_y.clear();
      minmax_z.clear();

      // NB: Do not clear position vectors yet, as they will still be required later

      // ===================================================
      // PART III: DETERMINE PH LEVEL AND CELL BOUNDING BOX
      // ===================================================

      std::vector<double> vCellMins; // to hold minimum CELL position in x, y, z
      std::vector<int> vNumCells; // to hold number of cells per dimension
      int nPHLevel = find_ph_level(vPartBounds, vCellMins, vNumCells);
      
      // Make local particle offset+length lists

      long nNumCellsTot = static_cast<long>(vNumCells.at(0)) * static_cast<long>(vNumCells.at(1)) * static_cast<long>(vNumCells.at(2));
      std::vector<long> vTT_CellMinIndex(nNumCellsTot, -1);
      std::vector<long> vTT_CellMaxIndex(nNumCellsTot, -1);

      double dCellMinX = vCellMins.at(0);
      double dCellMinY = vCellMins.at(1);
      double dCellMinZ = vCellMins.at(2);
      double dCellSize = g_dBoxSize/ipow(2,nPHLevel);

      long nNumCellsX = vNumCells.at(0);
      long nNumCellsY = vNumCells.at(1);
      long nNumCellsZ = vNumCells.at(2);

      if (mpirank() == 0) {
	std::cout << rp() + "Cell size = " << dCellSize << " cMpc/h" << std::endl;
	std::cout << rp() + "CellRegionCorner = [" << dCellMinX << ", " << dCellMinY << ", " << dCellMinZ << "]" << std::endl;
	std::cout << rp() + "Number of Cells = [" << nNumCellsX << ", " << nNumCellsY << ", " << nNumCellsZ << "]" << std::endl;
	std::cout << rp() + "   (total cell number = " << nNumCellsTot << ")" << std::endl;
      }

      long nPartOffset = vTPOlist.at(mpirank());

      dDummy = ElTime();
      if (mpirank() == 0) 
	std::cout << rp() + "Looping through local particles..." << std::endl;
      // --> loop through all particles on current task and find which cell they sit in
      for (long iipart = 0; iipart < vTT_x.size(); iipart++) {
	double dRelX = vTT_x.at(iipart) - dCellMinX;
	double dRelY = vTT_y.at(iipart) - dCellMinY;
	double dRelZ = vTT_z.at(iipart) - dCellMinZ;
      
	long nCellX = static_cast<long> (dRelX/dCellSize);
	long nCellY = static_cast<long> (dRelY/dCellSize);
	long nCellZ = static_cast<long> (dRelZ/dCellSize);

	long nCellIndexCurr = nCellX + nCellY * nNumCellsX + nCellZ * nNumCellsY * nNumCellsX;

	// Case I: No particle in this cell so far
	if (vTT_CellMinIndex.at(nCellIndexCurr) < 0) {
	  vTT_CellMinIndex.at(nCellIndexCurr) = iipart + nPartOffset;
	  vTT_CellMaxIndex.at(nCellIndexCurr) = iipart + nPartOffset;
	}
	else {
	  // Case II: Cell already populated.
	  // In this case, current particle is new maximum inhabitant
	  vTT_CellMaxIndex.at(nCellIndexCurr) = iipart + nPartOffset;
	}
      } // ends loop through particles

      report<double> (ElTime(), "Going through local particle list took [sec.]");

      // Correlate lists across tasks
      
      std::vector<long> vDummy;
      find_minmax_vector<long> (vTT_CellMinIndex, vDummy, 0, 1, 0, 1, 0, 0);
      find_minmax_vector<long> (vTT_CellMaxIndex, vDummy, 1, 1, 0, 0, 0, 0);  //max w/o thresh
      //                                            ------^  ^  ^\  ^  ^--          
      //                                           /        /     \  \-----  \---------
      //                       Input   [Output]  Min/Max  InPlace  BCast, FlagThresh, Threshold, Verb

      // Task 0 now has the complete list --> convert to output and write
      
      if (mpirank() != 0) {
	vTT_CellMinIndex.clear();
	vTT_CellMaxIndex.clear();
      }
      
      else {

	std::vector<unsigned int> vCellMinFull(nNumCellsTot, 0);
	std::vector<unsigned int> vCellLengthFull(nNumCellsTot, 0);
	
	// Go through each cell and set up its offset/length
	
	long nSumCellCounts = 0;
	int nCellsEmpty = 0;
	for (int iicell = 0; iicell < nNumCellsTot; iicell++) {
	  if (vTT_CellMinIndex.at(iicell) < 0) {
	    nCellsEmpty++;
	    continue;
	  }
	  
	  vCellMinFull.at(iicell) = static_cast<unsigned int> (vTT_CellMinIndex.at(iicell));
	  vCellLengthFull.at(iicell) = static_cast<unsigned int> (vTT_CellMaxIndex.at(iicell)-vTT_CellMinIndex.at(iicell)+1);
	  nSumCellCounts += static_cast<long> (vCellLengthFull.at(iicell));
	  
	}
	
	std::cout << rp() + "Cell counts for particle type " << g_type << " sum to " << nSumCellCounts << std::endl;
	std::cout << rp() + "   (overload factor = " << static_cast<double>(nSumCellCounts)/vFPOlist.back() << ")" << std::endl;
	std::cout << rp() + "Out of " << nNumCellsTot << " cells, " << nCellsEmpty << " are empty (= " << static_cast<double>(nCellsEmpty)/nNumCellsTot*100 << " per cent)" << std::endl;


	// ------------------
	// WRITE OUTPUT (!!!)
	// ------------------
	
	hdf5_create_group(sCurrOutFile, g_sTypeName);
	
	// *** File Offset list ***
	hdf5_create_dataset(sCurrOutFile, g_sTypeName + "/FileOffset", std::vector<int>(1,vFPOlist.size()), "unsigned int", "Offsets in individual files on disk, i.e. the first particle index stored in each file. The last entry gives the total number of particles.");
	std::vector<unsigned int> vFPOlist_UI(vFPOlist.size());
	for (int ii = 0; ii < vFPOlist.size(); ii++)
	  vFPOlist_UI.at(ii) = static_cast<unsigned int> (vFPOlist.at(ii));
	write_hdf5_data(sCurrOutFile, g_sTypeName + "/FileOffset", vFPOlist_UI);

	// *** Cell Offset list ***
	hdf5_create_dataset(sCurrOutFile, g_sTypeName + "/CellOffset", std::vector<int>(1,nNumCellsTot), "unsigned int", "Offsets for each cell, i.e. the first particle index stored in it. Note that this is *not* a contiguous offset list, so the particles belonging to cell i range from CellOffset[i] --> CellOffset[i]+CellCount[i]-1, not to CellOffset[i+1]-1. Also note that the value stored here is meaningless for cells with CellCount of 0.");
	write_hdf5_data(sCurrOutFile, g_sTypeName + "/CellOffset", vCellMinFull);

	// *** Cell Count list ***
	hdf5_create_dataset(sCurrOutFile, g_sTypeName + "/CellCount", std::vector<int>(1,nNumCellsTot), "unsigned int", "Number of particles in each cell, which may be zero. The indices of particles in each cell may be found as CellOffset[i] --> CellOffset[i] + CellCount[i]-1. Note that, in principle, the sum of all CellCount entries may exceed the total number of particles, if particles are not ordered perfectly.");
	write_hdf5_data(sCurrOutFile, g_sTypeName + "/CellCount", vCellLengthFull);
	
	// And some particle-type specific info in attributes:
	hdf5_write_attribute(sCurrOutFile, g_sTypeName, "NumPart", vFPOlist.back(), 'g');
	hdf5_write_attribute(sCurrOutFile, g_sTypeName, "PeanoHilbertLevel", nPHLevel, 'g');
	hdf5_write_attribute(sCurrOutFile, g_sTypeName, "NumCellsTot", nNumCellsTot, 'g');
	hdf5_write_attribute_array(sCurrOutFile, g_sTypeName, "NumCellsPerDim", vNumCells, 'g');
	hdf5_write_attribute(sCurrOutFile, g_sTypeName, "CellSize", dCellSize, 'g');
	hdf5_write_attribute_array(sCurrOutFile, g_sTypeName, "CellRegionCorner", vCellMins, 'g');
	
      } // ends section for root only
    } // ends loop through particle types

    if (mpirank() == 0) {
      cout << endl;
      cout << "******************************************************************" << endl;
      cout << "Snepshot " << tt << " took " <<  GetTime() - dItStartTime << " sec. " << rt() << endl;
      cout << "******************************************************************" << endl; 
      cout << endl;
    } 
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    
  } // ends loop through snepshots
  
  if (mpirank() == 0) {
    cout << endl;
    cout << rp() + "MAPMAKER took " << GetTime() - dProgStartTime << " sec. to map " << nSnepshots << " snepshots (average = " << (GetTime()-dProgStartTime)/nSnepshots << " sec./snep)" << endl;
  }
  
  MPI_Finalize();
  return 0;
}







