#include "mpi.h"
#include <cmath>

#include "utilities_mpi.h"
#include "io_hdf5.h"


template <typename T>
void eagleread(std::string sPrimaryFileName,    // Full path to primary file of data product
	       std::string sDataSetName,        // Full (internal) dataset name
	       std::vector<T> &vOutput,      // [O] The output vector

	       std::vector<long> &vTDOlist,        // [I/O] TDOlist - can be empty on input

	       int nBroadcastFlag,              // 0: no comm., 1: assemble on root, 2: +distr.
	       
	       std::vector<long> &vFDOlist,        // [I/O] FDOlist - can be empty on input
	       int nDimension,                  // For rank-2 datasets, which dimension to read
	       std::string sFileNumAttribute,   // Attrib. saying how many files there are
	       std::string sElemNumAttribute, // Elem. per file
	       char cFileNumAttributeType,     // Type of ^ (g[roup], [d]ataset)
	       char cElemNumAttributeType,
	       int nElemNumAttributeIndex,
	       int FlagAstroConv )


{
  
  using std::cout;
  using std::endl;
  using std::vector;
  using std::string;  


  function_switch("eagleread");
  
    /* Ok. So the general structure of this is as follows:

     1.) Check whether FDOlist is set up --> if not, do so.
     2.) Check whether TDOlist is set up --> if not, do so.
         [Do this later, because then we already know number of data elements]

     3.) Identify which Files/Sections need to be read for current task

     4.) Read in (local) data

     5.) If desired, assemble data on root and broadcast to everyone.
  
   */

  vOutput.clear();

  // ----------------------------------------------
  // 1.) Check whether supplied FDOlist is useful
  //     Convention: If its length is != 0, use it!
  // ----------------------------------------------

  if (check_fdo_list(vFDOlist, sPrimaryFileName, sDataSetName, sFileNumAttribute, cFileNumAttributeType, sElemNumAttribute, cElemNumAttributeType, nElemNumAttributeIndex) == 1)
    if (mpirank() == 0)
      cout << rp() + "FDO list was created successfully." << endl;
    
	
  // --------------------------------------------------------
  // 2.) Check whether supplied TDOlist is useful
  //     Convention: If its length is == numtasks, use it!
  // --------------------------------------------------------

  if (check_tdo_list(vTDOlist, vFDOlist.back()) == 1)
    if (mpirank() == 0)
      cout << rp() + "TDO list was created successfully." << endl;
  

  // --------------------------------------------------------
  // 3.) Build the crosslink list, i.e. which files to load
  //     for each task
  // --------------------------------------------------------

  std::vector<int> vFilesThisTask;
  std::vector<long> vFirstUsedInFile;
  std::vector<long> vFileOffsetInThisTask;

  identify_relevant_files (vFDOlist, vTDOlist.at(mpirank()), vTDOlist.at(mpirank()+1)-1, vFilesThisTask, vFirstUsedInFile, vFileOffsetInThisTask,0);
  
  
  // -----------------------------------------------------------
  // 4.) Read in local data -- this is the type-dependent part
  //     (but ONLY IMPLICITLY because of function overloading!)
  // -----------------------------------------------------------
  

  vOutput.resize(vFileOffsetInThisTask.back());

  // For astro-conv, need to record ONE file where there is data
  std::string sGoodFileName; 

  for (int iifile = 0; iifile < vFilesThisTask.size(); iifile++)
    {
      
#ifdef VERBOSE    
      print_n_times(iifile, vFilesThisTask.size(), 4, "Loading file");
#endif 
      
      long OffsetInCurrFile = vFirstUsedInFile.at(iifile);
      long LengthInCurrFile = vFileOffsetInThisTask.at(iifile+1) - vFileOffsetInThisTask.at(iifile);
      
      // Can end right here if there is nothing to read from current file!
      if (LengthInCurrFile <= 0)
	continue;
      
      
      std::string sFileNameCurr = change_file_sequence_nr(sPrimaryFileName, vFilesThisTask.at(iifile));
      sGoodFileName = sFileNameCurr;
      
      // Must specify to *not* resize the output, and transfer to specified 
      // offset in vOutput vector (!)
      read_hdf5_data(sFileNameCurr, sDataSetName, vOutput, OffsetInCurrFile, LengthInCurrFile, nDimension, 0, vFileOffsetInThisTask.at(iifile)); 
      
    } // ends loop through files which these tasks has to read


  // Additional bit added 2 Dec 15: Apply astro-conversion if desired
  if (FlagAstroConv == 1) {
    
    double dAstroConv;

    if (mpirank() == 0) {
      
      double aexp_exp = read_hdf5_attribute_double(sGoodFileName, sDataSetName + "/aexp-scale-exponent", 'd');
      double h_exp = read_hdf5_attribute_double(sGoodFileName, sDataSetName + "/h-scale-exponent", 'd');
      
      double aexp_curr = read_hdf5_attribute_double(sGoodFileName, "Header/ExpansionFactor", 'g');
      double hubble_curr = read_hdf5_attribute_double(sGoodFileName, "Header/HubbleParam", 'g');

      dAstroConv = pow(aexp_curr, aexp_exp) * pow(hubble_curr, h_exp);
    }
    
    MPI_Bcast(&dAstroConv, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (size_t ii = 0; ii < vOutput.size(); ii++)
      vOutput.at(ii) *= dAstroConv;

  }
  
  
  // -----------------------------------
  // 5.) Re-distribute data if required
  // -----------------------------------
  
  if (nBroadcastFlag > 0) {
    collect_vector_mpi<T> (vOutput, 0, 0, nBroadcastFlag);
  }
  
  
  function_switch("eagleread");

  return;
  
}
