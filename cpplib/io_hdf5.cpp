// HDF5 handling routines
// The idea is that most/all direct interfacing with the *** HDF5 library is dealt with here.
// Started 22 Nov 2015

#include "mpi.h"
#include <vector>
#include <string>
#include <iostream>
#include "H5Cpp.h"

//#include "/u/ybahe/ANALYSIS/Protea-H/Config.h"
//#include "/u/ybahe/ANALYSIS/Protea-H/globals.h"

#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/io_hdf5.h"



int hdf5_measure_attribute(H5::Attribute attr);


long read_hdf5_attribute_long(std::string sFileName,
			      std::string sAttribName,
			      char nAttribType,
			      int nAttribIndex) {

  H5::H5File file (sFileName, H5F_ACC_RDONLY);

  std::vector<std::string> vAttribParts = split_string(sAttribName, '/');
 
  if (vAttribParts.size() < 2) {
    std::cout << "Cannot determine attribute parts in read_hdf5_attribute_long()." << std::endl;
    exit(231); }
  
  std::string sContainerName = join_strings(vAttribParts, '/', 0, vAttribParts.size()-2);
  std::string sAttributeName = vAttribParts.back();


  H5::Attribute attr;

  if (nAttribType == 'g' || nAttribType == 'G') {
    H5::Group group = file.openGroup(sContainerName);
    attr = group.openAttribute(sAttributeName);
  } else {
    H5::DataSet dataset = file.openDataSet(sContainerName);
    attr = dataset.openAttribute(sAttributeName);
  }

  H5::DataType attrType = attr.getDataType();
  int size = hdf5_measure_attribute(attr);

  long output = 0;

  if (attrType == H5::PredType::NATIVE_LONG) {
    long *TargVal = new long[size];
    attr.read(attrType, TargVal);
    output = static_cast<long>(TargVal[nAttribIndex]);
  }

  else if (attrType == H5::PredType::NATIVE_INT) {
    int *TargVal = new int[size];
    attr.read(attrType, TargVal);
    output = static_cast<long>(TargVal[nAttribIndex]);
  }

  else if (attrType == H5::PredType::NATIVE_UINT) {
    unsigned int *TargVal = new unsigned int[size];
    attr.read(attrType, TargVal);
    output = static_cast<long>(TargVal[nAttribIndex]);
  }
  
  
  else {
    std::cout << "It would appear that the attribute '" << sAttributeName << "' is neither LONG nor INT, and cannot be read with read_hdf5_attribute_long()." << std::endl;
    exit(999);
  }
     

  return output;

}



std::string read_hdf5_attribute_string(std::string sFileName,
				       std::string sAttribName,
				       char nAttribType,
				       int nAttribIndex) {

  H5::H5File file (sFileName, H5F_ACC_RDONLY);

  std::vector<std::string> vAttribParts = split_string(sAttribName, '/');
 
  if (vAttribParts.size() < 2) {
    std::cout << "Cannot determine attribute parts in read_hdf5_attribute_string)." << std::endl;
    exit(231); }
  
  std::string sContainerName = join_strings(vAttribParts, '/', 0, vAttribParts.size()-2);
  std::string sAttributeName = vAttribParts.back();


  H5::Attribute attr;

  if (nAttribType == 'g' || nAttribType == 'G') {
    H5::Group group = file.openGroup(sContainerName);
    attr = group.openAttribute(sAttributeName);
  } else {
    H5::DataSet dataset = file.openDataSet(sContainerName);
    attr = dataset.openAttribute(sAttributeName);
  }

  H5::DataType attrType = attr.getDataType();
  /*int size = hdf5_measure_attribute(attr); */
  size_t size = attrType.getSize();

  std::string output;
  char *TargVal = new char[size];
  attr.read(attrType, TargVal);
  output = std::string(TargVal);
  
  /*
  if (attrType == H5::PredType::NATIVE_CHAR) {
    char *TargVal = new char[size];
    attr.read(attrType, TargVal);
    output = std::string(TargVal);
  }

  if (attrType == H5T_STRING) {
    char *TargVal = new char[size];
    attr.read(attrType, TargVal);
    output = std::string(TargVal);
  }
  

  else {
    std::cout << "It would appear that the attribute '" << sAttributeName << "' is not CHAR or H5T_STRING, and cannot be read with read_hdf5_attribute_string()." << std::endl;
    exit(999);
  }
  
  */

  return TargVal;

}



std::vector<long> read_hdf5_attribute_vlong(std::string sFileName,
					    std::string sAttribName,
					    char nAttribType) {

  
  H5::H5File file (sFileName, H5F_ACC_RDONLY);
  
  std::vector<std::string> vAttribParts = split_string(sAttribName, '/');
 
  if (vAttribParts.size() < 2) {
    std::cout << "Cannot determine attribute parts in read_hdf5_attribute_vlong()." << std::endl;
    exit(231); }
  
  std::string sContainerName = join_strings(vAttribParts, '/', 0, vAttribParts.size()-2);
  std::string sAttributeName = vAttribParts.back();
  
  H5::Attribute attr;

  if (nAttribType == 'g' || nAttribType == 'G') {
    H5::Group group = file.openGroup(sContainerName);
    attr = group.openAttribute(sAttributeName);
  } else {
    H5::DataSet dataset = file.openDataSet(sContainerName);
    attr = dataset.openAttribute(sAttributeName);
  }

  H5::DataType attrType = attr.getDataType();
  int size = hdf5_measure_attribute(attr);

  std::vector<long> output(size,0);
  
  if (attrType == H5::PredType::NATIVE_LONG) {
    long *TargVal = new long[size];
    attr.read(attrType, TargVal);
    for (int ii = 0; ii < size; ii++)
      output.at(ii) = static_cast<long>(TargVal[ii]);
    delete[] TargVal;
  }
  
  else if (attrType == H5::PredType::NATIVE_INT) {
    int *TargVal = new int[size];
    attr.read(attrType, TargVal);
    for (int ii = 0; ii < size; ii++)
      output.at(ii) = static_cast<long>(TargVal[ii]);
    delete[] TargVal;
  }

  else if (attrType == H5::PredType::NATIVE_UINT) {
    unsigned int *TargVal = new unsigned int[size];
    attr.read(attrType, TargVal);
    for (int ii = 0; ii < size; ii++)
      output.at(ii) = static_cast<long>(TargVal[ii]);
    delete[] TargVal;
  }
  
  
  else {
    std::cout << "It would appear that the attribute '" << sAttributeName << "' is neither LONG nor INT, and cannot be read with read_hdf5_attribute_vlong()." << std::endl;
    exit(999);
  }
  
  
  return output;

}


double read_hdf5_attribute_double(std::string sFileName,
				  std::string sAttribName,
				  char nAttribType,
				  int nAttribIndex) {


  H5::H5File file (sFileName, H5F_ACC_RDONLY);

  std::vector<std::string> vAttribParts = split_string(sAttribName, '/');
 
  if (vAttribParts.size() < 2) {
    std::cout << "Cannot determine attribute parts in read_hdf5_attribute_long()." << std::endl;
    exit(231); }
  
  std::string sContainerName = join_strings(vAttribParts, '/', 0, vAttribParts.size()-2);
  std::string sAttributeName = vAttribParts.back();


  H5::Attribute attr;

  if (nAttribType == 'g' || nAttribType == 'G') {
    H5::Group group = file.openGroup(sContainerName);
    attr = group.openAttribute(sAttributeName);
  } else {
    H5::DataSet dataset = file.openDataSet(sContainerName);
    attr = dataset.openAttribute(sAttributeName);
  }

  H5::DataType attrType = attr.getDataType();

  int size = hdf5_measure_attribute(attr);
  double output = 0;
  
  if (attrType == H5::PredType::NATIVE_FLOAT) {
    float *TargVal = new float[size];
    attr.read(attrType, TargVal);
    output = static_cast<double>(TargVal[nAttribIndex]);
  }

  else if (attrType == H5::PredType::NATIVE_DOUBLE) {
    double *TargVal = new double[size];
    attr.read(attrType, TargVal);
    output = static_cast<double>(TargVal[nAttribIndex]);
  }

  else if (attrType == H5::PredType::NATIVE_INT) {
    int *TargVal = new int[size];
    attr.read(attrType, TargVal);
    output = static_cast<double>(TargVal[nAttribIndex]);
  }

  else if (attrType == H5::PredType::NATIVE_UINT) {
    unsigned int *TargVal = new unsigned int[size];
    attr.read(attrType, TargVal);
    output = static_cast<double>(TargVal[nAttribIndex]);
  }
  

  else if (attrType == H5::PredType::NATIVE_LONG) {
    long *TargVal = new long[size];
    attr.read(attrType, TargVal);
    output = static_cast<double>(TargVal[nAttribIndex]);
  }
  
  else {
    std::cout << "It would appear that the attribute '" << sAttributeName << "' is neither DOUBLE, FLOAT, INT, or LONG, and cannot be read with read_hdf5_attribute_double()." << std::endl;
    exit(999);
  }
     

  return output;

}


int hdf5_measure_attribute(H5::Attribute attr) {

  H5::DataSpace attrDSpace = attr.getSpace();
  
  int rank = attrDSpace.getSimpleExtentNdims();
  
  if (rank > 1) {
    std::cout << "Reading attributes of rank > 1 is not currently implemented." << std::endl;
    exit(99);
  }
 
  int size = 1;
  if (rank > 0) {
    hsize_t *dims = new hsize_t[rank];
    int ndims = attrDSpace.getSimpleExtentDims(dims, NULL);
    size = dims[0];
    delete[] dims;
  }

  return size;
}


// LONG version
void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<long> &vOut,                   // [O] data
		    long pos_start,                 // first entry to read...
		    long pos_length,                // ... and how many in total
		    int nDim,                      // Dimension to be read
		    int FlagResizeOutput,
		    long nOutputOffset) {


  using namespace std;
  using namespace H5;

  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);
  int rank = vDataSetDims.size();

  if (rank > 2) {
    std::cout << "Reading 3+-dimensional HDF5 data is not currently implemented. Sorry..." << std::endl;
    exit(333);
  }

  if (pos_length < 0)
    pos_length = vDataSetDims.front();
  
#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data/LONG");
  
  cout << rp() + "sFileName  = " << sFileName << endl;
  cout << rp() + "pos_start  = " << pos_start << endl;
  cout << rp() + "pos_length = " << pos_length << endl;
#endif  

  H5File file (sFileName, H5F_ACC_RDONLY);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace dataspace = dataset.getSpace();
  
  hsize_t *count = new hsize_t[rank];
  hsize_t *start = new hsize_t[rank]; 

  start[0] = pos_start;  
  count[0] = pos_length;

  if (rank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  dataspace.selectHyperslab(H5S_SELECT_SET, count, start); 
  
  hsize_t *dim = new hsize_t[1];
  dim[0] = pos_length;
  
  DataSpace memspace(1, dim);

  if (FlagResizeOutput == 1)
    vOut.resize(pos_length);

  H5::DataType dType = dataset.getDataType();
  if (dType == H5::PredType::NATIVE_LONG) {
    if (pos_length > 0)
      dataset.read(&vOut.at(nOutputOffset), PredType::NATIVE_LONG, memspace, dataspace);
  }
  else {
    std::cout << "The dataset '" + sDataSetName + "' is not in LONG format." << std::endl;
    exit(666);
  }
  

#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data/LONG");
#endif

  delete[] count;
  delete[] start;
  delete[] dim;

  dataset.close();
  file.close();

  return;

}

// ULONG version
void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<unsigned long> &vOut,                   // [O] data
		    long pos_start,                 // first entry to read...
		    long pos_length,                // ... and how many in total
		    int nDim,                      // Dimension to be read
		    int FlagResizeOutput,
		    long nOutputOffset) {


  using namespace std;
  using namespace H5;

  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);
  int rank = vDataSetDims.size();

  if (rank > 2) {
    std::cout << "Reading 3+-dimensional HDF5 data is not currently implemented. Sorry..." << std::endl;
    exit(333);
  }

  if (pos_length < 0)
    pos_length = vDataSetDims.front();
  
#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data/ULONG");
  
  cout << rp() + "sFileName  = " << sFileName << endl;
  cout << rp() + "pos_start  = " << pos_start << endl;
  cout << rp() + "pos_length = " << pos_length << endl;
#endif  

  H5File file (sFileName, H5F_ACC_RDONLY);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace dataspace = dataset.getSpace();
  
  hsize_t *count = new hsize_t[rank];
  hsize_t *start = new hsize_t[rank]; 

  start[0] = pos_start;  
  count[0] = pos_length;

  if (rank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  dataspace.selectHyperslab(H5S_SELECT_SET, count, start); 
  
  hsize_t *dim = new hsize_t[1];
  dim[0] = pos_length;
  
  DataSpace memspace(1, dim);

  if (FlagResizeOutput == 1)
    vOut.resize(pos_length);

  H5::DataType dType = dataset.getDataType();
  if (dType == H5::PredType::NATIVE_ULONG) {
    if (pos_length > 0)
      dataset.read(&vOut.at(nOutputOffset), PredType::NATIVE_ULONG, memspace, dataspace);
  }
  else {
    std::cout << "The dataset '" + sDataSetName + "' is not in ULONG format." << std::endl;
    exit(666);
  }


#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data/ULONG");
#endif

  delete[] count;
  delete[] start;
  delete[] dim;

  dataset.close();
  file.close();

  return;

}


// INT version
void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<int> &vOut,           // [O] data
		    long pos_start,                 // first entry to read...
		    long pos_length,                // ... and how many in total
		    int nDim,                       // Dimension to be read
		    int FlagResizeOutput,
		    long nOutputOffset) {     


  using namespace std;
  using namespace H5;

#ifdef DEBUG
  function_switch("read_hdf5_data/INT");
  std::cout << rp() + "Attempting to read dataset '" << sDataSetName << "' from file '" << sFileName << "..." << std::endl;
#endif

  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);
  int rank = vDataSetDims.size();

  if (rank > 2) {
    std::cout << "Reading 3+-dimensional HDF5 data is not currently implemented. Sorry..." << std::endl;
    exit(333);
  }

  if (pos_length < 0)
    pos_length = vDataSetDims.front();
  

  H5File file (sFileName, H5F_ACC_RDONLY);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace dataspace = dataset.getSpace();
  
  hsize_t *count = new hsize_t[rank];
  hsize_t *start = new hsize_t[rank]; 

  start[0] = pos_start;  
  count[0] = pos_length;

  if (rank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  dataspace.selectHyperslab(H5S_SELECT_SET, count, start); 
  
  hsize_t *dim = new hsize_t[1];
  dim[0] = pos_length;
  
  DataSpace memspace(1, dim);

  if (FlagResizeOutput == 1)
    vOut.resize(pos_length);
  
  H5::DataType dType = dataset.getDataType();
  if (dType == H5::PredType::NATIVE_INT) {
    if (pos_length > 0)
      dataset.read(&vOut.at(nOutputOffset), PredType::NATIVE_INT, memspace, dataspace);
  }
  else {
    std::cout << "The dataset '" + sDataSetName + "' is not in INT format." << std::endl;
    exit(666);
  }

#ifdef DEBUG
  function_switch("read_hdf5_data/INT");
#endif

  delete[] count;
  delete[] start;
  delete[] dim;

  dataset.close();
  file.close();

  return;

}

// UNSIGNED INT version
void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<unsigned int> &vOut,           // [O] data
		    long pos_start,                 // first entry to read...
		    long pos_length,                // ... and how many in total
		    int nDim,                       // Dimension to be read
		    int FlagResizeOutput,
		    long nOutputOffset) {     


  using namespace std;
  using namespace H5;

#ifdef DEBUG
  function_switch("read_hdf5_data/UINT");
  std::cout << rp() + "Attempting to read UNSIGNED INT dataset '" << sDataSetName << "' from file '" << sFileName << "..." << std::endl;
#endif

  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);
  int rank = vDataSetDims.size();

  if (rank > 2) {
    std::cout << "Reading 3+-dimensional HDF5 data is not currently implemented. Sorry..." << std::endl;
    exit(333);
  }

  if (pos_length < 0)
    pos_length = vDataSetDims.front();
  

  H5File file (sFileName, H5F_ACC_RDONLY);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace dataspace = dataset.getSpace();
  
  hsize_t *count = new hsize_t[rank];
  hsize_t *start = new hsize_t[rank]; 

  start[0] = pos_start;  
  count[0] = pos_length;

  if (rank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  dataspace.selectHyperslab(H5S_SELECT_SET, count, start); 
  
  hsize_t *dim = new hsize_t[1];
  dim[0] = pos_length;
  
  DataSpace memspace(1, dim);

  if (FlagResizeOutput == 1)
    vOut.resize(pos_length);
  
  H5::DataType dType = dataset.getDataType();
  if (dType == H5::PredType::NATIVE_UINT) {
    if (pos_length > 0)
      dataset.read(&vOut.at(nOutputOffset), PredType::NATIVE_UINT, memspace, dataspace);
  }
  else {
    std::cout << "The dataset '" + sDataSetName + "' is not in UNSIGNED INT format." << std::endl;
    exit(666);
  }

#ifdef DEBUG
  function_switch("read_hdf5_data/UINT");
#endif

  delete[] count;
  delete[] start;
  delete[] dim;

  dataset.close();
  file.close();

  return;

}

// FLOAT version
void read_hdf5_data(std::string sFileName,             // HDF5 file name
		    std::string sDataSetName,          // Data set to read
		    std::vector<float> &vOut,          // [O] data
		    long pos_start,                 // first entry to read...
		    long pos_length,                // ... and how many in total
		    int nDim,                  // Dimension to be read
		    int FlagResizeOutput,
		    long nOutputOffset)       {


  using namespace std;
  using namespace H5;

  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);
  int rank = vDataSetDims.size();

  if (rank > 2) {
    std::cout << "Reading 3+-dimensional HDF5 data is not currently implemented. Sorry..." << std::endl;
    exit(333);
  }

  if (pos_length < 0)
    pos_length = vDataSetDims.front();
  
#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data/FLOAT");
  
  cout << rp() + "sFileName  = " << sFileName << endl;
  cout << rp() + "pos_start  = " << pos_start << endl;
  cout << rp() + "pos_length = " << pos_length << endl;
#endif  

  H5File file (sFileName, H5F_ACC_RDONLY);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace dataspace = dataset.getSpace();
  
  
  hsize_t *count = new hsize_t[rank];
  hsize_t *start = new hsize_t[rank]; 

  start[0] = pos_start;  
  count[0] = pos_length;

  if (rank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  dataspace.selectHyperslab(H5S_SELECT_SET, count, start); 
  
  hsize_t *dim = new hsize_t[1];
  dim[0] = pos_length;
  
  DataSpace memspace(1, dim);

  if (FlagResizeOutput == 1)
    vOut.resize(pos_length);
  
  H5::DataType dType = dataset.getDataType();
  if (dType == H5::PredType::NATIVE_FLOAT) {
    if (pos_length > 0)
      dataset.read(&vOut.at(nOutputOffset), PredType::NATIVE_FLOAT, memspace, dataspace);
  }
  else {
    std::cout << "The dataset '" + sDataSetName + "' is not in FLOAT format." << std::endl;
    exit(666);
  }

#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data/FLOAT");
#endif

  delete[] count;
  delete[] start;
  delete[] dim;

  dataset.close();
  file.close();

  return;

}


// DOUBLE version
void read_hdf5_data(std::string sFileName,         // HDF5 file name
		    std::string sDataSetName,      // Data set to read
		    std::vector<double> &vOut,     // [O] data
		    long pos_start,                // first entry to read...
		    long pos_length,               // ... and how many in total
		    int nDim,                      // Dimension to be read
		    int FlagResizeOutput,
		    long nOutputOffset) {       


  using namespace std;
  using namespace H5;

  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);
  int rank = vDataSetDims.size();



  if (rank > 2) {
    std::cout << "Reading 3+-dimensional HDF5 data is not currently implemented. Sorry..." << std::endl;
    exit(333);
  }

  if (pos_length < 0)
    pos_length = vDataSetDims.front();
  
#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data/DOUBLE");
  
  cout << rp() + "sFileName  = " << sFileName << endl;
  cout << rp() + "pos_start  = " << pos_start << endl;
  cout << rp() + "pos_length = " << pos_length << endl;
#endif  

  H5File file (sFileName, H5F_ACC_RDONLY);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace dataspace = dataset.getSpace();

  int tRank = dataspace.getSimpleExtentNdims();
  
  hsize_t *dims = new hsize_t[rank];
  int ndims = dataspace.getSimpleExtentDims(dims, NULL);
  
  
  H5::DataType dType = dataset.getDataType();
  
  hsize_t *count = new hsize_t[rank];
  hsize_t *start = new hsize_t[rank]; 

  start[0] = pos_start;  
  count[0] = pos_length;

  if (rank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  dataspace.selectNone();
  dataspace.selectHyperslab(H5S_SELECT_SET, count, start); 
  
  hsize_t *dim = new hsize_t[1];
  dim[0] = pos_length;
  
  DataSpace memspace(1, dim);

  if (FlagResizeOutput == 1)
    vOut.resize(pos_length);
  
  if (dType == H5::PredType::NATIVE_DOUBLE) {
    if (pos_length > 0)
      dataset.read(&vOut.at(nOutputOffset), PredType::NATIVE_DOUBLE, memspace, dataspace);
  }
  else {
    std::cout << "The dataset '" + sDataSetName + "' is not in DOUBLE format." << std::endl;
    exit(666);
  }

#ifdef VERBOSE_READ_HDF5_DATA_LONG
  function_switch("read_hdf5_data/DOUBLE");
#endif

  delete[] count;
  delete[] start;
  delete[] dim;

  dataset.close();
  file.close();

  return;

}

// INT version
void write_hdf5_data(std::string sFileName,
		     std::string sDataSetName,
		     const std::vector<int> &vData,
		     long pos_start,
		     int nDim,
		     int nVerb) {
  
  using namespace H5;
  
  if (nVerb == 1)
    std::cout << rp() + "Writing INT dataset '" + sDataSetName + "' to file '" << sFileName << "'... (nDim = " << nDim << ", length = " << vData.size() << ")" << std::endl;  
  
  // Open the file
  H5File file (sFileName, H5F_ACC_RDWR);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace fspace = dataset.getSpace();
  int nRank = fspace.getSimpleExtentNdims();  
  
  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);

  // Make simple dataspace describing layout ON FILE
  hsize_t *dims_ds = new hsize_t[nRank];
  dims_ds[0] = vDataSetDims.front();
  if (nRank == 2)
    dims_ds[1] = vDataSetDims.back();
  DataSpace tspace(nRank, dims_ds);

  // Make dataspace for MEMORY
  hsize_t *count_mem = new hsize_t[1];
  count_mem[0] = vData.size();
  DataSpace mspace(1, count_mem);

  // Select hyperslab in FILE space:
  hsize_t *count = new hsize_t[nRank];
  hsize_t *start = new hsize_t[nRank];

  start[0] = pos_start;
  count[0] = vData.size();

  if (nRank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  tspace.selectHyperslab(H5S_SELECT_SET, count, start);
  if (vData.size() > 0)
    dataset.write(&vData.front(), PredType::NATIVE_INT, mspace, tspace);

  delete[] count;
  delete[] start;
  delete[] count_mem;
  delete[] dims_ds;
  
  return;
}

// CHAR version
void write_hdf5_data(std::string sFileName,
		     std::string sDataSetName,
		     const std::vector<char> &vData,
		     long pos_start,
		     int nDim,
		     int nVerb) {
  
  using namespace H5;
  
  if (nVerb == 1)
    std::cout << rp() + "Writing CHAR dataset '" + sDataSetName + "' to file '" << sFileName << "'... (nDim = " << nDim << ", length = " << vData.size() << ")" << std::endl;  
  
  // Open the file
  H5File file (sFileName, H5F_ACC_RDWR);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace fspace = dataset.getSpace();
  int nRank = fspace.getSimpleExtentNdims();  
  
  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);

  // Make simple dataspace describing layout ON FILE
  hsize_t *dims_ds = new hsize_t[nRank];
  dims_ds[0] = vDataSetDims.front();
  if (nRank == 2)
    dims_ds[1] = vDataSetDims.back();
  DataSpace tspace(nRank, dims_ds);

  // Make dataspace for MEMORY
  hsize_t *count_mem = new hsize_t[1];
  count_mem[0] = vData.size();
  DataSpace mspace(1, count_mem);

  // Select hyperslab in FILE space:
  hsize_t *count = new hsize_t[nRank];
  hsize_t *start = new hsize_t[nRank];

  start[0] = pos_start;
  count[0] = vData.size();

  if (nRank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  tspace.selectHyperslab(H5S_SELECT_SET, count, start);
  if (vData.size() > 0)
    dataset.write(&vData.front(), PredType::NATIVE_CHAR, mspace, tspace);

  delete[] count;
  delete[] start;
  delete[] count_mem;
  delete[] dims_ds;
  
  return;
}

// UNSIGNED LONG version
void write_hdf5_data(std::string sFileName,
		     std::string sDataSetName,
		     const std::vector<unsigned long> &vData,
		     long pos_start,
		     int nDim,
		     int nVerb) {
  
  using namespace H5;

  if (nVerb == 1)
    std::cout << rp() + "Writing UNSIGNED LONG dataset '" + sDataSetName + "' to file '" << sFileName << "'... (nDim = " << nDim << ", length = " << vData.size() << ")" << std::endl;  
  
  // Open the file
  H5File file (sFileName, H5F_ACC_RDWR);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace fspace = dataset.getSpace();
  int nRank = fspace.getSimpleExtentNdims();  
  
  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);

  // Make simple dataspace describing layout ON FILE
  hsize_t *dims_ds = new hsize_t[nRank];
  dims_ds[0] = vDataSetDims.front();
  if (nRank == 2)
    dims_ds[1] = vDataSetDims.back();

  DataSpace tspace(nRank, dims_ds);

  // Make dataspace for MEMORY
  hsize_t *count_mem = new hsize_t[1];
  count_mem[0] = vData.size();
  DataSpace mspace(1, count_mem);

  // Select hyperslab in FILE space:
  hsize_t *count = new hsize_t[nRank];
  hsize_t *start = new hsize_t[nRank];

  start[0] = pos_start;
  count[0] = vData.size();

  if (nRank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  tspace.selectHyperslab(H5S_SELECT_SET, count, start);

  if (vData.size() > 0)
    dataset.write(&vData.front(), PredType::NATIVE_ULONG, mspace, tspace);

  delete[] count;
  delete[] start;
  delete[] count_mem;
  delete[] dims_ds;
  
  return;
}


// LONG version
void write_hdf5_data(std::string sFileName,
		     std::string sDataSetName,
		     const std::vector<long> &vData,
		     long pos_start,
		     int nDim,
		     int nVerb) {
  
  using namespace H5;

  if (nVerb == 1)
    std::cout << rp() + "Writing LONG dataset '" + sDataSetName + "' to file '" << sFileName << "'... (nDim = " << nDim << ", length = " << vData.size() << ")" << std::endl;  
  
  // Open the file
  H5File file (sFileName, H5F_ACC_RDWR);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace fspace = dataset.getSpace();
  int nRank = fspace.getSimpleExtentNdims();  
  
  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);

  // Make simple dataspace describing layout ON FILE
  hsize_t *dims_ds = new hsize_t[nRank];
  dims_ds[0] = vDataSetDims.front();
  if (nRank == 2)
    dims_ds[1] = vDataSetDims.back();
  DataSpace tspace(nRank, dims_ds);

  // Make dataspace for MEMORY
  hsize_t *count_mem = new hsize_t[1];
  count_mem[0] = vData.size();
  DataSpace mspace(1, count_mem);

  // Select hyperslab in FILE space:
  hsize_t *count = new hsize_t[nRank];
  hsize_t *start = new hsize_t[nRank];

  start[0] = pos_start;
  count[0] = vData.size();

  if (nRank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  tspace.selectHyperslab(H5S_SELECT_SET, count, start);

  if (vData.size() > 0)
    dataset.write(&vData.front(), PredType::NATIVE_LONG, mspace, tspace);

  delete[] count;
  delete[] start;
  delete[] count_mem;
  delete[] dims_ds;
  
  return;
}


// UNSIGNED INT version
void write_hdf5_data(std::string sFileName,
		     std::string sDataSetName,
		     const std::vector<unsigned int> &vData,
		     long pos_start,
		     int nDim,
		     int nVerb) {
  
  using namespace H5;
  
  if (nVerb)
    std::cout << rp() + "Writing UNSIGNED INT dataset '" + sDataSetName + "' to file '" << sFileName << "'... (nDim = " << nDim << ", length = " << vData.size() << ")" << std::endl;  
  
  // Open the file
  H5File file (sFileName, H5F_ACC_RDWR);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace fspace = dataset.getSpace();
  int nRank = fspace.getSimpleExtentNdims();  
  
  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);

  // Make simple dataspace describing layout ON FILE
  hsize_t *dims_ds = new hsize_t[nRank];
  dims_ds[0] = vDataSetDims.front();
  if (nRank == 2)
    dims_ds[1] = vDataSetDims.back();
  DataSpace tspace(nRank, dims_ds);

  // Make dataspace for MEMORY
  hsize_t *count_mem = new hsize_t[1];
  count_mem[0] = vData.size();
  DataSpace mspace(1, count_mem);

  // Select hyperslab in FILE space:
  hsize_t *count = new hsize_t[nRank];
  hsize_t *start = new hsize_t[nRank];

  start[0] = pos_start;
  count[0] = vData.size();

  if (nRank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  tspace.selectHyperslab(H5S_SELECT_SET, count, start);

  if (vData.size() > 0)
    dataset.write(&vData.front(), PredType::NATIVE_UINT, mspace, tspace);

  delete[] count;
  delete[] start;
  delete[] count_mem;
  delete[] dims_ds;
  
  return;
}


// FLOAT version
void write_hdf5_data(std::string sFileName,
		     std::string sDataSetName,
		     const std::vector<float> &vData,
		     long pos_start,
		     int nDim,
		     int nVerb) {
  
  using namespace H5;
  
  if (nVerb)
    std::cout << rp() + "Writing FLOAT dataset '" + sDataSetName + "' to file '" << sFileName << "'... (nDim = " << nDim << ", length = " << vData.size() << ")" << std::endl;  
  
  // Open the file
  H5File file (sFileName, H5F_ACC_RDWR);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace fspace = dataset.getSpace();
  int nRank = fspace.getSimpleExtentNdims();  
  
  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);

  // Make simple dataspace describing layout ON FILE
  hsize_t *dims_ds = new hsize_t[nRank];
  dims_ds[0] = vDataSetDims.front();
  if (nRank == 2)
    dims_ds[1] = vDataSetDims.back();
  DataSpace tspace(nRank, dims_ds);

  // Make dataspace for MEMORY
  hsize_t *count_mem = new hsize_t[1];
  count_mem[0] = vData.size();
  DataSpace mspace(1, count_mem);

  // Select hyperslab in FILE space:
  hsize_t *count = new hsize_t[nRank];
  hsize_t *start = new hsize_t[nRank];

  start[0] = pos_start;
  count[0] = vData.size();

  if (nRank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  tspace.selectHyperslab(H5S_SELECT_SET, count, start);

  if (vData.size() > 0)
    dataset.write(&vData.front(), PredType::NATIVE_FLOAT, mspace, tspace);

  delete[] count;
  delete[] start;
  delete[] count_mem;
  delete[] dims_ds;
  
  return;
}


// DOUBLE version
void write_hdf5_data(std::string sFileName,
		     std::string sDataSetName,
		     const std::vector<double> &vData,
		     long pos_start,
		     int nDim,
		     int nVerb) {
  
  using namespace H5;
  
  if (nVerb)
    std::cout << rp() + "Writing DOUBLE dataset '" + sDataSetName + "' to file '" << sFileName << "'... (nDim = " << nDim << ", length = " << vData.size() << ")" << std::endl;  
  
  // Open the file
  H5File file (sFileName, H5F_ACC_RDWR);

  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace fspace = dataset.getSpace();
  int nRank = fspace.getSimpleExtentNdims();  
  
  std::vector<long> vDataSetDims = hdf5_measure_dataset(sFileName, sDataSetName);

  // Make simple dataspace describing layout ON FILE
  hsize_t *dims_ds = new hsize_t[nRank];
  dims_ds[0] = vDataSetDims.front();
  if (nRank == 2)
    dims_ds[1] = vDataSetDims.back();
  DataSpace tspace(nRank, dims_ds);

  // Make dataspace for MEMORY
  hsize_t *count_mem = new hsize_t[1];
  count_mem[0] = vData.size();
  DataSpace mspace(1, count_mem);

  // Select hyperslab in FILE space:
  hsize_t *count = new hsize_t[nRank];
  hsize_t *start = new hsize_t[nRank];

  start[0] = pos_start;
  count[0] = vData.size();

  if (nRank == 2) {
    start[1] = nDim;
    count[1] = 1;
  }

  tspace.selectHyperslab(H5S_SELECT_SET, count, start);

  if (vData.size() > 0)
    dataset.write(&vData.front(), PredType::NATIVE_DOUBLE, mspace, tspace);

  delete[] count;
  delete[] start;
  delete[] count_mem;
  delete[] dims_ds;
  
  return;
}




std::vector<long> hdf5_measure_dataset(std::string sFileName, 
				       std::string sDataSetName) {

  using namespace H5;

#ifdef DEBUG
  std::cout << rp() + "Trying to measure dimensions of dataset '" << sDataSetName << "' in file '" << sFileName << "'..." << std::endl;
#endif

  H5File file (sFileName, H5F_ACC_RDONLY);
  
  DataSet dataset = file.openDataSet(sDataSetName);
  DataSpace dataspace = dataset.getSpace();
 
  int rank = dataspace.getSimpleExtentNdims();
  
  hsize_t *dims = new hsize_t[rank];
  int ndims = dataspace.getSimpleExtentDims(dims, NULL);

  if (rank != ndims) {
    std::cout << "Error encountered in hdf5_measure_dataset(). Terminating." << std::endl;
    exit(74); }

  std::vector<long> vVecDims(rank);  
  for (int ii = 0; ii < rank; ii++)
    vVecDims.at(ii) = static_cast<long> (dims[ii]);

#ifdef DEBUG
  std::cout << rp() + "...done!" << std::endl;
#endif

  return vVecDims;
}

void hdf5_create_file(std::string sFileName) {

  using namespace H5;
  H5File* file = new H5File(sFileName, H5F_ACC_TRUNC);
  delete file;
  
  return;
}


void hdf5_create_group(std::string sFileName,
		       std::string sGroupName) {

  using namespace H5;

  H5File file(sFileName, H5F_ACC_RDWR);
  H5::Group group = file.createGroup(sGroupName);
  group.close();
  file.close();
  
  return;
}



// Function to create a dataset in an *existing* HDF5 file
void hdf5_create_dataset(std::string sFileName,
			 std::string sDataSetName,
			 std::vector<int> vDims,
			 std::string sDataSetType,
			 std::string sDescription,
			 int nFillValue) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  
  
  int nDims = vDims.size();
  if (nDims == 0) {
    std::cout << rp() + "Empty dimension vector -- dataset '" << sDataSetName << "' not created." << std::endl;
    file.close();
    return;
  }

  hsize_t *fdim = new hsize_t[nDims];
  for (int ii = 0; ii < nDims; ii++)
    fdim[ii] = vDims.at(ii);

  DataSpace fspace(nDims, fdim);

  DataSet dataset;

  if (sDataSetType == "int") {
    // Default values:
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_INT, &nFillValue);
    dataset = file.createDataSet(sDataSetName, PredType::NATIVE_INT, fspace, plist);
  }

  if (sDataSetType == "unsigned int") {
    // Default values:
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_UINT, &nFillValue);
    dataset = file.createDataSet(sDataSetName, PredType::NATIVE_UINT, fspace, plist);
  }

  if (sDataSetType == "char") {
    // Default values:
    DSetCreatPropList plist;
    char cFillVal = static_cast<char> (nFillValue);
    plist.setFillValue(PredType::NATIVE_CHAR, &cFillVal);
    dataset = file.createDataSet(sDataSetName, PredType::NATIVE_CHAR, fspace, plist);
  }

  if (sDataSetType == "unsigned long") {
    // Default values:
    DSetCreatPropList plist;
    unsigned long ulFillVal = static_cast<unsigned long> (nFillValue);
    plist.setFillValue(PredType::NATIVE_ULONG, &ulFillVal);
    dataset = file.createDataSet(sDataSetName, PredType::NATIVE_ULONG, fspace, plist);
  }

  if (sDataSetType == "long") {
    // Default values:
    DSetCreatPropList plist;
    long lFillVal = static_cast<long> (nFillValue);
    plist.setFillValue(PredType::NATIVE_LONG, &lFillVal);
    dataset = file.createDataSet(sDataSetName, PredType::NATIVE_LONG, fspace, plist);
  }

  if (sDataSetType == "float") {
    // Default values:
    DSetCreatPropList plist;
    float fFillVal = static_cast<float> (nFillValue);
    plist.setFillValue(PredType::NATIVE_FLOAT, &fFillVal);
    dataset = file.createDataSet(sDataSetName, PredType::NATIVE_FLOAT, fspace, plist);
  }

  if (sDataSetType == "double") {
    // Default values:
    DSetCreatPropList plist;
    double dFillVal = static_cast<double> (nFillValue);
    plist.setFillValue(PredType::NATIVE_DOUBLE, &dFillVal);
    dataset = file.createDataSet(sDataSetName, PredType::NATIVE_DOUBLE, fspace, plist);
  }



  // Write explanation as attribute
  if (!sDescription.empty()) {
    StrType str_type(PredType::C_S1, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);

    Attribute att = dataset.createAttribute("Explanation", str_type, att_space);
    std::string Text = sDescription;
    att.write(str_type, &Text);
  }

  dataset.close();
  file.close();
  
  delete[] fdim;

  return;
}

// Double version
void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  double dAttribVal,
			  char nAttribType) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  

  // Create the data space for the attribute.
  hsize_t dims[1] = {1};
  DataSpace attr_dataspace = DataSpace (1, dims);
  
  H5::Attribute attribute;

  if (nAttribType == 'd') {
    DataSet dataset = file.openDataSet(sContName);
    attribute = dataset.createAttribute(sAttribName, PredType::NATIVE_DOUBLE, attr_dataspace);
  }
  else {
    Group group = file.openGroup(sContName);
    attribute = group.createAttribute(sAttribName, PredType::NATIVE_DOUBLE, attr_dataspace);
  }

  // Write the attribute data 
  attribute.write( PredType::NATIVE_DOUBLE, &dAttribVal);

  attribute.close();
  file.close();
  return;
}

// LONG version
void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  long lAttribVal,
			  char nAttribType) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  

  // Create the data space for the attribute.
  hsize_t dims[1] = {1};
  DataSpace attr_dataspace = DataSpace (1, dims);
  
  H5::Attribute attribute;

  if (nAttribType == 'd') {
    DataSet dataset = file.openDataSet(sContName);
    attribute = dataset.createAttribute(sAttribName, PredType::NATIVE_LONG, attr_dataspace);
  }
  else {
    Group group = file.openGroup(sContName);
    attribute = group.createAttribute(sAttribName, PredType::NATIVE_LONG, attr_dataspace);
  }

  // Write the attribute data 
  attribute.write( PredType::NATIVE_LONG, &lAttribVal);

  attribute.close();
  file.close();
  return;
}

// INT version
void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  int nAttribVal,
			  char cAttribType) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  

  // Create the data space for the attribute.
  hsize_t dims[1] = {1};
  DataSpace attr_dataspace = DataSpace (1, dims);
  
  H5::Attribute attribute;

  if (cAttribType == 'd') {
    DataSet dataset = file.openDataSet(sContName);
    attribute = dataset.createAttribute(sAttribName, PredType::NATIVE_INT, attr_dataspace);
  }
  else {
    Group group = file.openGroup(sContName);
    attribute = group.createAttribute(sAttribName, PredType::NATIVE_INT, attr_dataspace);
  }

  // Write the attribute data 
  attribute.write( PredType::NATIVE_INT, &nAttribVal);

  attribute.close();
  file.close();
  return;
}

// STRING version
void hdf5_write_attribute(std::string sFileName,
			  std::string sContName,
			  std::string sAttribName,
			  std::string sAttribVal,
			  char cAttribType) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  

  // Create the data space for the attribute.
  hsize_t dims[1] = {1};
  DataSpace attr_dataspace = DataSpace (1, dims);
  
  H5::Attribute attribute;
  StrType str_type(PredType::C_S1, H5T_VARIABLE);
  DataSpace att_space(H5S_SCALAR);
  
  if (cAttribType == 'd') {
    DataSet dataset = file.openDataSet(sContName);
    attribute = dataset.createAttribute(sAttribName, str_type, att_space);
  }
  else {
    Group group = file.openGroup(sContName);
    attribute = group.createAttribute(sAttribName, str_type, att_space);
  }

  attribute.write(str_type, &sAttribVal);
  

  attribute.close();
  file.close();
  return;
}


// Array-LONG version
void hdf5_write_attribute_array(std::string sFileName,
				std::string sContName,
				std::string sAttribName,
				std::vector<long> lAttribVals,
				char nAttribType) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  

  // Create the data space for the attribute.
  hsize_t dims[1] = {lAttribVals.size()};
  DataSpace attr_dataspace = DataSpace (1, dims);
  
  H5::Attribute attribute;

  if (nAttribType == 'd') {
    DataSet dataset = file.openDataSet(sContName);
    attribute = dataset.createAttribute(sAttribName, PredType::NATIVE_LONG, attr_dataspace);
  }
  else {
    Group group = file.openGroup(sContName);
    attribute = group.createAttribute(sAttribName, PredType::NATIVE_LONG, attr_dataspace);
  }

  // Write the attribute data 
  attribute.write( PredType::NATIVE_LONG, &lAttribVals.front());

  attribute.close();
  file.close();
  return;
}

// Array-INT version
void hdf5_write_attribute_array(std::string sFileName,
				std::string sContName,
				std::string sAttribName,
				std::vector<int> nAttribVals,
				char nAttribType) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  

  // Create the data space for the attribute.
  hsize_t dims[1] = {nAttribVals.size()};
  DataSpace attr_dataspace = DataSpace (1, dims);
  
  H5::Attribute attribute;

  if (nAttribType == 'd') {
    DataSet dataset = file.openDataSet(sContName);
    attribute = dataset.createAttribute(sAttribName, PredType::NATIVE_INT, attr_dataspace);
  }
  else {
    Group group = file.openGroup(sContName);
    attribute = group.createAttribute(sAttribName, PredType::NATIVE_INT, attr_dataspace);
  }

  // Write the attribute data 
  attribute.write( PredType::NATIVE_INT, &nAttribVals.front());

  attribute.close();
  file.close();
  return;
}

// Array-DOUBLE version
void hdf5_write_attribute_array(std::string sFileName,
				std::string sContName,
				std::string sAttribName,
				std::vector<double> lAttribVals,
				char nAttribType) {

  using namespace H5;

  // Open the file
  H5File file(sFileName, H5F_ACC_RDWR);  

  // Create the data space for the attribute.
  hsize_t dims[1] = {lAttribVals.size()};
  DataSpace attr_dataspace = DataSpace (1, dims);
  
  H5::Attribute attribute;

  if (nAttribType == 'd') {
    DataSet dataset = file.openDataSet(sContName);
    attribute = dataset.createAttribute(sAttribName, PredType::NATIVE_DOUBLE, attr_dataspace);
  }
  else {
    Group group = file.openGroup(sContName);
    attribute = group.createAttribute(sAttribName, PredType::NATIVE_DOUBLE, attr_dataspace);
  }

  // Write the attribute data 
  attribute.write( PredType::NATIVE_DOUBLE, &lAttribVals.front());

  attribute.close();
  file.close();
  return;
}
