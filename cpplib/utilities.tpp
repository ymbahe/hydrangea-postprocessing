// utilities.tpp
// Template implementations for utilities
// Started 19 OCT 2014

#include <cstdlib>
#include <sstream>

#include "utilities_mpi.h"


template <typename T>
T min(std::vector<T> v, T thresh, int &success)

{
  T minval = v[0];	
  success = 0;		
  for (long long jj=0;jj<v.size();jj++) {
  
  if (v[jj] >= thresh) {
      if (success == 0) {
       	  success = 1;
	  minval = v[jj]; }	  
    if (v[jj] < minval)
      minval=v[jj];
}
 }
  return minval;
}

template <typename T>
T min(std::vector<T> v)
{
  T minval = v[0];
  for (long long jj=0;jj<v.size();jj++) {
    if (v[jj] < minval)
      minval=v[jj];
  }
  return minval;
}

template <typename T>
T min(std::vector<T> v, std::vector<int> vMask)
{
  T minval;
  int nCheck = 0;

  for (long long jj=0;jj<v.size();jj++) {
    if (vMask.at(jj) == 0)
      continue;
    
    if (nCheck == 0) {
      minval = v.at(jj);
      nCheck = 1; }

    else if (v[jj] < minval)
      minval=v[jj];
  }

  if (nCheck == 0) {
    std::cout << "No single valid entry in vector - cannot determine minimum." << std::endl;
    exit(111);
  }

  return minval;
}
 
template <typename T>
T max(std::vector<T> v, std::vector<int> vMask)
{
  T maxval;
  int nCheck = 0;

  for (long long jj=0;jj<v.size();jj++) {
    if (vMask.at(jj) == 0)
      continue;
    
    if (nCheck == 0) {
      maxval = v.at(jj);
      nCheck = 1; }

    else if (v[jj] > maxval)
      maxval=v[jj];
  }

  if (nCheck == 0) {
    std::cout << "No single valid entry in vector - cannot determine maximum." << std::endl;
    exit(111);
  }

  return maxval;
}



template <typename T>
T max(std::vector<T> v)
{
  T maxval = v[0];
  for (long long jj=0;jj<v.size();jj++) {
    if (v[jj] > maxval)
      maxval=v[jj];
  }
  return maxval;
}

template <typename T> 
std::vector<T> minmax(std::vector<T> v, T thresh, int &success) 
{
  T minval = -99;
  T maxval = -99;
  success = 0;

  for (long long jj=0;jj<v.size();jj++) {
    if (v[jj] >= thresh) {

       if (success == 0) {
       	  success = 1;
	  minval = v[jj]; 	  
	  maxval = v[jj];}

      if (v[jj] < minval)
        minval=v[jj];
      if (v[jj] > maxval)
        maxval=v[jj];
    }
  }
  
  std::vector<T> outvec(2,-10);
  outvec.at(0)=minval;
  outvec.at(1)=maxval;
  return outvec;
}

template <typename T> 
std::vector<T> minmax(std::vector<T> v) 
{
  T minval = v.at(0);
  T maxval = v.at(0);
  
  for (size_t jj=0;jj<v.size();jj++) {

      if (v[jj] < minval)
        minval=v[jj];
      if (v[jj] > maxval)
        maxval=v[jj];
    
  }
  
  std::vector<T> outvec(2);
  outvec.at(0)=minval;
  outvec.at(1)=maxval;
  return outvec;
}


/* 
template<typename T>
std::string to_string(T var) {
  std::stringstream ss;
  ss << var;
  std::string str = ss.str();
  return str;
}	    
*/

template<typename T>
std::string to_string(T var, int nPad, char cPad) {
  std::stringstream ss;
  ss << var;
  std::string str = ss.str();

  if (nPad > 0 && str.size() < nPad)
    for (int ii = 0; ii < str.size() - nPad; ii++)
      str = '0' + str;
  return str;
}	    


// sort_indices (IDL-like sorting function):
template <typename T>
std::vector<long> sort_indices(const std::vector<T> v, int verbose) 
{
  
  double dummy = ElTime();	
			
  // Initialise		original index locations:
  std::vector<long> idx(v.size());
  for (size_t i=0; i != idx.size(); i++) 
    idx[i] = i;
  
  // Now do the actual sorting...
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  
  if (verbose == 1) {
    report<double>(ElTime(), "Sorting took [sec.]");      
  }

  return idx;
}


// sort_indices_descending (IDL-like sorting function):
template <typename T>
std::vector<long> sort_indices_descending(const std::vector<T> v, int verbose) 
{
  double dummy = ElTime();	

			
  // Initialise		original index locations:
  std::vector<long> idx(v.size());
  for (size_t i=0; i != idx.size(); i++) 
    idx[i] = i;
  
  // Now do the actual sorting...
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  if (verbose == 1) {
    report<double>(ElTime(), "Sorting took [sec.]");      
  }
  
  return idx;
}


// print_vector:
template <typename T>
void print_vector(const std::vector<T> &v, std::string prefix, size_t nBeg, size_t nEnd) 
{


  if (nBeg < 0)
    nBeg = 0;

  if (nEnd > (v.size()-1) || nEnd == 0)
    nEnd = v.size()-1;
  
  if (v.size() == 0) {

    std::cout << rp() + prefix + " = [[EMPTY]]" << std::endl;

    return;
  }
  
  if (nEnd == -1)
    nEnd = v.size()-1;
   std::cout << rp() + prefix + " = ";
   for (int ii=nBeg;ii<=nEnd;ii++)
     std::cout << v.at(ii) << " ";
   std::cout << std::endl; 

   return;
}



template <typename T>
T string_to_num(std::string sString) {

  T nResult;

  std::istringstream ss(sString);
  if (!(ss >> nResult))
    std::cerr << "String could not be converted: " << sString << '\n'; 
    	      
  return nResult;

}

template <typename T>
std::vector<T> split_list(T length,
			  int parts,
			  int nVerb) {

#ifdef DEBUG
  nVerb = 1;
#endif


  // Initialize the output vector:
  std::vector<T> vSplitOffset(parts+1, 0);


  // Calculate the 'base' number of elements in each sub-task:
  T n_base = length / parts;
  T n_rem = length % parts;


  for (int ii = 0; ii < parts; ii++) {
    T nCurr = n_base;
    if (ii < n_rem)
      nCurr++;

    vSplitOffset.at(ii+1) = vSplitOffset.at(ii) + nCurr;
  }

#ifdef VERBOSE
  print_vector<T>(vSplitOffset, "vSplitOffset");
#endif 



  return vSplitOffset;
}

template <typename T>
void chunk_split_list(const std::vector<T> &vFullDataOffset,
		      int parts,
		      std::vector<T> &vTDOlist,
		      std::vector<T> &vTCOlist,
		      int nVerb) {
  
  using namespace std;

  long nSpan = vFullDataOffset.size()-1;
  long nDataTot = vFullDataOffset.back()-vFullDataOffset.front();

#ifdef VERBOSE
  cout << rp() + "nHaloSpan = " << nSpan << endl;
  cout << rp() + "vFullPartOffset[0]   = " << vFullPartOffset.front() << endl;
#endif 
  
  long nDesDataCore = static_cast<long>(static_cast<double>(nDataTot) / parts);
  long nCurrDesDataCore = nDesDataCore;

#ifdef VERBOSE
#ifdef INDENT_OUTPUT
  cout << rp() + "There are " << nDataTot << " particles in total." << endl;
  cout << rp() + "Aiming for " << nDesDataCore << " particles per core" << endl;
#else
  cout << "There are " << nDataTot << " particles in total." << endl;
  cout << "Aiming for " << nDesDataCore << " particles per core" << endl;  
#endif
#endif

  long currcounter = 0;
  int currtask = 0;

  vTDOlist.resize(parts+1, 0);
  vTCOlist.resize(parts+1, 0);
  
  // Initialise the FIRST element (easy):
  vTDOlist.front() = vFullDataOffset.front();
  
  // Now we need to loop through the chunk list...
  for (long iichunk = 0; iichunk < nSpan; iichunk++)
    {
      long Length = vFullDataOffset.at(iichunk+1)-vFullDataOffset.at(iichunk);
      currcounter += Length;
	
      if (currcounter >= nCurrDesDataCore and currtask < (parts-1))
	{
	  vTDOlist.at(currtask+1) = vFullDataOffset.at(iichunk+1);
	  vTCOlist.at(currtask+1) = iichunk+1;
	  
	  currcounter = 0;
	  currtask++;
	  
	  nCurrDesDataCore = static_cast<long>(static_cast<double>(vFullDataOffset.back()-vFullDataOffset.at(iichunk+1))/(parts-currtask));
	  
	}
    }
  
  // New bit added 27-APR-2016, to cope with nSpan < parts:
  // (N.B.: This includes writing the coda)

  for (int iifill = currtask; iifill < parts; iifill++) {
    vTDOlist.at(iifill+1) = vFullDataOffset.back();
    vTCOlist.at(iifill+1) = nSpan;
  }

  // Done!

  return;

}
  

template <typename T1, typename T2>
std::vector<T1> idlhist(const std::vector<T2> &vInput,    // Input vector
			std::vector<T1> &vOffset,         // [O] Offset list
			std::vector<T1> &vRevInds,        // [O] Reverse indices
			int &nMin,                        // [O] Minimum
			int &nMax,                        // [O] Maximum
			int FlagDetMinMax,                // if 0, take given min/max
			int nVerb) {                  // Switch for verbosity


#ifdef DEBUG
  nVerb = 1;
#endif

  vOffset.clear();
  vRevInds.clear();
  int nVecSize = vInput.size();  

  // Pass 1: Find min/max
  // NB: If FlagDetMinMax is 0, then we just take the input nMin and nMax.
  if (FlagDetMinMax == 1) {
    std::vector<T2> vMinMax = minmax<T2>(vInput);
    nMin = vMinMax.at(0);
    nMax = vMinMax.at(1);
  }
  
  int nHistoLength = nMax-nMin+1;
  vOffset.resize(nHistoLength+1, 0);

  // Pass 2: Construct histogram
  std::vector<T1> vHistogram(nMax-nMin+1,0);

  for (int ii = 0; ii < nVecSize; ii++) {
    if (vInput.at(ii) < nMin)
      continue;
    
    vHistogram.at(vInput.at(ii)-nMin) += 1;
  }

  if (nVecSize == 0) {
    return vHistogram;
  }
    
  // Pass 3: Make offset list
  for (int jj = 0; jj < nHistoLength; jj++)
    vOffset.at(jj+1) = vOffset.at(jj) + vHistogram.at(jj);


  // Pass 4: Make revind list
  vRevInds.resize(nVecSize, -1);
  std::vector<T1> vCurrOffsetInBin(nMax-nMin+1, 0); 

  for (int ii = 0; ii < nVecSize; ii++) {
    int nIndBin = vInput.at(ii)-nMin;
    if (nIndBin < 0)
      continue;
    
    int nLoc = vOffset.at(nIndBin) + vCurrOffsetInBin.at(nIndBin);
    vRevInds.at(nLoc) = ii;
    vCurrOffsetInBin.at(nIndBin) += 1;
  }


  return vHistogram;

}


// This function creates an offset list from an input vector
template <typename T>
std::vector<T> make_offset(const std::vector<T> &vInput,
			   T nMinVal,
			   T nMaxVal,
			   int nVerb) {
  
#ifdef DEBUG
  nVerb = 1;
#endif


  std::vector<T> vOffsets(nMaxVal-nMinVal+2,0);

  T nCurrVal = nMinVal;
  
  for (size_t ii = 0; ii < vInput.size(); ii++) {

    T val_ii = vInput.at(ii);
    if (val_ii > nCurrVal) {
      do {
	vOffsets.at(nCurrVal-nMinVal+1) = ii;
	nCurrVal++;
      } while (val_ii > nCurrVal);

    }
  } // ends for-loop

  // Coda:
  for (T jj = nCurrVal; jj <= nMaxVal; jj++)
    vOffsets.at(jj-nMinVal+1) = vInput.size();


  return vOffsets;

}

#ifdef PEANOKEYS
// -------- Routine to compute peano-hilbert values
// -------- Taken from Gadget-3

/*! This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
  *  with x,y,z in the range between 0 and 2^bits-1.
  */
peanokey peano_hilbert_key(int x, int y, int z, int bits)
{

  /*  The following rewrite of the original function
   *  peano_hilbert_key_old() has been written by MARTIN REINECKE. 
   *  It is about a factor 2.3 - 2.5 faster than Volker's old routine!
   */
  
  const unsigned char rottable3[48][8] = {
    {36, 28, 25, 27, 10, 10, 25, 27},
    {29, 11, 24, 24, 37, 11, 26, 26},
    {8, 8, 25, 27, 30, 38, 25, 27},
    {9, 39, 24, 24, 9, 31, 26, 26},
    {40, 24, 44, 32, 40, 6, 44, 6},
    {25, 7, 33, 7, 41, 41, 45, 45},
    {4, 42, 4, 46, 26, 42, 34, 46},
    {43, 43, 47, 47, 5, 27, 5, 35},
    {33, 35, 36, 28, 33, 35, 2, 2},
    {32, 32, 29, 3, 34, 34, 37, 3},
    {33, 35, 0, 0, 33, 35, 30, 38},
    {32, 32, 1, 39, 34, 34, 1, 31},
    {24, 42, 32, 46, 14, 42, 14, 46},
    {43, 43, 47, 47, 25, 15, 33, 15},
    {40, 12, 44, 12, 40, 26, 44, 34},
    {13, 27, 13, 35, 41, 41, 45, 45},
    {28, 41, 28, 22, 38, 43, 38, 22},
    {42, 40, 23, 23, 29, 39, 29, 39},
    {41, 36, 20, 36, 43, 30, 20, 30},
    {37, 31, 37, 31, 42, 40, 21, 21},
    {28, 18, 28, 45, 38, 18, 38, 47},
    {19, 19, 46, 44, 29, 39, 29, 39},
    {16, 36, 45, 36, 16, 30, 47, 30},
    {37, 31, 37, 31, 17, 17, 46, 44},
    {12, 4, 1, 3, 34, 34, 1, 3},
    {5, 35, 0, 0, 13, 35, 2, 2},
    {32, 32, 1, 3, 6, 14, 1, 3},
    {33, 15, 0, 0, 33, 7, 2, 2},
    {16, 0, 20, 8, 16, 30, 20, 30},
    {1, 31, 9, 31, 17, 17, 21, 21},
    {28, 18, 28, 22, 2, 18, 10, 22},
    {19, 19, 23, 23, 29, 3, 29, 11},
    {9, 11, 12, 4, 9, 11, 26, 26},
    {8, 8, 5, 27, 10, 10, 13, 27},
    {9, 11, 24, 24, 9, 11, 6, 14},
    {8, 8, 25, 15, 10, 10, 25, 7},
    {0, 18, 8, 22, 38, 18, 38, 22},
    {19, 19, 23, 23, 1, 39, 9, 39},
    {16, 36, 20, 36, 16, 2, 20, 10},
    {37, 3, 37, 11, 17, 17, 21, 21},
    {4, 17, 4, 46, 14, 19, 14, 46},
    {18, 16, 47, 47, 5, 15, 5, 15},
    {17, 12, 44, 12, 19, 6, 44, 6},
    {13, 7, 13, 7, 18, 16, 45, 45},
    {4, 42, 4, 21, 14, 42, 14, 23},
    {43, 43, 22, 20, 5, 15, 5, 15},
    {40, 12, 21, 12, 40, 6, 23, 6},
    {13, 7, 13, 7, 41, 41, 22, 20}
  };
  
  const unsigned char subpix3[48][8] = {
    {0, 7, 1, 6, 3, 4, 2, 5},
    {7, 4, 6, 5, 0, 3, 1, 2},
    {4, 3, 5, 2, 7, 0, 6, 1},
    {3, 0, 2, 1, 4, 7, 5, 6},
    {1, 0, 6, 7, 2, 3, 5, 4},
    {0, 3, 7, 4, 1, 2, 6, 5},
    {3, 2, 4, 5, 0, 1, 7, 6},
    {2, 1, 5, 6, 3, 0, 4, 7},
    {6, 1, 7, 0, 5, 2, 4, 3},
    {1, 2, 0, 3, 6, 5, 7, 4},
    {2, 5, 3, 4, 1, 6, 0, 7},
    {5, 6, 4, 7, 2, 1, 3, 0},
    {7, 6, 0, 1, 4, 5, 3, 2},
    {6, 5, 1, 2, 7, 4, 0, 3},
    {5, 4, 2, 3, 6, 7, 1, 0},
    {4, 7, 3, 0, 5, 6, 2, 1},
    {6, 7, 5, 4, 1, 0, 2, 3},
    {7, 0, 4, 3, 6, 1, 5, 2},
    {0, 1, 3, 2, 7, 6, 4, 5},
    {1, 6, 2, 5, 0, 7, 3, 4},
    {2, 3, 1, 0, 5, 4, 6, 7},
    {3, 4, 0, 7, 2, 5, 1, 6},
    {4, 5, 7, 6, 3, 2, 0, 1},
    {5, 2, 6, 1, 4, 3, 7, 0},
    {7, 0, 6, 1, 4, 3, 5, 2},
    {0, 3, 1, 2, 7, 4, 6, 5},
    {3, 4, 2, 5, 0, 7, 1, 6},
    {4, 7, 5, 6, 3, 0, 2, 1},
    {6, 7, 1, 0, 5, 4, 2, 3},
    {7, 4, 0, 3, 6, 5, 1, 2},
    {4, 5, 3, 2, 7, 6, 0, 1},
    {5, 6, 2, 1, 4, 7, 3, 0},
    {1, 6, 0, 7, 2, 5, 3, 4},
    {6, 5, 7, 4, 1, 2, 0, 3},
    {5, 2, 4, 3, 6, 1, 7, 0},
    {2, 1, 3, 0, 5, 6, 4, 7},
    {0, 1, 7, 6, 3, 2, 4, 5},
    {1, 2, 6, 5, 0, 3, 7, 4},
    {2, 3, 5, 4, 1, 0, 6, 7},
    {3, 0, 4, 7, 2, 1, 5, 6},
    {1, 0, 2, 3, 6, 7, 5, 4},
    {0, 7, 3, 4, 1, 6, 2, 5},
    {7, 6, 4, 5, 0, 1, 3, 2},
    {6, 1, 5, 2, 7, 0, 4, 3},
    {5, 4, 6, 7, 2, 3, 1, 0},
    {4, 3, 7, 0, 5, 2, 6, 1},
    {3, 2, 0, 1, 4, 5, 7, 6},
    {2, 5, 1, 6, 3, 4, 0, 7}
  };
  
  int mask;
  unsigned char rotation = 0;
  peanokey key = 0;
  
  for(mask = 1 << (bits - 1); mask > 0; mask >>= 1)
    {
      unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

      key <<= 3;
      key |= subpix3[rotation][pix];
      rotation = rottable3[rotation][pix];
    }

  return key;
}
#endif
