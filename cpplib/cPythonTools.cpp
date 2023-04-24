#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>

extern "C"
{
  int idlhist(int nArgs, void *argv[]);
  int fl_idlhist(int nArgs, void *argc[]);
  int cFindSegments(int nArgs, void *argc[]);
}



int idlhist(int nArgs,
	    void *argv[])
{

  long* invec = (long *) argv[0];
  long* histogram = (long *) argv[1];
  long* offset = (long *) argv[2];
  long* sortinds = (long *) argv[3];
  
  long nMin = *(long *) argv[4];
  long nMax = *(long *) argv[5];
  long nNElem = *(long *) argv[7];
    
  std::cout << "Performing IDLhist with min=" << nMin <<", max=" << nMax << ", nbins=" << nMax-nMin+1 << std::endl; 

  // Pass 1: Construct histogram
  for (int ii = 0; ii < nNElem; ii++) {
    long currVal = invec[ii];
    if (currVal < nMin || currVal > nMax)
	continue;

    histogram[currVal-nMin] += 1;
  }    
    
  // Pass 2: Make offset list
  int nHistoLength = nMax-nMin+1;
  for (int jj = 0; jj < nHistoLength; jj++)
    offset[jj+1] = offset[jj] + histogram[jj];


  // Pass 3: Make revind list
  std::vector<long> vCurrOffsetInBin(nMax-nMin+1, 0); 

  for (int ii = 0; ii < nNElem; ii++) {
    int nIndBin = invec[ii]-nMin;
    if (nIndBin < 0)
      continue;
    
    int nLoc = offset[nIndBin] + vCurrOffsetInBin.at(nIndBin);
    sortinds[nLoc] = ii;
    vCurrOffsetInBin.at(nIndBin) += 1;
  }


  return 0;

}


int fl_idlhist(int nArgs,
	    void *argv[])
{
  double* invec = (double *) argv[0];
  long* histogram = (long *) argv[1];
  long* offset = (long *) argv[2];
  long* sortinds = (long *) argv[3];
  
  double dMin = *(double *) argv[4];
  long nNElem = *(long *) argv[5];
  double dBinsize = *(double *) argv[6];
  long nNBins = *(long *) argv[7];
  
  double dMax = dMin + dBinsize*nNBins;

  std::cout << "dMin=" << dMin << ", nNElem=" << nNElem << ", dBinsize=" << dBinsize << ", nNBins=" << nNBins << std::endl;

  // Pass 1: Construct histogram
  for (long ii = 0; ii < nNElem; ii++) {
    if (invec[ii] < dMin || invec[ii] > dMax)
	continue;

    int nIndBin = static_cast<long>(floor((invec[ii]-dMin)/dBinsize));

    if (invec[ii] == dMax)
      nIndBin = nNBins-1;
    
    histogram[nIndBin] += 1;
    
  }    
    
  // Pass 2: Make offset list
  for (int jj = 0; jj < nNBins; jj++)
    offset[jj+1] = offset[jj] + histogram[jj];


  // Pass 3: Make revind list
  std::vector<long> vCurrOffsetInBin(nNBins, 0); 

  for (long ii = 0; ii < nNElem; ii++) {
    int nIndBin = static_cast<long>(floor((invec[ii]-dMin)/dBinsize));

    if (invec[ii] == dMax)
      nIndBin = nNBins-1;
    
    //std::cout << "ii=" << ii << ", nIndBin=" << nIndBin << std::endl;

    if (nIndBin < 0 || nIndBin >= nNBins)
      continue;
    
    int nLoc = offset[nIndBin] + vCurrOffsetInBin.at(nIndBin);
    sortinds[nLoc] = ii;
    vCurrOffsetInBin.at(nIndBin) += 1;
  }


  return 0;

}


long cFindSegments(int nargs,
		  void *argv[])
{
  long* cell_offsets = (long*) argv[0];
  long* cell_lengths = (long*) argv[1];
  long* numCellsPerDim = (long*) argv[2];
  long* cell_count = (long*) argv[3];
  long* cell_offset = (long*) argv[4];
  long* file_offset = (long*) argv[5];

  long* files = (long*) argv[6];
  long* offsets = (long*) argv[7];
  long* lengths = (long*) argv[8];

  int flag_periodic = *(int*) argv[9];
  long nCells = *(long*) argv[10];

  long nSegments = 0;
    
  long countCheck = 0, fullCheck = 0;
  
  long cx, cy, cz;
  long cxx, cyy, czz;

  long index, currOffsetInFile;
  long coi;
  int file;

  for (cz = cell_offsets[2]; cz < cell_offsets[2]+cell_lengths[2]; cz++) 
    for (cy = cell_offsets[1]; cy < cell_offsets[1]+cell_lengths[1]; cy++) 
      for (cx = cell_offsets[0]; cx < cell_offsets[0]+cell_lengths[0]; cx++) 
	{
	  cxx = cx;   /* Need to make copies so we don't interfere with the */
	  cyy = cy;   /* loop variables */
	  czz = cz;

	  if (flag_periodic) 
	    {
	      if (cxx < 0)
		cxx += numCellsPerDim[0];
	      else if (cxx >= numCellsPerDim[0])
		cxx -= numCellsPerDim[0];
		
	      if (cyy < 0)
		cyy += numCellsPerDim[1];
	      else if (cyy >= numCellsPerDim[1])
		cyy -= numCellsPerDim[1];

	      if (czz < 0)
		czz += numCellsPerDim[2];
	      else if (czz >= numCellsPerDim[2])
		czz -= numCellsPerDim[2];
	    }
	  
	  index = cxx + cyy*numCellsPerDim[0] + czz*numCellsPerDim[0]*numCellsPerDim[1];
	  CountCheck++;
	  
	  assert(index >= 0);
	  assert(index < nCells);
	  
	  if (!cell_count[index])
	    continue;

	  fullCheck++;
	  firstElem = cell_offset[index];
	  lastElem = firstElem+cell_count[index]-1;
            
	  file = bsearch(firstElem, file_offset, nfiles+1, sizeof(long), cmpfunc);
	  currOffsetInFile = firstElem - file_offset[file];

	  
	    
					   
	    
	  
	}

  return nSegments;
}

int cmpfunc(const void * a, const void * b)
{
  return ( *(long*)a - *(long*)b );
}




