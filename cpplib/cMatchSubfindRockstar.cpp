#include <iostream>
#include <vector>
#include <math.h>
#include <sys/time.h>
#include <algorithm>

extern "C"
{
  int match_subfind_rockstar(int nArgs, void* argv[]);
}


template <typename T1, typename T2>
std::vector<T1> idlhist(const std::vector<T2> &vInput,    // Input vector
			std::vector<T1> &vOffset,         // [O] Offset list
			std::vector<T1> &vRevInds,        // [O] Reverse indices
			int &nMin,                        // [O] Minimum
			int &nMax,                        // [O] Maximum
			int FlagDetMinMax,                // if 0, take given min/max
			int nVerb);                       // Switch for verbosity


std::vector<int> index_to_halo(std::vector<long> &vIndex,              // Index list (particles)
			       const std::vector<long> &vOffset,       // HPOlist
			       const std::vector<long> &vLength,       // Length of each halo
			       std::vector<long> &vOrder     );         // Order in remote halo

template <typename T> 
std::vector<T> minmax(std::vector<T> v);

template <typename T>
T max(std::vector<T> v);

template <typename T>
std::vector<long> sort_indices(const std::vector<T> v, int verbose = 0); 



int match_subfind_rockstar(int nArgs,
			   void *argv[])
{

  long nHaloes_SF = *(long*) argv[0];    /* Number of haloes in Subfind catalogue */
  long* ids_SF = (long*) argv[1];       /* List of IDs in subfind cat */
  long* offset_SF = (long*) argv[2];    /* List of particle offsets in subfind cat */
  long* length_SF = (long*) argv[3];    /* List of particle lengths in subfind cat */
  double* partMass_SF = (double*) argv[4];   /* Masses of particles in Subfind */
  double* haloMass_SF = (double*) argv[5];   /* Masses of subhaloes in Subfind */

  long nHaloes_RS = *(long*) argv[6];         /* Number of Rockstar haloes */
  long* ids_RS = (long*) argv[7];            /* Rockstar ID list */
  long* offset_RS = (long*) argv[8];         /* Particle offsets in Rockstar */
  long* length_RS = (long*) argv[14];
  double* haloMass_RS = (double*) argv[9];   /* Halo masses in Rockstar */
  long nIDs_RS = *(long*) argv[10];

  int* matchHaloesRS = (int*) argv[11];
  double* matchFracRS = (double*) argv[12];
  double* matchFracSF = (double*) argv[13];
  
  int* haloList_SF = (int*) argv[15];
  long maxID_SF = *(long*) argv[16];
  double dMinFrac = *(double*) argv[17];

  /* 1.) Build reverse ID list for Rockstar */

  std::vector<long> vIDs_RS, vHist_offset_RS, vHist_revind_RS, vOffset_RS, vLength_RS, vOrderRS;
  std::vector<double> vMassTabInRS(nHaloes_RS, 0);
  std::vector<long> vNumTabInRS(nHaloes_RS, 0);

  vIDs_RS.assign(ids_RS, ids_RS+nIDs_RS);
  vOffset_RS.assign(offset_RS, offset_RS+nHaloes_RS+1);
  vLength_RS.assign(length_RS, length_RS+nHaloes_RS);
  
  int nMin = 0;
  int nMax = max<long>(vIDs_RS);
  int ii;

  if (maxID_SF > nMax)
    nMax = maxID_SF;

  std::cout << "Max ID in Rockstar input determined as " << nMax << std::endl;
  
  std::vector<long> vPartHist_RS = idlhist<long, long>(vIDs_RS, vHist_offset_RS, vHist_revind_RS, nMin, nMax, 0, 0);
  std::cout << "Created ID histogram for Rockstar... " << nMax << std::endl;
  
  std::vector<int> vSuccHaloRS;
  std::vector<double> vSuccFracRS, vSuccFracSF;
  
  /* 1b) Build full RS halo vector */

  /*  std::vector<long> vFullRSIndexList(nIDs_RS);
  for (long ii = 0; ii < nIDs_RS; ii++)
  vFullRSIndexList.at(ii) = vHist_revind_RS.at(ii); */

  std::vector<int> vFullRSHaloes = index_to_halo(vHist_revind_RS, vOffset_RS, vLength_RS, vOrderRS); 
  
  std::cout << "Finished index-to-halo conversion..." << std::endl;


  /*std::vector<int> vMatchRSHaloes(nHaloes_SF, -1);
  std::vector<double> vMatchRSFrac(nHaloes_SF, 0);
  std::vector<double> vMatchSFFrac(nHaloes_SF, 0); */

  /* 2.) Loop through individual SF haloes... */

  for (int iii = 0; iii < nHaloes_SF; iii++)
    {

      ii = haloList_SF[iii];

      if (ii >= 0) {
	std::cout << "SF-halo " << ii << "..." << std::endl;
	std::cout << "   (length=" << length_SF[ii] << ", offset_SF=" << offset_SF[ii] << ", M_SF = " << haloMass_SF[ii] << ")" << std::endl;
      }
      if (length_SF[ii] < 20)   /* Can probably make this threshold much more aggressive... */
	continue;
      
      for (int kk = 0; kk < nHaloes_RS; kk++) {
	vMassTabInRS.at(kk) = 0;
	vNumTabInRS.at(kk) = 0;
      }

      vSuccHaloRS.clear();
      vSuccFracRS.clear();
      vSuccFracSF.clear();
      
      /* Loop through individual halo particles */

      double dTotPartMass_SF = 0;
      for (long jj = offset_SF[ii]; jj < offset_SF[ii]+length_SF[ii]; jj++) 
	{
	  long currID = ids_SF[jj];
	  if (currID < 0 || currID >= vPartHist_RS.size()) {
	    std::cout << "Encountered invalid ID (=" << currID << ") in particle jj=" << jj << ", halo ii=" << ii << "!!" << std::endl;
	    exit(44);
	  }

	  double currMass = partMass_SF[jj];

	  dTotPartMass_SF += currMass;
	  
	  //if (jj < 100) 
	    //std::cout << "   Particle jj=" << jj << ", ID=" << currID << ", mass=" << currMass << std::endl;
	  
	  if (vPartHist_RS.at(currID) == 0) {
	    //if (jj < 100)
	    //std::cout << "      Not in RS ID list!" << std::endl;
	    continue;
	  }
	  
	  long currRS_offset = vHist_offset_RS.at(currID);
	  //if (jj < 100)
	  //std::cout << "      Particle has " << vPartHist_RS.at(currID) << " entries, starting at " << currRS_offset << std::endl;
	  
	  /* Now loop through matching haloes for current particle */
	  //std::vector<int> vCheckDupl(nHaloes_RS, -1);
	  for (int mm = 0; mm < vPartHist_RS.at(currID); mm++)
	    {
	      long currRSHalo = vFullRSHaloes.at(currRS_offset+mm);

	      if (currRSHalo >= 0)
		{ 
		  //if (vCheckDupl.at(currRSHalo) >= 0) {
		  //  std::cout << "Ha! Multiple depositions to same halo! (ii=" << ii << ", jj=" << jj << ", halo = " << currRSHalo << ", from mm=" << vCheckDupl.at(currRSHalo) << ")" << std::endl;
		  //  exit(44);
		  // } 
		  // vCheckDupl.at(currRSHalo) = mm;
		  
		  vMassTabInRS[currRSHalo] += currMass;
		  vNumTabInRS[currRSHalo] += 1;
		}
	    }
	  
	} /* ends loop through particles on current SF halo */
      
      
      /* Convert RS masstab relative to total RS halo masses */
      for (long kk = 0; kk < nHaloes_RS; kk++) 
	{
	  if (vNumTabInRS.at(kk) < 20)
	    continue;
	  
	  if (haloMass_RS[kk] <= 0) {
	    std::cout << "WAAAAAAAAAAARNING: haloMass_RS[kk=" << kk << "] = " << haloMass_RS[kk] << "!!" << std::endl;
	  }
	  
	  double dRelMassRS = vMassTabInRS.at(kk) / haloMass_RS[kk];
	  double dRelMassSF = vMassTabInRS.at(kk) / haloMass_SF[ii];
	  
	  //std::cout << "   RS-halo " << kk << " has match fracs (" << dRelMassRS << "/" << dRelMassSF << ") " << std::endl;
	  
	  if (dRelMassRS >= dMinFrac && dRelMassSF >= dMinFrac) {
	    vSuccHaloRS.push_back(kk);
	    vSuccFracRS.push_back(dRelMassRS);
	    vSuccFracSF.push_back(dRelMassSF);
	  }

	} /* ends loop through RS haloes */

      if (vSuccHaloRS.size() == 0)   /* NOT ONE suitable match halo */
	continue;
      
      if (vSuccHaloRS.size() == 1) {
	matchHaloesRS[iii] = vSuccHaloRS.front();
	matchFracRS[iii] = vSuccFracRS.front();
	matchFracSF[iii] = vSuccFracSF.front();
      } else {
	std::cout << "WARNING: Halo " << ii << " has " << vSuccHaloRS.size() << " matches in Rockstar..." << std::endl;
	
	double dMaxSum = 0;
	int nBestMatch = -1;

	std::vector<double> dSums(vSuccHaloRS.size(), 0);

	for (int kk = 0; kk < vSuccHaloRS.size(); kk++) {
	  double currSum = vSuccFracRS.at(kk) + vSuccFracSF.at(kk);
	  dSums.at(kk) = currSum; 

	  if (currSum > dMaxSum) {
	    nBestMatch = kk;
	    dMaxSum = currSum;
	  }
	  
	}

	int nAtMaxSum = 0;
	for (int kk = 0; kk < vSuccHaloRS.size(); kk++) {
	  if (dSums.at(kk) == dMaxSum)
	    nAtMaxSum++;
	}

	if (nAtMaxSum == 0) {
	  std::cout << "... WFT? No halo at maximum sum (= " << dMaxSum << ")?" << std::endl;
	} else if (nAtMaxSum == 1) {
	  std::cout << "... Best match is halo " << nBestMatch << " with sum = " << dMaxSum << std::endl;
	  matchHaloesRS[iii] = vSuccHaloRS.at(nBestMatch);
	  matchFracRS[iii] = vSuccFracRS.at(nBestMatch);
	  matchFracSF[iii] = vSuccFracSF.at(nBestMatch);
	} else {
	  std::cout << "... there are " << nAtMaxSum << " haloes with equally-best sum (= " << dMaxSum << ")..." << std::endl;
	  
	  int nSubBest = -1;
	  double dMaxMass = 0;
	  for (int kk = 0; kk < vSuccHaloRS.size(); kk++) {
	    if (dSums.at(kk) == dMaxSum) {
	      double currMass = haloMass_RS[vSuccHaloRS.at(kk)];
	      if (currMass > dMaxMass) {
		nSubBest = kk;
		dMaxMass = currMass;
	      }
	    }
	  }

	  std::cout << "... most massive of these is " << nSubBest << std::endl;
	  matchHaloesRS[iii] = vSuccHaloRS.at(nSubBest);
	  matchFracRS[iii] = vSuccFracRS.at(nSubBest);
	  matchFracSF[iii] = vSuccFracSF.at(nSubBest);
	  
	} /* ends super-unlikely section for multiple haloes with equally-best sum */
      } /* ends section for multiple eligible match haloes */

      std::cout << "Halo ii=" << ii << " matched to RS-" << matchHaloesRS[iii] << " (" << matchFracRS[iii] << "/" << matchFracSF[iii] << ")" << std::endl;
      
    } /* ends loop through SF haloes */

  std::cout << "c CODE finished" << std::endl;

  return 0;
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
  int nHistoLength = nMax-nMin+1;
  vOffset.resize(nHistoLength+1, 0);
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

double GetTime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long usec = tv.tv_usec;
  usec += (tv.tv_sec * 1e6);
  return (static_cast<double>(usec))/1e6;
}


double ElTime()
{
  static double dTimeOfLastCall = GetTime();

  double dTimeSinceLastCall = GetTime()-dTimeOfLastCall;
  
  dTimeOfLastCall = GetTime();
  return dTimeSinceLastCall;
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
  
  return idx;
}


std::vector<int> index_to_halo(std::vector<long> &vIndex,              // Index list (particles)
			       const std::vector<long> &vOffset,       // HPOlist
			       const std::vector<long> &vLength,       // Length list
			       std::vector<long> &vOrder     ) {       // Order in remote halo
  
  using namespace std;
  double dStartTime = GetTime();
  double dDummy = ElTime();

  std::vector<int> vHaloes(vIndex.size(),-1);
  vOrder.clear();
  vOrder.resize(vIndex.size(),-1);

  if (vIndex.size() == 0){
    return vHaloes;
  }

#ifdef DEBUG
  if (mpirank() == 0) {
    cout << "First 10 offset/length pairs in B: " << endl;
    for (int ii=0;ii<10;ii++) {
      cout << ii << ": " << vOffset[ii] << " --> " << vOffset[ii]+vOffset[ii+1]-1 << endl; } 
  }
#endif  
  
  // Sort input indices:
  vector <long> vSortVec = sort_indices<long> (vIndex);
  
  cout << "   Sorting matched indices took " << ElTime() << "sec. " << endl;
  
  // Find first element where index is non-negative:
  long i_min=0;

#ifdef DEBUG
  cout << rp() + "vIndex[vSortVec[0]] = " << vIndex.at(vSortVec.at(i_min)) << endl;
#endif

  while (vIndex.at(vSortVec.at(i_min)) < 0) { 
    i_min++;
    if (i_min >= vSortVec.size()) {
      cout << "No matches at all???" << endl;
      i_min = -1;
      break;
    }
  }
  
#ifdef VERBOSE
  cout << rp() + "First non-negative index at position " << i_min << ": " << vIndex.at(vSortVec.at(i_min)) << endl;
#endif

  // If there are no matched IDs at all, we can stop right here:
  if (i_min < 0) {
    cout << "Found no matching IDs at all. Proceed at own risk..." << endl;
    return vHaloes; 
  }
  
  // Katamaran-loop through individual indices to find haloes:
  long i = i_min;
  long h = 0;
  long currhalo_max = vOffset.at(h) + vLength.at(h) - 1;
  long currhalo_min = vOffset.at(h);

#ifdef VERBOSE
  cout << rp() + "Halo identification loop..." << endl;
#endif  

  while (1) {
    
    long currPos = vSortVec.at(i); // for convenience
    long currInd = vIndex.at(currPos);

    if (currInd <= currhalo_max) {
      if (currInd >= currhalo_min) {
	vHaloes.at(currPos) = h;    // only mark as belonging to current halo if it does actually 
	vOrder.at(currPos) = currInd - vOffset.at(h);
      }
      i++; // go to next particle
      
      if (i >= vIndex.size())
	break;

    } else {
      // What to do if we have exhausted the indices belonging to current halo:
      h++;
      if (h >= (vOffset.size()-1))
	break;
      
      currhalo_min = vOffset.at(h);
      currhalo_max = vOffset.at(h) + vLength.at(h) - 1;  // h has already been incremented 
    }
      
  } // ends katamaran loop

#ifdef VERBOSE
  long nTest = 0;
  for (long ii=0;ii<vIndex.size(); ii++)
    { if (vHaloes.at(ii) >= 0)
	nTest++;
    }
  cout << rp() << nTest << " elements (out of " << vIndex.size() << ") are in a halo" << endl;

  if (mpirank() == 0)
    cout << rp() + "   Finished matching particles to haloes (" << GetTime() - dStartTime << " sec. " << endl;
#endif
  
  return vHaloes;
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


