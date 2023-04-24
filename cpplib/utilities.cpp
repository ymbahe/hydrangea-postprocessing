
#include "mpi.h"
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <sstream>
#include <iostream>
#include "glob.h"
#include <fstream>

//#include "/u/ybahe/ANALYSIS/Protea-H/Config.h"
//#include "/u/ybahe/ANALYSIS/Protea-H/globals.h"

#include "/u/ybahe/cpplib/utilities_mpi.h"
#include "/u/ybahe/cpplib/utilities.h"
#include "utilities_mpi.h"

int ipow(int base, int exp)
{
  int result = 1;
  while (exp)
    {
      if (exp & 1)
	result *= base;
      exp >>= 1;
      base *= base;
    }

  return result;
}

long ipow_long(int base, int exp)
{
  long result = 1;
  while (exp)
    {
      if (exp & 1)
	result *= base;
      exp >>= 1;
      base *= base;
    }

  return result;
}

// Function to return base-2 logarithm of integer number
// If the input is not a power of two, the **next-larger** 
// power of two is returned, unless 'FlagLower' == 1.
int ilog2(long val, char FlagLower)
{
  int result = 0;
  long nCurrP2 = 1;

  while (nCurrP2 < val)
    {
      result++;
      nCurrP2 *= 2;
    }

  if (FlagLower == 0 && nCurrP2 < val)
    result++;

  return result;
}


std::string join_strings(std::vector<std::string> vStrings, char delim, int first, int last) 
{

  if (last < 0)
    last = vStrings.size()-1;

  std::string jstring = "";
  for (int ii = first; ii < last; ii++) {
    jstring.append(vStrings.at(ii));
    jstring.append(1, delim); }
  
 jstring.append(vStrings.at(last));

  return jstring;
}

std::vector<std::string> &split_string(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}


std::vector<std::string> split_string(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split_string(s, delim, elems);
  return elems;
}

double GetTime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long usec = tv.tv_usec;
  usec += (tv.tv_sec * 1e6);
  return (static_cast<double>(usec))/1e6;
}

std::string TimeStamp()
{
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%d-%m-%Y, %X", &tstruct);
  return buf;
}

double ElTime()
{
  static double dTimeOfLastCall = GetTime();

  double dTimeSinceLastCall = GetTime()-dTimeOfLastCall;
  
  dTimeOfLastCall = GetTime();
  return dTimeSinceLastCall;
}

std::string rt() {

  static long nCounter = 0;
  static double dStartTime = -1;
  
  // Initialization if this is the first time it's called:
  if (nCounter == 0) 
    dStartTime = GetTime();
  
  double ElTime = GetTime()- dStartTime;
  std::string sRTString = " [" + to_string(static_cast<long double>(ElTime)) + " sec.]";

  nCounter++;
  return sRTString;

}


void print_n_times(int nIt_variable,
		   int nNumTot,
		   int nNumPrint,
		   std::string sMessage) {

  int nModTerm = nNumTot/nNumPrint;
  if (nModTerm < 1)
    nModTerm =1;

  if (nIt_variable % nModTerm == 0)
    std::cout << rp() << sMessage << " " << nIt_variable+1 << "/" << nNumTot << std::endl;
  
  return;
}


// This function creates an offset list from an input vector
std::vector<int> make_offset(const std::vector<int> &vInput,
			     int nMinVal,
			     int nMaxVal,
			     int nVerb) {
  
  
  if (nVerb == 1)
    function_switch("make_offset");
  
  std::vector<int> vOffsets(nMaxVal-nMinVal+2,0);
  
  int nCurrVal = nMinVal;
  
  for (int ii = 0; ii < vInput.size(); ii++) {

    int val_ii = vInput.at(ii);
    if (val_ii > nCurrVal) {
      do {
	vOffsets.at(nCurrVal-nMinVal+1) = ii;
	nCurrVal++;
      } while (val_ii > nCurrVal);

    }
  } // ends for-loop

  // Coda:
  for (int jj = nCurrVal; jj <= nMaxVal; jj++)
    vOffsets.at(jj-nMinVal+1) = vInput.size();
  
  if (nVerb == 1)
    function_switch("make_offset");

  return vOffsets;

}

int file_test(std::string sPath) {
  
  glob_t glob_result;
  glob(sPath.c_str(), GLOB_TILDE, NULL, &glob_result);
  
  return glob_result.gl_pathc;
}


std::string complete_path(std::string sPathStart,     // Start of path to be completed 
			  int FlagFirstOnly) {    // If == 1, deal with multiple matches

  std::string sSearchString = sPathStart + "*";
  
  glob_t glob_result;
  glob(sSearchString.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (glob_result.gl_pathc == 0) {
    std::cout << "Cannot find a single match for [" << sPathStart << "]" << std::endl;
    exit(1000);
  }
  
  if (glob_result.gl_pathc > 1 && FlagFirstOnly == 0) {
    std::cout << "There are " << glob_result.gl_pathc << " matches for path start " << sPathStart << std::endl;
    std::cout << "Single match enforced - exiting..." << std::endl;
    exit(1001);
  }

  std::string sCompletePath = std::string(glob_result.gl_pathv[0]);
  globfree(&glob_result);

  return sCompletePath;

}


std::string rp(std::string sReplString, 
	       int FlagSwitch) {

  static std::string sRecString = "";
  static int nFlag = 1;

  // Update internal variables if desired
  if (FlagSwitch != -1)
    nFlag = FlagSwitch;
  else if (sReplString.compare("") !=0 )
    sRecString = sReplString;
  
  // Return appropriate value
  if (nFlag == 1)
    return sRecString;
  else
    return "";

}

void function_switch(std::string sFuncName,
		     int nMode,
		     int nSwitchIndent)

{
  
  static std::vector<double> vFunctionStartTimes;
  static int nIndentCount = 3;
  static std::vector<std::string> vFunctionNames;
  
  if (vFunctionNames.size() == 0)
    vFunctionNames.push_back("MAIN");

  if (nSwitchIndent >= 0)
    nIndentCount = nSwitchIndent;
  else {

    std::string sOldRP = rp();
    
    if (sFuncName.compare(vFunctionNames.back()) != 0) // New function
      nMode = 1;
    else
      nMode = 0;
    
    if (nMode == 1)
      {
	vFunctionStartTimes.push_back(GetTime());
	vFunctionNames.push_back(sFuncName);

	if (rp().size() > 0) {

	  for (int ii = 0; ii < nIndentCount-1; ii++)
	    sOldRP.push_back(' ');
	  
	  if (mpirank() == 0)
	    std::cout << sOldRP << ">> Beginning " + sFuncName << "()" << std::endl;
	
	  sOldRP.push_back(' ');
	  rp(sOldRP);
	} else {
	  if (mpirank() == 0)
	    std::cout << ">> Beginning " + sFuncName << "()" << std::endl;
	}
      }

    else if (nMode == 0) // Function end
    
      {
 	double dDuration = GetTime() - vFunctionStartTimes.back();
	vFunctionStartTimes.pop_back();
	vFunctionNames.pop_back();

	if (sOldRP.size() > 0)
	  sOldRP.resize(sOldRP.size()-1);
	
	if (mpirank() == 0)
	  std::cout << sOldRP << "<< Completed " + sFuncName + "() in " << dDuration << " sec. " << rt() << std::endl;;
	
	if (sOldRP.size() >= (nIndentCount-1))
	  sOldRP.resize(sOldRP.size()-(nIndentCount-1));
	
	rp(sOldRP);
	
      }
  }
  return;
}


// Simple convenience function to return the directory for a given file name
std::string file_to_dir(std::string sFileName)

{
  std::vector<std::string> vsNameParts = split_string(sFileName, '/');
  std::string sDirName = join_strings(vsNameParts, '/', 0, vsNameParts.size()-2);
  return sDirName;
}

// Test whether a file exists
bool fileExists(std::string fileName) {
  std::ifstream ifile(fileName);
  return ifile;
}
