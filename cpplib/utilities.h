// utilities

#ifndef UTILITIES_H
#define UTILITIES_H

int file_test(std::string sPath);
bool fileExists(std::string fileName);

int ipow(int base, int exp); // Integer exponentiation function
long ipow_long(int base, int exp);

std::string rp(std::string sReplString = "", 
	       int FlagSwitch = -1);


std::string join_strings(std::vector<std::string> vStrings, char delim, 
			 int first = 0, int last = -1); 

std::vector<std::string> &split_string(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split_string(const std::string &s, char delim);

double GetTime();
std::string TimeStamp();

double ElTime();

std::string rt();

void print_n_times(int nIt_variable,
		   int nNumTot,
		   int nNumPrint,
		   std::string sMessage);

template <typename T>
void print_vector(const std::vector<T> &v, std::string prefix, size_t nBeg = 0, size_t nEnd = 0); 

template <typename T>
std::vector<T> make_offset(const std::vector<T> &vInput,
			   T nMinVal,
			   T nMaxVal,
			   int nVerb = 0);


template <typename T1, typename T2>
std::vector<T1> idlhist(const std::vector<T2> &vInput,    // Input vector
			std::vector<T1> &vOffset,         // [O] Offset list
			std::vector<T1> &vRevInds,        // [O] Reverse indices
			int &nMin,                        // [O] Minimum
			int &nMax,                        // [O] Maximum
			int FlagDetMinMax,                // if 0, take given min/max
			int nVerb);                       // Switch for verbosity

std::string complete_path(std::string sPathStart,     // Start of path to be completed 
			  int FlagFirstOnly = 0);     // If == 1, deal with multiple matches


template <typename T>
void chunk_split_list(const std::vector<T> &vFullDataOffset,
		      int parts,
		      std::vector<T> &vTDOlist,
		      std::vector<T> &vTCOlist,
		      int nVerb);


template <typename T> 
std::vector<long> sort_indices(const std::vector<T> v, int verbose = 0);

template <typename T> 
std::vector<long> sort_indices_descending(const std::vector<T> v, int verbose = 0);

template <typename T>
T min(std::vector<T> v, T thresh, int &success);

template <typename T>
T min(std::vector<T> v, std::vector<int> vMask);

template <typename T>
T min(std::vector<T> v);

template <typename T>
T max(std::vector<T> v);

template <typename T>
T max(std::vector<T> v, std::vector<int> vMask);

template <typename T> 
std::vector<T> minmax(std::vector<T> v, T thresh, int &success);

template <typename T> 
std::vector<T> minmax(std::vector<T> v);

template<typename T>
std::string to_string(T var, int nPad = 0, char cPad = '0');


void function_switch(std::string sFuncName,      // Name of function (for output)
		     int mode = -1,              // 1-->begin, 0-->end
		     int nSwitchIndent = -1);    // If >= 0, adjust indent (only) 

// Simple convenience function to return the directory for a given file name
std::string file_to_dir(std::string sFileName);

int ilog2(long val, char FlagLower = 0);

#include "utilities.tpp"

#endif
