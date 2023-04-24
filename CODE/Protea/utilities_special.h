// utilities_special

#ifndef UTILITIES_SPECIAL_H
#define UTILITIES_SPECIAL_H

#include "init.h"


void print_result(const Result &rResult,
		  int nLoc);

void copy_result(const Result &Orig,
		 Result &Copy);

void initialize_result(Result &rResult,
		       int nLength);


#ifdef USE_OLD_FUNCTIONS
void function_begin(std::string sFuncName);

void function_end(std::string sFuncName);

std::string filename_est(int nSnap,
			 int nSeq);

#endif



#endif
