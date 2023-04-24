#ifndef GLOBAL_H
#define GLOBAL_H

#include "init.h"

#include <string>
#include <vector>

extern int g_nLinkRatio;
extern std::string g_sFileNameRoot;
extern std::string g_sFileNameOut;
extern RunParams Params;
extern std::vector<double> g_vFunctionStartTimes;

extern int g_nHaloesA;
extern int g_nHaloesB;
extern int g_nHaloesC;

extern int g_nSnapshots;
extern int g_ttSnap;
extern int g_nSnapA;
extern int g_nSnapB;
extern int g_nSnapC;

extern int g_nGalaxies;


extern int g_CurrSnapInd;

extern std::string g_sSubDirA;
extern std::string g_sSubDirB;
extern std::string g_sSubDirC;

extern double g_aexp_curr;
extern double g_hubble_curr;

const int g_nMaxNetworkIter = 10;

#endif 
