#include "Config.h"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <algorithm>

#include "/u/ybahe/cpplib/utilities.h"
#include "utilities_special.h"
#include "globals.h"

#ifdef USE_OLD_FUNCTION_SWITCH
void function_begin(std::string sFuncName)
{
  g_vFunctionStartTimes.push_back(GetTime());

  for (int ii = 0; ii < g_function_indent-1; ii++)
    g_rp.push_back(' ');

#ifndef DEBUG
  if (g_rank == 0)
#endif
    std::cout << g_rp + ">> Beginning " + sFuncName << "()" << std::endl;

  g_rp.push_back(' ');
  
  return;
}


void function_end(std::string sFuncName)
{
  double dDuration = GetTime() - g_vFunctionStartTimes.back();


  g_vFunctionStartTimes.pop_back();
  g_rp.resize(g_rp.size()-1);


#ifndef DEBUG
  if (g_rank == 0)
#endif
    std::cout << g_rp + "<< Completed " + sFuncName + "() in " << dDuration << " sec. " << rt() << std::endl;;
  
  g_rp.resize(g_rp.size()-(g_function_indent-1));
    
  return;
}
#endif

void initialize_result(Result &rResult,
		       int nLength) {
  
  rResult.Match.clear();
  rResult.Match.resize(nLength, -1);

  rResult.Length.clear();
  rResult.Length.resize(nLength, -1);

  rResult.Rank.clear();
  rResult.Rank.resize(nLength, -1);

  rResult.Choice.clear();
  rResult.Choice.resize(nLength, -1);

  rResult.LinkInd.clear();
  rResult.LinkInd.resize(nLength, -1);

  rResult.SenderFraction.clear();
  rResult.SenderFraction.resize(nLength, -1);

  rResult.ReceiverFraction.clear();
  rResult.ReceiverFraction.resize(nLength, -1);
  
  return;

}


void copy_result(const Result &Orig,
		 Result &Copy) {

  Copy.Match = Orig.Match;
  Copy.Length = Orig.Length;
  Copy.Rank = Orig.Rank;
  Copy.Choice = Orig.Choice;
  Copy.LinkInd = Orig.LinkInd;
  Copy.SenderFraction = Orig.SenderFraction;
  Copy.ReceiverFraction = Orig.ReceiverFraction;

  return;

}

void print_result(const Result &rResult,
		  int nLoc) {
  
  using std::cout;
  using std::endl;

  std::cout << rp() + "Entry        " << nLoc << endl;
  std::cout << rp() + "-----------------------------" << endl;
  std::cout << rp() + "Match:       " << rResult.Match.at(nLoc) << endl;
  std::cout << rp() + "Length:      " << rResult.Length.at(nLoc) << endl;
  std::cout << rp() + "Rank:        " << rResult.Rank.at(nLoc) << endl;
  std::cout << rp() + "Choice:      " << rResult.Choice.at(nLoc) << endl;
  std::cout << rp() + "SenderFrac:  " << rResult.SenderFraction.at(nLoc) << endl;
  std::cout << rp() + "RecvFrac:    " << rResult.ReceiverFraction.at(nLoc) << endl;
  std::cout << rp() + "LinkInd:     " << rResult.LinkInd.at(nLoc) << endl;
  std::cout << std::endl;

  return;
}

#ifdef USE_OLD_FILENAME_EST
std::string filename_est(int nSnap,
			 int nSeq) {
  
  using namespace std;
  
  vector<std::string> vZString(29);

  

  vZString[0] = "z020p000";
  vZString[1] = "z015p132";
  vZString[2] = "z009p993";
  vZString[3] = "z008p988";
  vZString[4] = "z008p075";
  vZString[5] = "z007p050";
  vZString[6] = "z005p971";
  vZString[7] = "z005p487";
  vZString[8] = "z005p037";
  vZString[9] = "z004p485";

  vZString[10] = "z003p984";
  vZString[11] = "z003p528";
  vZString[12] = "z003p017";
  vZString[13] = "z002p478";
  vZString[14] = "z002p237";
  vZString[15] = "z002p012";
  vZString[16] = "z001p737";
  vZString[17] = "z001p487";
  vZString[18] = "z001p259";
  vZString[19] = "z001p004";

  vZString[20] = "z000p865";
  vZString[21] = "z000p736";
  vZString[22] = "z000p615";
  vZString[23] = "z000p503";
  vZString[24] = "z000p366";
  vZString[25] = "z000p271";
  vZString[26] = "z000p183";
  vZString[27] = "z000p101";
  vZString[28] = "z000p000";

  string sSnap = "0" + to_string(static_cast<long long>(nSnap));
  if (nSnap < 10)
    sSnap = "0" + sSnap;

  string sSeq = to_string(static_cast<long long>(nSeq));

  std::string dirstring = g_sFileNameRoot + "/data/groups_" + sSnap + "*";

  glob_t glob_result;
  glob(dirstring.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (glob_result.gl_pathc != 1) {
    cout << "Difficulty making up pathnames..." << endl;
    exit(99);
  }
  std::string sGroupName = string(glob_result.gl_pathv[0]);
  //  cout << "Determined group name as: " << sGroupName << endl;

  globfree(&glob_result);
  vector<string> vPathParts = split_string(sGroupName, '/');
  vector<string> vFilenameParts = split_string(vPathParts.back(), '_');
  string sZString = vFilenameParts.back();

  string filename = g_sFileNameRoot + "/data/groups_" + sSnap + "_" + sZString + "/eagle_subfind_tab_" + sSnap + "_" + sZString + "." + sSeq + ".hdf5";
  

  /*  
  DIR *dir;
  struct dirent *de;


  char tab2[1024];
  strcpy(tab2, dirstring.c_str());
  
  dir = opendir(tab2);
  while(dir)
    {
      de = readdir(dir);
      if (!de) break;
      printf("%i %s\n", de->d_type, de->d_name);
    }
  closedir(dir);
  */

  //  string filename = g_sFileNameRoot + "/data/groups_" + sSnap + "_" + vZString.at(nSnap) + "/eagle_subfind_tab_" + sSnap + "_" + vZString.at(nSnap) + "." + sSeq + ".hdf5";
  
  return filename;

}
#endif
