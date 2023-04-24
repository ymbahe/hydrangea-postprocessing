#include <mpi.h>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <sstream>
#include <string.h>

//#include "/u/ybahe/ANALYSIS/Protea-H/Config.h"
//#include "/u/ybahe/ANALYSIS/Protea-H/globals.h"

#include "/u/ybahe/cpplib/utilities.h"
#include "/u/ybahe/cpplib/utilities_mpi.h"


int mpirank() {

  static int nRank = -1;

  if (nRank < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &nRank);

  return nRank;

}

int numtasks() {

  static int nNumTasks = -1;
  
  if (nNumTasks < 0)
    MPI_Comm_size(MPI_COMM_WORLD, &nNumTasks);

  return nNumTasks;
}


long find_global_maximum(std::vector<long> vector) {

  using namespace std;

  // Find *local* maximum of vector:
  long local_max = max<long>(vector);
  
  // Compare local_max values across tasks:
  int rc = 0;
  long global_max = 0;
  
  rc = MPI_Allreduce(&local_max, &global_max, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD); 

  return global_max;
}

void broadcast_string(std::string &s,    // String to be broadcast
		      int nRoot) {       // Root task

  int StringLength;

  if (mpirank() == nRoot)
    StringLength = strlen(s.c_str())+1;  // Must manually include the null-terminator 

  MPI_Bcast(&StringLength, 1, MPI_INT, nRoot, MPI_COMM_WORLD);     

  char *cstring = new char[StringLength]; 

  if (mpirank() == nRoot)
    for (int ii = 0; ii < StringLength; ii++)
      cstring[ii] = s[ii];

  MPI_Bcast(cstring, StringLength, MPI_CHAR, nRoot, MPI_COMM_WORLD);

  s = std::string(cstring);
  delete[] cstring;

  return;

}


