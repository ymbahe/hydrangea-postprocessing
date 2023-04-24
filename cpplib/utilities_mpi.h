// utilities_mpi

#ifndef UTILITIES_MPI_H
#define UTILITIES_MPI_H

int mpirank();

int numtasks();

template <typename T>
void collect_vector_mpi(std::vector<T> &v,
			int dest_task,
			int offset_flag,
			int clear_flag,
			int nFlagBarrier = 1);



template <typename T>
int check_for_consistency(T nVar,          // Variable whose value is checked 
			  int nMode = 0,  // 0: Terminate if unequal. 1: ret first task != root 
			  std::string sDescr = "");  // Explanatory string (printed in errmsg)  

long find_global_maximum(std::vector<long> vector);

template <typename T>
std::vector<T> find_global_minmax(std::vector<T> vTT, int nVerb = 0);

template <typename T>
void report(T var, std::string varname);

void broadcast_string(std::string &s,    // String to be broadcast
		      int nRoot);        // Root task

template <typename T>
void broadcast_struct(T &StructToBC, int nRoot = 0);

template <typename T>
void broadcast_vector(std::vector<T> &vVec, int orig_task);

// Function to find the min/max for each element in a vector

template <typename T>
void find_minmax_vector (std::vector<T> &vIn,      // [I/0] Vector to process
			 std::vector<T> &vExtreme, // [O] result (dummy if cFlagInPlace == 1)
			 char cFlagMinMax,         // 0: Min, 1: Max 
			 char cFlagInPlace = 0,    // If == 1, input overwritten with result 
			 char cFlagBroadcast = 0,  // If == 1, result broadcast to all tasks
			 char cFlagThreshold = 0,  // If == 1, values </> Threshold ignored
			 T Threhold = 0,           // Threshold value (ignored unless ^ ==1) 
			 char cVerb = 0); 



#include "utilities_mpi.tpp"

#endif



