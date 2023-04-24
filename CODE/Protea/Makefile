MPI_CXX := icpc
export HDF5_CXX = mpiicpc
export HDF5_CLINKER = mpiicpc

MPI_COMPILER = mpiicpc
EXEC = protea_freya

INCLUDE = -I${HDF5_HOME}/include  ##/ptmp/mpa/ybahe/HDF5/include

OPT = -g -O2

##OPT += -DDENSE_IDS

all: main.o load_ids.o utilities.o search_katamaran.o correlate_ids.o network.o write_result.o snipshots.o  utilities_mpi.o utilities_special.o hydrangea.o io_hdf5.o Config.h
	${HDF5_HOME}/bin/h5c++ -o $(EXEC) main.o load_ids.o utilities.o correlate_ids.o search_katamaran.o network.o write_result.o snipshots.o utilities_mpi.o utilities_special.o hydrangea.o io_hdf5.o $(CFLAGS) $(LDFLAGS)
 
main.o: main.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 main.cpp -o main.o $(CFLAGS)

load_ids.o: load_ids.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 load_ids.cpp -o load_ids.o $(INCLUDE) $(CFLAGS)

search_katamaran.o: /u/ybahe/cpplib/search_katamaran.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 /u/ybahe/cpplib/search_katamaran.cpp -o search_katamaran.o $(CFLAGS)

correlate_ids.o: /u/ybahe/cpplib/correlate_ids.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 /u/ybahe/cpplib/correlate_ids.cpp -o correlate_ids.o $(CFLAGS)

network.o: network.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 network.cpp -o network.o $(CFLAGS)

write_result.o: write_result.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 write_result.cpp -o write_result.o $(INCLUDE) $(CFLAGS) 

snipshots.o: snipshots.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 snipshots.cpp -o snipshots.o $(INCLUDE) $(CFLAGS) 

utilities.o: /u/ybahe/cpplib/utilities.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 /u/ybahe/cpplib/utilities.cpp -o utilities.o $(CFLAGS)

utilities_mpi.o: /u/ybahe/cpplib/utilities_mpi.cpp /u/ybahe/cpplib/utilities_mpi.tpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 /u/ybahe/cpplib/utilities_mpi.cpp -o utilities_mpi.o $(CFLAGS)

utilities_special.o: utilities_special.cpp 
	$(MPI_COMPILER) -c $(OPT) -std=c++11 utilities_special.cpp -o utilities_special.o $(CFLAGS)

hydrangea.o: /u/ybahe/cpplib/hydrangea.cpp /u/ybahe/cpplib/hydrangea.tpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 /u/ybahe/cpplib/hydrangea.cpp -o hydrangea.o $(CFLAGS)

io_hdf5.o: /u/ybahe/cpplib/io_hdf5.cpp
	$(MPI_COMPILER) -c $(OPT) -std=c++11 /u/ybahe/cpplib/io_hdf5.cpp -o io_hdf5.o $(INCLUDE) $(CFLAGS)



clean: 
	rm -rf *.o
	rm -f protea
