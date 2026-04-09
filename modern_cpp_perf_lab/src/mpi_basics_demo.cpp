#include "lab/common.hpp"

#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>

namespace {

void point_to_point_demo(int rank, int size) {
  lab::print_divider("MPI point-to-point");
  if (size < 2) {
    if (rank == 0) {
      std::cout << "Need at least 2 ranks for Send/Recv demo\n";
    }
    return;
  }

  if (rank == 0) {
    const int value = 42;
    MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    std::cout << "rank 0 sent value=" << value << " to rank 1\n";
  } else if (rank == 1) {
    int received = 0;
    MPI_Recv(&received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << "rank 1 received value=" << received << " from rank 0\n";
  }
}

void collective_demo(int rank, int size) {
  lab::print_divider("MPI collectives");

  int broadcast_value = rank == 0 ? 2026 : 0;
  MPI_Bcast(&broadcast_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::cout << "rank " << rank << " sees broadcast_value=" << broadcast_value << "\n";

  std::vector<int> scattered(static_cast<std::size_t>(2), 0);
  std::vector<int> root_values;
  if (rank == 0) {
    root_values.resize(static_cast<std::size_t>(size) * 2);
    std::iota(root_values.begin(), root_values.end(), 1);
  }

  MPI_Scatter(root_values.data(), 2, MPI_INT, scattered.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  const int local_sum = scattered[0] + scattered[1];

  std::vector<int> gathered;
  if (rank == 0) {
    gathered.resize(static_cast<std::size_t>(size));
  }
  MPI_Gather(&local_sum, 1, MPI_INT, gathered.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  int reduced_sum = 0;
  MPI_Reduce(&local_sum, &reduced_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  int allreduced_sum = 0;
  MPI_Allreduce(&local_sum, &allreduced_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  std::cout << "rank " << rank << " local_sum=" << local_sum
            << ", allreduced_sum=" << allreduced_sum << "\n";
  if (rank == 0) {
    std::cout << "root gathered partial sums:";
    for (int value : gathered) {
      std::cout << ' ' << value;
    }
    std::cout << "\nroot reduced_sum=" << reduced_sum << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "MPI initialized with size=" << size << "\n";
  }
  std::cout << "hello from rank " << rank << "\n";

  point_to_point_demo(rank, size);
  MPI_Barrier(MPI_COMM_WORLD);
  collective_demo(rank, size);

  MPI_Finalize();
  return 0;
}
