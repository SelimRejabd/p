#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <algorithm>

using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    const int max_name_length = 100;
    int found_number = 0;
    char search_name[max_name_length];
    bool stopSearch = false;
    bool printed = false;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // This part checks if at least two file names are given
    if (argc < 2)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: mpiexec -n <number_of_processes> ./your_executable file1.txt file2.txt ...\n";
        }
        MPI_Finalize();
        return 1;
    }

    vector<int> lines_per_file(argc, 0);

    // Print file names in rank 0
    if (rank == 0)
    {
        cout << "Give the name you want to search: ";
        cin.getline(search_name, max_name_length);
        transform(search_name, search_name + strlen(search_name), search_name, ::toupper);
        for (int i = 1; i < argc; ++i)
        {
            ifstream file(argv[i]);
            if (file.is_open())
            {
                std::string line;
                int lineCount = 0;
                while (std::getline(file, line))
                {
                    ++lineCount;
                }
                file.close();
                lines_per_file[i] = lineCount;
            }
            else
            {
                std::cerr << "Unable to open file: " << argv[i] << "\n";
            }
        }
    }

    // Broadcast linesPerFile to all processes
    MPI_Bcast(lines_per_file.data(), lines_per_file.size(), MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast the search name from Process 0 to all other processes
    MPI_Bcast(search_name, max_name_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 1; i < argc; i++)
    {
        if (stopSearch)
            break;
        ifstream file(argv[i]);
        if (file.is_open())
        {
            string line;
            int linesToRead = lines_per_file[i] / size;
            int startLine = rank * linesToRead;
            if (rank == size - 1)
            {
                linesToRead += lines_per_file[i] % size;
            }
            for (int j = 0; j < startLine; ++j)
            {
                getline(file, line);
            }
            for (int j = 0; j < linesToRead; ++j)
            {
                if (stopSearch)
                    break;
                getline(file, line);
                istringstream iss(line);
                string name;
                int number;
                iss >> name >> number;
                if (iss)
                {
                    transform(name.begin(), name.end(), name.begin(), ::toupper);
                    if (name == search_name)
                    {
                        cout << name << " : " << number << " Found in rank: " << rank << " file: " << argv[i] << endl;
                        found_number = number;
                        stopSearch = true;
                        break;
                    }
                }
                else
                {
                    std::cerr << "Failed to extract name and number from the line.\n";
                }
            }
            file.close();
        }
        else
        {
            std::cerr << "Unable to open file: " << argv[i] << "in Rank: " << rank << "\n";
        }
    }

    if (rank != 0)
    {
        MPI_Send(&found_number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        int actual_number_found = 0;
        if (found_number != 0)
        {
            actual_number_found = found_number;
        }
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(&found_number, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (found_number != 0)
            {
                actual_number_found = found_number;
            }
        }
        transform(search_name, search_name + strlen(search_name), search_name, ::tolower);
        search_name[0] = toupper(search_name[0]);
        if (actual_number_found != 0)
            cout << search_name << " : " << actual_number_found << endl;
        else
            cout << search_name << " : Not found" << endl;
    }

    MPI_Finalize();
    return 0;
}