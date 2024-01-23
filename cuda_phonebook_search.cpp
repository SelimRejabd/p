%%cu
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#define T 1000
#define S 50

    using namespace std;

struct Contact
{
    char name[S];
    char phone[S];
};

// Kernel
__global__ void searchContact(Contact *contacts, const char *searchName, int *result, int totalContacts)
{
    int i = threadIdx.x;

    int searchNameLen = 0;
    for (int j = 0; searchName[j] != '\0'; j++)
        searchNameLen++;
    int contactLen = 0;
    for (int j = 0; contacts[i].name[j] != '\0'; j++)
        contactLen++;

    if (searchNameLen <= contactLen)
    {
        int strLen = searchNameLen;
        int flag = 1;

        for (int j = 0; j < strLen; ++j)
        {
            if (searchName[j] != contacts[i].name[j])
            {
                flag = 0;
                break;
            }
        }

        // This part is for  exact matching
        /*if (searchNameLen != contactLen) {
            flag = 0;
        }*/

        if (flag == 1)
        {
            result[i] = i;
        }
        else
        {
            result[i] = -1;
        }
    }
    else
    {
        result[i] = -1;
    }
}

int main()
{
    Contact contacts[T];
    char searchName[S] = "Em";

    const char *phonebookFile_1 = "/content/input1.txt";
    const char *phonebookFile_2 = "/content/input2.txt";

    int totalContacts = 0;

    ifstream file_1(phonebookFile_1);
    ifstream file_2(phonebookFile_2);

    if (!file_1.is_open() || !file_2.is_open())
    {
        cerr << "Error opening one or more files!" << endl;
        return 1;
    }

    string line;
    while (getline(file_1, line))
    {
        istringstream iss(line);
        iss.getline(contacts[totalContacts].name, S, ',');
        iss.getline(contacts[totalContacts].phone, S);
        totalContacts += 1;
    }

    while (getline(file_2, line))
    {
        istringstream iss(line);
        iss.getline(contacts[totalContacts].name, S, ',');
        iss.getline(contacts[totalContacts].phone, S);
        totalContacts += 1;
    }

    file_1.close();
    file_2.close();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // Device memory
    Contact *dContacts;
    char *dSearchName;
    int *dResult;

    int t = totalContacts;

    cudaMalloc((void **)&dContacts, sizeof(Contact) * t);
    cudaMalloc((void **)&dSearchName, S * sizeof(char));
    cudaMalloc((void **)&dResult, sizeof(int) * t);

    cudaMemcpy(dContacts, contacts, sizeof(Contact) * t, cudaMemcpyHostToDevice);
    cudaMemcpy(dSearchName, searchName, sizeof(char) * S, cudaMemcpyHostToDevice);

    dim3 gridSize(1);
    dim3 blockSize(t);

    searchContact<<<gridSize, blockSize>>>(dContacts, dSearchName, dResult, totalContacts);
    cudaDeviceSynchronize();
    int result[t] = {0};
    cudaMemcpy(result, dResult, sizeof(int) * t, cudaMemcpyDeviceToHost);

    int notFound = 1;

    for (int y = 0; y < t; y++)
    {
        if (result[y] >= 0)
        {
            notFound = 0;
            cout << contacts[result[y]].name << "  " << contacts[result[y]].phone << endl;
            // break;
        }
    }

    if (notFound)
    {
        cout << "Not Found" << endl;
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    cout << "Time taken : " << milliseconds << " miliseconds." << endl;

    // Free device memory
    cudaFree(dContacts);
    cudaFree(dResult);

    return 0;
}
