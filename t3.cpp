#include <iostream>
#include <string>
#include <chrono>
using namespace std;

#ifdef OPENACC__
#include <openacc.h>
#endif

#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif

#include <cublas_v2.h>

int max_iterations;
double max_err;
int size_arr;
constexpr double negOne = -1;

int main(int argc, char *argv[]) {
    max_err = atof(argv[argc - 3]);
    size_arr = stoi(argv[argc - 2]);
    max_iterations = stoi(argv[argc - 1]);

    double *arr{new double[size_arr * size_arr]{}};
    double *new_arr{new double[size_arr * size_arr]{}};
    double *inter{new double[size_arr * size_arr]{}};

    #pragma acc enter data create(arr[0:size_arr * size_arr], new_arr[0:size_arr * size_arr])

    auto start = chrono::high_resolution_clock::now();

    arr[0] = 10;
    arr[size_arr - 1] = 20;
    arr[(size_arr - 1) * size_arr] = 20;
    arr[size_arr * size_arr - 1] = 30;
    new_arr[0] = 10;
    new_arr[size_arr - 1] = 20;
    new_arr[(size_arr - 1) * size_arr] = 20;
    new_arr[size_arr * size_arr - 1] = 30;
    double step = 10.0 / (size_arr - 1);

    #pragma acc parallel loop present(arr[0:size_arr * size_arr], new_arr[0:size_arr * size_arr])
    for (int i = 1; i < size_arr - 1; i++) {
        arr[i] = 10 + i * step;
        arr[(size_arr - 1) * size_arr + i] = 20 + i * step;
        arr[i * size_arr] = 10 + i * step;
        arr[i * size_arr + size_arr - 1] = 20 + i * step;
        new_arr[i] = 10 + i * step;
        new_arr[(size_arr - 1) * size_arr + i] = 20 + i * step;
        new_arr[i * size_arr] = 10 + i * step;
        new_arr[i * size_arr + size_arr - 1] = 20 + i * step;
    }
    #pragma acc parallel loop present(arr[0:size_arr * size_arr], new_arr[0:size_arr * size_arr])
    for (int i = 1; i < size_arr - 1; i++) {
        for (int j = 1; j < size_arr - 1; j++) {
            arr[i * size_arr + j] = 0;
            new_arr[i * size_arr + j] = 0;
        }
    }

    auto elapsed = chrono::high_resolution_clock::now() - start;
	long long msec = chrono::duration_cast<chrono::microseconds>(elapsed).count();
    cout << "initialisation: " << msec << "\n";

    int iterations = 0;
    double err = 1;
    double* swap;
    double* a = arr;
    double* na = new_arr;
    int maxim = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status;

    auto nstart = chrono::high_resolution_clock::now();
    #pragma acc enter data copyin(a[0:size_arr * size_arr], na[0:size_arr * size_arr], inter[:size_arr * size_arr], err)
    #ifdef NVPROF_
    nvtxRangePush("MainCycle");
    #endif
    int o = 0;
    while ((err > max_err) && (iterations < max_iterations))
    {
        iterations++;
        #pragma acc parallel present(err)
        err = 0;
        #pragma acc parallel loop present(a[0:size_arr * size_arr], na[0:size_arr * size_arr], err) reduction(max:err)
        for (int i = 0; i < (size_arr - 2) * (size_arr - 2); i++) {
            na[(i / (size_arr - 2) + 1) * size_arr + 1 + i % (size_arr - 2)] = (a[(i / (size_arr - 2) + 2) * size_arr + 1 + i % (size_arr - 2)] + a[(i / (size_arr - 2)) * size_arr + 1 + i % (size_arr - 2)] + a[(i / (size_arr - 2) + 1) * size_arr + 2 + i % (size_arr - 2)] + a[(i / (size_arr - 2) + 1) * size_arr + i % (size_arr - 2)]) / 4;
            na[(i % (size_arr - 2) + 1) * size_arr + 1 + i / (size_arr - 2)] = na[(i / (size_arr - 2) + 1) * size_arr + 1 + i % (size_arr - 2)];
        }

        swap = a;
        a = na;
        na = swap;

        #ifdef OPENACC__
            acc_attach((void**)a);
            acc_attach((void**)na);
        #endif

        if (o % 100 == 0){
            #pragma acc data present(inter[:size_arr * size_arr], new_arr[:size_arr * size_arr], arr[:size_arr * size_arr]) async
            {
            #pragma acc host_data use_device(new_arr, arr, inter)
                {

                    status = cublasDcopy(handle, size_arr * size_arr, arr, 1, inter, 1);

                    status = cublasDaxpy(handle, size_arr * size_arr, &negOne, new_arr, 1, inter, 1);
                
                    status = cublasIdamax(handle, size_arr * size_arr, inter, 1, &maxim);

                }
            }

            #pragma acc update self(inter[maxim - 1]) wait
            err = fabs(inter[maxim - 1]);
        }

        o++;
    }

    #ifdef NVPROF_
    nvtxRangePop();
    #endif

    #pragma acc exit data delete(na[0:size_arr * size_arr]) copyout(a[0:size_arr * size_arr])

    auto nelapsed = chrono::high_resolution_clock::now() - nstart;
	msec = chrono::duration_cast<chrono::microseconds>(nelapsed).count();
    cout << "While: " << msec << "\n";

    cout << "Result\n";
    cout << "iterations: " << iterations << " error: " << err << "\n";

    cublasDestroy(handle);
    delete[] arr;
    delete[] new_arr;

    return 0;
}