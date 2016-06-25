#include <algorithm>

#ifndef _WIN32
#include <string.h>
#endif
using namespace std;

template <typename T> void partial_sort_cols(T *outdata, T* indata, int rank, int ncols, int nrows) {
    T *tmpmem = new T[nrows];
    T *start = indata;
    T *pOutdata = outdata;
    for (int i = 0; i < ncols; i++) {
        for (int j = 0; j < nrows; j++) {
            tmpmem[j] = start[j];
        }
        // Run nth_element to iteratively partition to the specified rank
        std::partial_sort(tmpmem, tmpmem + rank, tmpmem + nrows);
        for (int j = 0; j < rank; j++) {
            pOutdata[j] = tmpmem[j];
        }
        start += nrows;
        pOutdata += rank;
    }
    delete []tmpmem;
}

template <typename T, typename T1> void partial_sort_cols_withIdx (T *outdata, T1 *idx, T* indata, int rank, int ncols, int nrows) {
    pair<T, int> *tmpmem = new pair<T, int>[nrows];
    T *start = indata;
    T *pOutdata = outdata;
    T1 *pIdx = idx;
    for (int i = 0; i < ncols; i++) {
        for (int j = 0; j < nrows; j++) {
            tmpmem[j].first = start[j];
            tmpmem[j].second = j+1;  // +1
        }
        // Run nth_element to iteratively partition to the specified rank
        std::partial_sort(tmpmem, tmpmem + rank, tmpmem + nrows);
        for (int j = 0; j < rank; j++) {
            pOutdata[j] = tmpmem[j].first;
            pIdx[j] = tmpmem[j].second;
        }
        start += nrows;
        pOutdata += rank;
        pIdx += rank;
    }
    delete []tmpmem;
}

template <typename T> void partial_sort_rows(T *outdata, T* indata, int rank, int ncols, int nrows) {
    T *tmpmem = new T[ncols];
    for (int i = 0; i < nrows; i++) {
        T *start = indata + i;
        for (int j = 0; j < ncols; j++) {
            tmpmem[j] = *start;
            start += nrows;
        }
        // Run nth_element to iteratively partition to the specified rank
        std::partial_sort(tmpmem, tmpmem + rank, tmpmem + ncols);
        T *pOutdata = outdata + i;
        for (int j = 0; j < rank; j++) {
            *pOutdata = tmpmem[j];
            pOutdata += nrows;
        }
    }
    delete []tmpmem;
}

template <typename T, typename T1> void partial_sort_rows_withIdx (T *outdata, T1 *idx, T* indata, int rank, int ncols, int nrows) {
    pair<T, int> *tmpmem = new pair<T, int>[ncols];
    for (int i = 0; i < nrows; i++) {
        T *start = indata + i;
        for (int j = 0; j < ncols; j++) {
            tmpmem[j].first = *start;
            tmpmem[j].second = j+1;  // +1
            start += nrows;
        }
        // Run nth_element to iteratively partition to the specified rank
        std::partial_sort(tmpmem, tmpmem + rank, tmpmem + ncols);
        T *pOutdata = outdata + i;
        T1 *pIdx = idx + i;
        for (int j = 0; j < rank; j++) {
            *pOutdata = tmpmem[j].first;
            *pIdx = tmpmem[j].second;
            pOutdata += nrows;
            pIdx += nrows;
        }
    }
    delete []tmpmem;
}