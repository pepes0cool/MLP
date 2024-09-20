#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include "list/listheader.h"
using namespace std;
#include "ann/funtions.h"
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"
#include "ann/dataloader.h"
#include "list/DLinkedList.h"
#include "list/DLinkedListDemo.h"
#include "util/Point.h"
#include "list/IList.h"
int main(int argc, char** argv) {
    int nsamples = 80;
    xt::xarray<double> X = xt::random::randn<double>({nsamples, 10});
    xt::xarray<double> T = xt::random::randn<double>({nsamples, 5});
    TensorDataset<double, double> ds(X, T);
    DataLoader<double, double> loader(&ds, 30, true, false);
    for (auto& batch : loader) {
        cout << "Data shape: " << xt::adapt(batch.getData().shape()) << endl;
        cout << "Label shape: " << xt::adapt(batch.getLabel().shape()) << endl;
    }

    return 0;
}

