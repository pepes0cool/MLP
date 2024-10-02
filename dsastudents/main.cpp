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
string shape2str(const xt::svector<size_t>& shape) {
    stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i < shape.size() - 1) {
            ss << ", ";
        }
    }
    ss << ")";
    return ss.str();
}

int main(int argc, char** argv) {
    int nsamples = 100;
  xt::xarray<double> X = xt::random::randn<double>({nsamples, 10});
  xt::xarray<double> T = xt::random::randn<double>({nsamples, 5});
  TensorDataset<double, double> ds(X, T);
  DataLoader<double, double> loader(&ds, 30, true, true);
  //! expect
  string expect = "(30, 10);(30, 5) (30, 10);(30, 5) (30, 10);(30, 5) ";

  //! output

  stringstream output;
  for (auto batch : loader) {
    output << shape2str(batch.getData().shape()) << ";";
    output << shape2str(batch.getLabel().shape()) << " ";
  }
  if(output.str() == expect){
    cout << "TRUE\n";
  }else cout <<"FALSE\n";

    return 0;
}

