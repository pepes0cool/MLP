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
  xt::xarray<float> data = {1., 2., 3.};
  xt::xarray<int> label = 1;

  TensorDataset<float, int> tensor(data, label);
  DataLabel<float, int> getitem = tensor.getitem(2);
  xt::xarray<float> getData = getitem.getData();
  xt::xarray<int> getLabel = getitem.getLabel();

  //! expect
  string expect = " 3.;1";

  //! output
  stringstream output;
  output << getData << ";";
  output << getLabel;
  cout<< output.str();
  if(output.str() == expect)cout<< "\nTRUE";
    return 0;
}

