/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   dataset.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 3:59 PM
 */

#ifndef DATASET_H
#define DATASET_H
#include "ann/xtensor_lib.h"
using namespace std;

template<typename DType, typename LType>
class DataLabel{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    DataLabel(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    xt::xarray<DType> getData() const{ return data; }
    xt::xarray<LType> getLabel() const{ return label; }
};

template<typename DType, typename LType>
class Batch{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    Batch() = default;
    Batch(const Batch& other) : data(other.data), label(other.label) {}
    Batch(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(std::move(data)), label(std::move(label)){
    }
    virtual ~Batch(){}
    xt::xarray<DType>& getData(){return data; }
    xt::xarray<LType>& getLabel(){return label; }
    bool operator==(const Batch<DType, LType>& other) const {
        return xt::all(xt::equal(this->data, other.data)) && xt::all(xt::equal(this->label, other.label));
    }
    string toString() const {
        ostringstream oss;
        oss << data << label;
        return oss.str();
    }
};
template<typename DType, typename LType>
std::ostream& operator<<(std::ostream& os, const Batch<DType, LType>& batch) {
    return os << batch.toString();
    }  //overload operator << for batch

template<typename DType, typename LType>
class Dataset{
private:
public:
    Dataset(){};
    virtual ~Dataset(){};
    
    virtual int len()=0;
    virtual DataLabel<DType, LType> getitem(int index)=0;
    virtual xt::svector<unsigned long> get_data_shape()=0;
    virtual xt::svector<unsigned long> get_label_shape()=0;
    
};

//////////////////////////////////////////////////////////////////////
template<typename DType, typename LType>
class TensorDataset: public Dataset<DType, LType>{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    xt::svector<unsigned long> data_shape, label_shape;
    
public:
    /* TensorDataset: 
     * need to initialize:
     * 1. data, label;
     * 2. data_shape, label_shape
    */
    TensorDataset(xt::xarray<DType> data, xt::xarray<LType> label){
        /* TODO: your code is here for the initialization
         */
        this->data = data;
        this->label = label;
        this->data_shape = xt::svector<unsigned long>(data.shape().begin(), data.shape().end());
        if (label.dimension() == 0) {
            this->label_shape = xt::svector<unsigned long>{};
        } else {
            this->label_shape = xt::svector<unsigned long>(label.shape().begin(), label.shape().end());
        }
    }
    /* len():
     *  return the size of dimension 0
    */
    int len() override{
        /* TODO: your code is here to return the dataset's length
         */
        return data_shape[0];
    }
    
    /* getitem:
     * return the data item (of type: DataLabel) that is specified by index
     */
    DataLabel<DType, LType> getitem(int index) override{
        /* TODO: your code is here
         */
        if(index < 0 || index >= this->len())throw out_of_range("Index is out of range!");
        xt::xarray<DType> item_data = xt::view(data, index, xt::all());
        xt::xarray<LType> item_label;
        if (label.dimension() == 0) {
           item_label = label;
        }else item_label = xt::view(label, index, xt::all());
        return DataLabel<DType, LType>(item_data, item_label);
    }
    
    xt::svector<unsigned long> get_data_shape() override{
        /* TODO: your code is here to return data_shape
         */
        return data_shape;
    }
    xt::svector<unsigned long> get_label_shape() override{
        /* TODO: your code is here to return label_shape
         */
        return label_shape;
    }
};



#endif /* DATASET_H */


