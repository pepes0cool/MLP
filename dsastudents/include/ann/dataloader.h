#ifndef DATALOADER_H
#define DATALOADER_H
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"
#include "list/XArrayList.h"

using namespace std;

template<typename DType, typename LType>
class DataLoader{
public:
private:
    Dataset<DType, LType>* ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    XArrayList<int> indices;
    int current_index;
    XArrayList<Batch<DType, LType>> batches;

public:
    DataLoader(Dataset<DType, LType>* ptr_dataset,
               int batch_size,
               bool shuffle=true,
               bool drop_last=false) : ptr_dataset(ptr_dataset), batch_size(batch_size), shuffle(shuffle), drop_last(drop_last), current_index(0) {
        /*TODO: Add your code to do the initialization */
        int length = this->ptr_dataset->len();
        for (int i = 0; i < length; ++i) indices.add(i);
        if (shuffle) {
            auto permuted_indices = xt::random::permutation(length);
            for (int i = 0; i < length; ++i) {
                indices.set(i, permuted_indices(i)); 
            }
        }
        for(int start = 0; start < length; start += batch_size){
            int end = min(start + batch_size, length);
            XArrayList<size_t> batch_indices;
            for(size_t i = start; i < end; ++i){
                batch_indices.add(indices.get(i));
            }//assign value of index to make a batch
            if(drop_last == true && end - start < batch_size)break;  
            auto first_item = ptr_dataset->getitem(batch_indices.get(0));
            xt::xarray<DType> batch_data = xt::expand_dims(first_item.getData(), 0);
            xt::xarray<LType> batch_labels = xt::expand_dims(first_item.getLabel(), 0); // make data become 2D, this handle the first item in batch
            //That ra co the bo thang vao for loop tu index 0 nhung lam nhu nay thi khong can phai check xem batch_data va label da dc init chua (?)
            for (size_t i = 1; i < batch_indices.size(); ++i)
            {   auto item = ptr_dataset->getitem(batch_indices.get(i));
                xt::xarray<DType> sample_data = xt::expand_dims(item.getData(), 0);
                xt::xarray<LType> sample_label = xt::expand_dims(item.getLabel(), 0);//handle the rest item
                
                batch_data = xt::concatenate(xt::xtuple(std::move(batch_data), std::move(sample_data)), 0);
                batch_labels = xt::concatenate(xt::xtuple(std::move(batch_labels), std::move(sample_label)), 0);
            }
            batches.add(Batch<DType, LType>(batch_data, batch_labels));
            current_index++;
        }

        if(drop_last == false && !batches.empty() && batches.get(batches.size() - 1).getData().shape(0) < batch_size){
            if(batches.size() > 1){
                int second_lastIdx = batches.size() - 2;
                int lastIdx = batches.size() - 1;
                xt::xarray<DType> merged_data = xt::concatenate(xt::xtuple(
                    std::move(batches.get(second_lastIdx).getData()),
                    std::move(batches.get(lastIdx).getData())
                ));
                xt::xarray<LType> merged_labels = xt::concatenate(xt::xtuple(
                    std::move(batches.get(second_lastIdx).getLabel()),
                    std::move(batches.get(lastIdx).getLabel())
                ));
                batches.removeAt(lastIdx);
                batches.removeAt(second_lastIdx);
                batches.add(Batch<DType, LType>(merged_data, merged_labels));
            }   
        }
    }
    virtual ~DataLoader(){}


     /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////

    /*TODO: Add your code here to support iteration on batch*/

    class Iterator
    {
    private:
        DataLoader *dataloader;
        size_t pos;

    public:
        Iterator(DataLoader *dataloader, size_t pos) : dataloader(dataloader), pos(pos) {}

        bool operator!=(const Iterator &other) const
        {
            return pos != other.pos;
        }

        bool operator==(const Iterator &other)const{
            return pos == other.pos;
        }

        Batch<DType, LType> &operator*()const
        {
            return dataloader->batches.get(pos);
        }

        const Iterator &operator++()
        {
            ++pos;
            return *this;
        }

        const Iterator &operator--()
        {
            --pos;
            return *this;
        }

        Iterator &operator=(const Iterator &other)
        {
            dataloader = other.dataloader;
            pos = other.pos;
            return *this;
        }
    };

    Iterator begin()
    {
        return Iterator(this, 0);
    }
    Iterator end()
    {
        return Iterator(this, batches.size());
    }
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};

#endif /* DATALOADER_H */
