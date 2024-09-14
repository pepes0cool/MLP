/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/*
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"

using namespace std;

template <typename DType, typename LType>
class DataLoader
{
public:
private:
    Dataset<DType, LType> *ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    vector<int> indices; // shuffle the indices then access the dataset from indices
    int curr;
    vector<Batch<DType, LType>> batches;
    /*TODO: add more member variables to support the iteration*/
public:
    DataLoader(Dataset<DType, LType> *ptr_dataset,
               int batch_size,
               bool shuffle = true,
               bool drop_last = false) : ptr_dataset(ptr_dataset), batch_size(batch_size), shuffle(shuffle), drop_last(drop_last), curr(0)
    {
        /*TODO: Add your code to do the initialization */
        int length = ptr_dataset->len();
        indices.resize(length);
        for (int i = 0; i < length; i++)
        {
            indices[i] = i; // set the index
        }
        if (shuffle)
        {
            random_device rd;
            default_random_engine rng(rd());
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        // Initialize batches
        length = indices.size();
        for (int start = 0; start < length; start += batch_size)
        {   
            int end = min(start + batch_size, length);
            if (drop_last && end - start < batch_size)
            {
                // If drop_last is true and there are not enough samples for a full batch, skip
                break;
            }
            vector<size_t> batch_indices(indices.begin() + start, indices.begin() + end);
            xt::xarray<DType> batch_data = xt::expand_dims(ptr_dataset->getitem(batch_indices[0]).getData(), 0);
            xt::xarray<LType> batch_labels = xt::expand_dims(ptr_dataset->getitem(batch_indices[0]).getLabel(), 0); // make data become 2D, this handle the first item in batch
            //That ra co the bo thang vao for loop tu index 0 nhung lam nhu nay thi khong can phai check xem batch_data va label da dc init chua (?)
            for (size_t i = 1; i < batch_indices.size(); ++i)
            {   
                xt::xarray<DType> sample_data = xt::expand_dims(ptr_dataset->getitem(batch_indices[i]).getData(), 0);
                xt::xarray<LType> sample_label = xt::expand_dims(ptr_dataset->getitem(batch_indices[i]).getLabel(), 0);//handle the rest item
                batch_data = xt::concatenate(xt::xtuple(batch_data, sample_data), 0);
                batch_labels = xt::concatenate(xt::xtuple(batch_labels, sample_label), 0);
            }
            batches.emplace_back(batch_data, batch_labels); // logically work like push_back, but diffrent way
        }

        if (!drop_last && !batches.empty() && batches.back().getData().shape(0) < batch_size) {
            if (batches.size() > 1) {
                int second_last_idx = batches.size() - 2;
                int last_idx = batches.size() - 1;

                // Merge data
                xt::xarray<DType> merged_data = xt::concatenate(xt::xtuple(batches[second_last_idx].getData(), batches[last_idx].getData()));
                xt::xarray<LType> merged_labels = xt::concatenate(xt::xtuple(batches[second_last_idx].getLabel(), batches[last_idx].getLabel()));

                batches.pop_back();
                batches.pop_back();
                // Push in the merge batch
                batches.emplace_back(merged_data, merged_labels);

            
            }
        }
    }

    bool has_next()
    {
        return curr < batches.size();
    }

    Batch<DType, LType> next()
    {
        if (!has_next())
        {
            throw std::out_of_range("No more batches available.");
        }
        return batches[curr++];
    }

    virtual ~DataLoader() {}

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

        Batch<DType, LType> &operator*()
        {
            return dataloader->batches[pos];
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
