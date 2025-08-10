/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/compact_2d_array.h
 */
#ifndef XGRAMMAR_SUPPORT_COMPACT_2D_ARRAY_H_
#define XGRAMMAR_SUPPORT_COMPACT_2D_ARRAY_H_

#include <picojson.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "logging.h"
#include "memory_size.h"
#include "reflection.h"

namespace xgrammar {

/*!
 * \brief This class implements a Compressed Sparse Row (CSR) array data structure. It stores
 * a 2D array in a compressed format, where each row can have a variable number of elements, and
 * all rows are stored contiguously in memory. The inserted row is immutable.
 *
 * \note Inserting new rows into the Compact2DArray will invalidate the existing Row objects.
 *
 * \tparam DataType The type of elements stored in the Compact2DArray.
 *
 * \details
 * The Compact2DArray stores elements of type DataType in a compressed format,
 * where each row can have a variable number of elements. It uses two vectors:
 * - data_: stores all elements contiguously
 * - indptr_: stores the starting index of each row in data_. Its last element is the size of data_
 *            representing the ending index.
 *
 * This structure allows efficient storage and access for sparse data.
 */
template <typename DataType = int32_t>
class Compact2DArray {
 public:
  /*!
   * \brief The struct representing a row in the Compact2DArray.
   */
  struct Row {
    /*! \brief The value type is DataType. */
    using value_type = DataType;

    /*! \brief Pointer to the data of the row. */
    const DataType* data;
    /*! \brief Length of the row data. */
    int32_t data_len;

    /*!
     * \brief Access an element in the row.
     * \param i Index of the element to access.
     * \return Reference to the element at index i.
     */
    const DataType& operator[](int32_t i) const {
      XGRAMMAR_DCHECK(i >= 0 && i < data_len)
          << "Index " << i << " of the Compact2DArray Row is out of bound";
      return data[i];
    }

    /*! \brief Get the beginning iterator of the row. */
    const DataType* begin() const { return data; }
    /*! \brief Get the end iterator of the row. */
    const DataType* end() const { return data + data_len; }
    /*! \brief Get the size of the row. */
    int32_t size() const { return data_len; }

    friend std::ostream& operator<<(std::ostream& os, const Row& row) {
      os << "[";
      for (auto i = 0; i < row.data_len; ++i) {
        if (i > 0) {
          os << ", ";
        }
        os << row[i];
      }
      os << "]";
      return os;
    }
  };

  /*! \brief The value type is Row. */
  using value_type = Row;

  /*! \brief Default constructor. */
  Compact2DArray() = default;

  /****************** Accessors ******************/

  /*! \brief Get the number of rows in the Compact2DArray. */
  int32_t size() const { return static_cast<int32_t>(indptr_.size()) - 1; }

  friend std::size_t MemorySize(const Compact2DArray<DataType>& arr) {
    return MemorySize(arr.data_) + MemorySize(arr.indptr_);
  }

  /*!
   * \brief Access a row in the Compact2DArray.
   * \param i Index of the row to access.
   * \return Row struct representing the i-th row.
   */
  Row operator[](int32_t i) const;

  /****************** Modifiers ******************/

  /*!
   * \brief Insert a new row of data into the Compact2DArray.
   * \param data Pointer to the data to be inserted.
   * \param data_len Length of the data to be inserted.
   * \return The index of the newly inserted row.
   */
  int32_t PushBack(const DataType* new_data, int32_t new_data_len);

  /*!
   * \brief Insert a new row of data into the Compact2DArray from a vector.
   * \param data Vector containing the data to be inserted.
   * \return The index of the newly inserted row.
   */
  int32_t PushBack(const std::vector<DataType>& new_data);

  /*!
   * \brief Insert a new row of data into the Compact2DArray from a Row struct.
   * \param row The Row struct containing the data to be inserted.
   * \return The index of the newly inserted row.
   */
  int32_t PushBack(const Row& row) { return PushBack(row.data, row.data_len); }

  /*!
   * \brief Push back a new element in the latest row.
   * \param new_data the element to be pushed.
   */
  void PushBackInLatestRow(const DataType& new_data) {
    XGRAMMAR_DCHECK(!indptr_.empty()) << "Cannot push back in an empty Compact2DArray";
    data_.push_back(new_data);
    indptr_.back()++;
  }

  Row Back() { return (*this)[size() - 1]; }

  /*!
   * \brief Insert a new row of non-contiguous data into the Compact2DArray. This method inserts a
   * single element followed by a sequence of elements. This is useful in the GrammarExpr data
   * structure.
   * \param data_1 The first element to be inserted.
   * \param data_2 Pointer to the remaining data to be inserted.
   * \param data_2_len Length of the remaining data to be inserted.
   * \return The index of the newly inserted row.
   */
  int32_t PushBackNonContiguous(DataType data_1, const DataType* data_2, int32_t data_2_len);

  /*!
   * \brief Pop back the last one or multiple rows of the Compact2DArray.
   * \param cnt The number of rows to be popped.
   */
  void PopBack(const int32_t& cnt) {
    indptr_.erase(indptr_.end() - cnt, indptr_.end());
    data_.erase(data_.begin() + indptr_.back(), data_.end());
    return;
  }

  /****************** Internal Accessors ******************/

  /*! \brief Get a pointer to the underlying data array. */
  const DataType* data() const { return data_.data(); }
  /*! \brief Get a pointer to the underlying index pointer array. */
  const int32_t* indptr() const { return indptr_.data(); }

  /****************** Printing ******************/

  friend std::ostream& operator<<(std::ostream& os, const Compact2DArray& compact_2d_array) {
    os << "Compact2DArray([";
    for (auto i = 0; i < compact_2d_array.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << compact_2d_array[i];
    }
    os << "])";
    return os;
  }

 private:
  /*! \brief Vector storing all elements contiguously. */
  std::vector<DataType> data_;
  /*! \brief Vector storing the starting index of each row in data_. */
  std::vector<int32_t> indptr_{0};
  friend struct member_trait<Compact2DArray<DataType>>;
};

template <typename DataType>
inline typename Compact2DArray<DataType>::Row Compact2DArray<DataType>::operator[](int32_t i
) const {
  XGRAMMAR_DCHECK(i >= 0 && i < size()) << "Compact2DArray index " << i << " is out of bound";
  int32_t start = indptr_[i];
  int32_t end = indptr_[i + 1];
  return {data_.data() + start, end - start};
}

template <typename DataType>
inline int32_t Compact2DArray<DataType>::PushBack(const DataType* new_data, int32_t new_data_len) {
  // TODO(yixin): whether to add a additional data_len
  // If the new data is already in the Compact2DArray, we need to copy it to the new memory
  // location.
  if (new_data >= data_.data() && new_data < data_.data() + data_.size()) {
    std::vector<DataType> new_data_copied(new_data, new_data + new_data_len);
    data_.insert(data_.end(), new_data_copied.begin(), new_data_copied.end());
  } else {
    data_.insert(data_.end(), new_data, new_data + new_data_len);
  }
  indptr_.push_back(static_cast<int32_t>(data_.size()));
  return static_cast<int32_t>(indptr_.size()) - 2;
}

template <typename DataType>
inline int32_t Compact2DArray<DataType>::PushBack(const std::vector<DataType>& new_data) {
  data_.insert(data_.end(), new_data.begin(), new_data.end());
  indptr_.push_back(static_cast<int32_t>(data_.size()));
  return static_cast<int32_t>(indptr_.size()) - 2;
}

template <typename DataType>
inline int32_t Compact2DArray<DataType>::PushBackNonContiguous(
    DataType data_1, const DataType* data_2, int32_t data_2_len
) {
  if (data_2 >= data_.data() && data_2 < data_.data() + data_.size()) {
    std::vector<DataType> new_data_copied(data_2, data_2 + data_2_len);
    data_.push_back(data_1);
    data_.insert(data_.end(), new_data_copied.begin(), new_data_copied.end());
  } else {
    data_.push_back(data_1);
    data_.insert(data_.end(), data_2, data_2 + data_2_len);
  }
  indptr_.push_back(static_cast<int32_t>(data_.size()));
  return static_cast<int32_t>(indptr_.size()) - 2;
}

template <typename DataType>
XGRAMMAR_MEMBER_TABLE_TEMPLATE(
    Compact2DArray<DataType>,
    "data_",
    &Compact2DArray<DataType>::data_,
    "indptr_",
    &Compact2DArray<DataType>::indptr_
);

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_COMPACT_2D_ARRAY_H_
