/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/csr_array.h
 */
#ifndef XGRAMMAR_SUPPORT_CSR_ARRAY_H_
#define XGRAMMAR_SUPPORT_CSR_ARRAY_H_

#include <picojson.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "logging.h"
#include "utils.h"

namespace xgrammar {

// TODO(yixin): consider renaming to CompactVector

/*!
 * \brief This class implements a Compressed Sparse Row (CSR) array data structure. It stores
 * a 2D array in a compressed format, where each row can have a variable number of elements, and
 * all rows are stored contiguously in memory. The inserted row is immutable.
 *
 * \note Inserting new rows into the CSRArray will invalidate the existing Row objects.
 *
 * \tparam DataType The type of elements stored in the CSRArray.
 *
 * \details
 * The CSRArray stores elements of type DataType in a compressed format,
 * where each row can have a variable number of elements. It uses two vectors:
 * - data_: stores all elements contiguously
 * - indptr_: stores the starting index of each row in data_. Its last element is the size of data_
 *            representing the ending index.
 *
 * This structure allows efficient storage and access for sparse data.
 */
template <typename DataType = int32_t>
class CSRArray {
 public:
  /*! \brief Default constructor. */
  CSRArray() = default;

  /****************** Accessors ******************/

  /*! \brief Get the number of rows in the CSRArray. */
  int32_t Size() const { return static_cast<int32_t>(indptr_.size()) - 1; }

  friend std::size_t MemorySize(const CSRArray<DataType>& arr) {
    return MemorySize(arr.data_) + MemorySize(arr.indptr_);
  }

  /*!
   * \brief Struct representing a row in the CSRArray.
   */
  struct Row {
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
          << "Index " << i << " of the CSRArray Row is out of bound";
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

  /*!
   * \brief Access a row in the CSRArray.
   * \param i Index of the row to access.
   * \return Row struct representing the i-th row.
   */
  Row operator[](int32_t i) const;

  /****************** Modifiers ******************/

  /*!
   * \brief Insert a new row of data into the CSRArray.
   * \param data Pointer to the data to be inserted.
   * \param data_len Length of the data to be inserted.
   * \return The index of the newly inserted row.
   */
  int32_t Insert(const DataType* new_data, int32_t new_data_len);

  /*!
   * \brief Insert a new row of data into the CSRArray from a vector.
   * \param data Vector containing the data to be inserted.
   * \return The index of the newly inserted row.
   */
  int32_t Insert(const std::vector<DataType>& new_data);

  /*!
   * \brief Insert a new row of data into the CSRArray from a Row struct.
   * \param row The Row struct containing the data to be inserted.
   * \return The index of the newly inserted row.
   */
  int32_t Insert(const Row& row) { return Insert(row.data, row.data_len); }

  /*!
   * \brief Insert a new row of non-contiguous data into the CSRArray. This method inserts a
   * single element followed by a sequence of elements. This is useful in the GrammarExpr data
   * structure.
   * \param data_1 The first element to be inserted.
   * \param data_2 Pointer to the remaining data to be inserted.
   * \param data_2_len Length of the remaining data to be inserted.
   * \return The index of the newly inserted row.
   */
  int32_t InsertNonContiguous(DataType data_1, const DataType* data_2, int32_t data_2_len);

  /****************** Internal Accessors ******************/

  /*! \brief Get a pointer to the underlying data array. */
  const DataType* data() const { return data_.data(); }
  /*! \brief Get a pointer to the underlying index pointer array. */
  const int32_t* indptr() const { return indptr_.data(); }

  /****************** Serialization ******************/

  /*!
   * \brief Serialize the CSRArray to a JSON string.
   * \return A JSON value representation of the CSRArray.
   */
  picojson::value Serialize() const;

  /*!
   * \brief Deserialize a JSON string to create a CSRArray.
   * \param v The JSON value to deserialize.
   * \return A new CSRArray object created from the deserialized data.
   * \throws xgrammar::InternalError if the JSON parsing fails or if the required fields are
   * missing.
   */
  static CSRArray Deserialize(const picojson::value& v);

  friend std::ostream& operator<<(std::ostream& os, const CSRArray& csr_array) {
    os << "CSRArray([";
    for (auto i = 0; i < csr_array.Size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << csr_array[i];
    }
    os << "])";
    return os;
  }

 private:
  /*! \brief Vector storing all elements contiguously. */
  std::vector<DataType> data_;
  /*! \brief Vector storing the starting index of each row in data_. */
  std::vector<int32_t> indptr_{0};
};

template <typename DataType>
inline typename CSRArray<DataType>::Row CSRArray<DataType>::operator[](int32_t i) const {
  XGRAMMAR_DCHECK(i >= 0 && i < Size()) << "CSRArray index " << i << " is out of bound";
  int32_t start = indptr_[i];
  int32_t end = indptr_[i + 1];
  return {data_.data() + start, end - start};
}

template <typename DataType>
inline int32_t CSRArray<DataType>::Insert(const DataType* new_data, int32_t new_data_len) {
  // TODO(yixin): whether to add a additional data_len
  // If the new data is already in the CSRArray, we need to copy it to the new memory location.
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
inline int32_t CSRArray<DataType>::Insert(const std::vector<DataType>& new_data) {
  data_.insert(data_.end(), new_data.begin(), new_data.end());
  indptr_.push_back(static_cast<int32_t>(data_.size()));
  return static_cast<int32_t>(indptr_.size()) - 2;
}

template <typename DataType>
inline int32_t CSRArray<DataType>::InsertNonContiguous(
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
inline picojson::value CSRArray<DataType>::Serialize() const {
  // Serialize data_
  picojson::array data_json;
  for (const auto& item : data_) {
    data_json.push_back(picojson::value(static_cast<int64_t>(item)));
  }

  // Serialize indptr_
  picojson::array indptr_json;
  for (const auto& item : indptr_) {
    indptr_json.push_back(picojson::value(static_cast<int64_t>(item)));
  }

  // Serialize the object
  picojson::object obj;
  obj["data"] = picojson::value(data_json);
  obj["indptr"] = picojson::value(indptr_json);

  return picojson::value(obj);
}

template <typename DataType>
inline CSRArray<DataType> CSRArray<DataType>::Deserialize(const picojson::value& v) {
  XGRAMMAR_CHECK(v.is<picojson::object>())
      << "Failed to deserialize CSRArray: expected a JSON object";

  picojson::object obj = v.get<picojson::object>();
  XGRAMMAR_CHECK(obj.find("data") != obj.end() && obj["data"].is<picojson::array>())
      << "Failed to parse data in CSRArray";
  XGRAMMAR_CHECK(obj.find("indptr") != obj.end() && obj["indptr"].is<picojson::array>())
      << "Failed to parse indptr in CSRArray";

  CSRArray csr_array;

  // Deserialize data_
  csr_array.data_.clear();
  for (const auto& item : obj["data"].get<picojson::array>()) {
    csr_array.data_.push_back(static_cast<DataType>(item.get<int64_t>()));
  }

  // Deserialize indptr_
  csr_array.indptr_.clear();
  for (const auto& item : obj["indptr"].get<picojson::array>()) {
    csr_array.indptr_.push_back(static_cast<int32_t>(item.get<int64_t>()));
  }

  return csr_array;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_CSR_ARRAY_H_
