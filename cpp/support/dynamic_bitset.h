/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/dynamic_bitset.h
 * \brief The header for utilities used in grammar-guided generation.
 */
#ifndef XGRAMMAR_SUPPORT_DYNAMIC_BITSET_H_
#define XGRAMMAR_SUPPORT_DYNAMIC_BITSET_H_

#include <cstdint>
#include <cstring>
#include <vector>

#include "logging.h"

namespace xgrammar {

/*!
 * \brief A bitset whose length is specified at runtime. Note the size cannot be changed after
 * construction.
 * \details The buffer of the bitset is a uint32_t array. There are two uses for this class:
 * - When passing nullptr to data, it maintains an internal buffer for the bitset.
 * - When passing a pointer to a buffer with enough size, it uses the external buffer for the
 *   bitset.
 * \details Part of the implementation is adopted from Boost::dynamic_bitset.
 */
class DynamicBitset {
 public:
  /*!
   * \brief Calculate the minimal size of the uint32_t buffer for the bitset with the given size.
   * \param element_size The size of the bitset.
   * \return The minimal buffer size.
   */
  static int CalculateBufferSize(int element_size) { return (element_size + 31) / 32; }

  /*!
   * \brief Construct a empty bitset. This object should be assigned to a valid bitset before using.
   */
  DynamicBitset() : size_(0), buffer_size_(0), data_(nullptr), is_internal_(true) {}

  /*!
   * \brief Construct a bitset with the given size.
   * \param size The size of the bitset.
   * \param data The buffer for the bitset. If nullptr, the bitset will maintain an internal buffer.
   */
  DynamicBitset(int size, uint32_t* data = nullptr)
      : size_(size), buffer_size_(CalculateBufferSize(size)) {
    if (data == nullptr) {
      internal_buffer_.resize(buffer_size_, 0);
      data_ = internal_buffer_.data();
      is_internal_ = true;
    } else {
      data_ = data;
      is_internal_ = false;
    }
  }

  /*! \brief Copy assignment. */
  DynamicBitset& operator=(const DynamicBitset& other) {
    XGRAMMAR_DCHECK(is_internal_ || size_ >= other.size_)
        << "Expanding bitset size is not allowed when the "
           "memory of the bitset is externally managed";
    size_ = other.size_;
    buffer_size_ = other.buffer_size_;
    if (is_internal_) {
      internal_buffer_.reserve(buffer_size_);
      data_ = internal_buffer_.data();
    }
    if (data_ != other.data_) {
      std::memcpy(data_, other.data_, buffer_size_ * sizeof(uint32_t));
    }
    return *this;
  }

  /*! \brief Move assignment. */
  DynamicBitset& operator=(DynamicBitset&& other) {
    size_ = other.size_;
    buffer_size_ = other.buffer_size_;
    is_internal_ = other.is_internal_;
    if (is_internal_) {
      internal_buffer_ = std::move(other.internal_buffer_);
      data_ = internal_buffer_.data();
    } else {
      data_ = other.data_;
    }
    return *this;
  }

  /*! \brief Get the value of the bit at the given index. */
  bool operator[](int index) const {
    XGRAMMAR_DCHECK(data_ && index >= 0 && index < size_);
    return (data_[index / 32] >> (index % 32)) & 1;
  }

  /*! \brief Get the size of the bitset. */
  int Size() const { return size_; }

  /*! \brief Set the whole bitset to true. */
  void Set() {
    XGRAMMAR_DCHECK(data_);
    std::memset(data_, 0xFF, buffer_size_ * sizeof(uint32_t));
  }

  /*! \brief Set the bit at the given index to the given value. */
  void Set(int index, bool value = true) {
    XGRAMMAR_DCHECK(data_ && index >= 0 && index < size_);
    if (value) {
      data_[index / 32] |= 1 << (index % 32);
    } else {
      data_[index / 32] &= ~(1 << (index % 32));
    }
  }

  /*! \brief Set the whole bitset to false. */
  void Reset() {
    XGRAMMAR_DCHECK(data_);
    std::memset(data_, 0, buffer_size_ * sizeof(uint32_t));
  }

  /*! \brief Set the bit at the given index to false. */
  void Reset(int index) { Set(index, false); }

  /*! \brief Perform a bitwise OR operation between the current bitset and another bitset. */
  DynamicBitset& operator|=(const DynamicBitset& other) {
    XGRAMMAR_DCHECK(buffer_size_ <= other.buffer_size_);
    for (int i = 0; i < buffer_size_; ++i) {
      data_[i] |= other.data_[i];
    }
    return *this;
  }

  int FindFirstZero() const { return DoFindZeroFrom(0); }

  int FindNextZero(int pos) const {
    if (pos >= size_ - 1 || size_ == 0) return -1;
    ++pos;
    int blk = pos / BITS_PER_BLOCK;
    int ind = pos % BITS_PER_BLOCK;
    uint32_t fore = (~data_[blk]) >> ind;
    int result = fore ? pos + LowestBit(fore) : DoFindZeroFrom(blk + 1);
    return result < size_ ? result : -1;
  }

 private:
  static int LowestBit(uint32_t value) {
#ifdef __GNUC__
    return __builtin_ctz(value);
#else   // __GNUC__
    // From https://stackoverflow.com/a/757266
    static const int MultiplyDeBruijnBitPosition[32] = {0,  1,  28, 2,  29, 14, 24, 3,  30, 22, 20,
                                                        15, 25, 17, 4,  8,  31, 27, 13, 23, 21, 19,
                                                        16, 7,  26, 12, 18, 6,  11, 5,  10, 9};
    return MultiplyDeBruijnBitPosition[((uint32_t)((value & -value) * 0x077CB531U)) >> 27];
#endif  // __GNUC__
  }

  int DoFindZeroFrom(int first_block) const {
    int position = -1;
    for (int i = first_block; i < buffer_size_; ++i) {
      if (data_[i] != ~static_cast<uint32_t>(0)) {
        position = i;
        break;
      }
    }
    if (position == -1) return -1;
    return position * BITS_PER_BLOCK + LowestBit(~data_[position]);
  }

  static constexpr int BITS_PER_BLOCK = 32;
  // The size of the bitset.
  int size_;
  // The size of the buffer.
  int buffer_size_;
  // The buffer for the bitset.
  uint32_t* data_;
  // The internal buffer. It is empty if not needed.
  std::vector<uint32_t> internal_buffer_;
  // Whether the buffer is internally managed.
  bool is_internal_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_DYNAMIC_BITSET_H_
