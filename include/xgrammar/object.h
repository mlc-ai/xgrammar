/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/object.h
 * \brief Utilities for creating objects.
 */

#ifndef XGRAMMAR_OBJECT_H_
#define XGRAMMAR_OBJECT_H_

#include <memory>
#include <utility>

namespace xgrammar {

#define XGRAMMAR_DEFINE_PIMPL_METHODS(TypeName)                                \
 public:                                                                       \
  class Impl;                                                                  \
  /* The default constructor constructs a null object. Note operating on a */  \
  /* null object will fail. */                                                 \
  explicit TypeName() : pimpl_(nullptr) {}                                     \
  /* Construct object with a shared pointer to impl. The object just stores */ \
  /* a pointer. */                                                             \
  explicit TypeName(std::shared_ptr<Impl> pimpl) : pimpl_(std::move(pimpl)) {} \
  TypeName(const TypeName& other) = default;                                   \
  TypeName(TypeName&& other) noexcept = default;                               \
  TypeName& operator=(const TypeName& other) = default;                        \
  TypeName& operator=(TypeName&& other) noexcept = default;                    \
  /* Access the impl pointer. Useful in implementation. */                     \
  Impl* operator->() { return pimpl_.get(); }                                  \
  const Impl* operator->() const { return pimpl_.get(); }                      \
                                                                               \
 private:                                                                      \
  std::shared_ptr<Impl> pimpl_

}  // namespace xgrammar

#endif  // XGRAMMAR_OBJECT_H_
