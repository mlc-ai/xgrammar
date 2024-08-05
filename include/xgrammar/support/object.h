/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/support/object.h
 * \brief Utilities for creating objects.
 */
#ifndef XGRAMMAR_SUPPORT_OBJECT_H_
#define XGRAMMAR_SUPPORT_OBJECT_H_

namespace xgrammar {

#define XGRAMMAR_DEFINE_PIMPL_METHODS(TypeName)                                \
 public:                                                                       \
  class Impl;                                                                  \
  explicit TypeName(std::shared_ptr<Impl> pimpl) : pimpl_(std::move(pimpl)) {} \
  TypeName(const TypeName& other) = default;                                   \
  TypeName(TypeName&& other) noexcept = default;                               \
  TypeName& operator=(const TypeName& other) = default;                        \
  TypeName& operator=(TypeName&& other) noexcept = default;                    \
  Impl* operator->() { return pimpl_.get(); }                                  \
  const Impl* operator->() const { return pimpl_.get(); }                      \
                                                                               \
 private:                                                                      \
  std::shared_ptr<Impl> pimpl_

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_OBJECT_H_
