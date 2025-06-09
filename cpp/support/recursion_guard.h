#ifndef XGRAMMAR_SUPPORT_RECURSION_GUARD_H_
#define XGRAMMAR_SUPPORT_RECURSION_GUARD_H_

#include <atomic>

#include "logging.h"

namespace xgrammar {

/*!
 * \brief Thread-safe recursion guard to prevent stack overflow
 *
 * This class provides a RAII-style guard that tracks recursion depth
 * and prevents excessive recursion that could lead to stack overflow.
 * It uses atomic operations for thread safety and supports configurable
 * maximum recursion depth.
 */
class RecursionGuard {
 public:
  /*!
   * \brief Constructor that increments recursion depth
   * \param current_recursion_depth Pointer to the current recursion depth counter
   * \throws Logs fatal error if max recursion depth is exceeded
   */
  explicit RecursionGuard(int* current_recursion_depth)
      : current_depth_ptr_(current_recursion_depth) {
    XGRAMMAR_DCHECK(current_depth_ptr_ != nullptr);

    int current_depth = ++(*current_depth_ptr_);
    int max_depth = max_recursion_depth_.load(std::memory_order_relaxed);

    if (current_depth > max_depth) {
      XGRAMMAR_LOG(FATAL) << "RecursionGuard: Maximum recursion depth exceeded. "
                          << "Current depth: " << current_depth << ", Max allowed: " << max_depth;
    }
  }

  /*!
   * \brief Reset the recursion depth to 0
   * \param current_recursion_depth Pointer to the current recursion depth counter
   */
  static void ResetRecursionDepth(int* current_recursion_depth) {
    XGRAMMAR_DCHECK(current_recursion_depth != nullptr);
    *current_recursion_depth = 0;
  }

  /*!
   * \brief Destructor that decrements recursion depth
   */
  ~RecursionGuard() {
    XGRAMMAR_DCHECK(current_depth_ptr_ != nullptr);
    --(*current_depth_ptr_);
  }

  /*!
   * \brief Get the maximum allowed recursion depth
   * \return Current maximum recursion depth limit
   */
  static int GetMaxRecursionDepth() { return max_recursion_depth_.load(std::memory_order_relaxed); }

  /*!
   * \brief Set the maximum allowed recursion depth
   * \param max_depth New maximum recursion depth limit (must be positive)
   */
  static void SetMaxRecursionDepth(int max_depth) {
    if (max_depth <= 0 || max_depth > kMaxReasonableDepth) {
      XGRAMMAR_LOG(FATAL
      ) << "RecursionGuard: Maximum recursion depth must be positive and less than "
        << kMaxReasonableDepth << ", got: " << max_depth;
    }
    max_recursion_depth_.store(max_depth, std::memory_order_relaxed);
  }

 private:
  /*!
   * \brief Get the maximum allowed recursion depth from the environment variable. Used to
   * initialize max_recursion_depth_.
   * \return Current maximum recursion depth limit
   */
  static int LoadMaxRecursionDepthFromEnv();

  /*!
   * \brief Pointer to the recursion depth counter
   */
  int* current_depth_ptr_;

  /*!
   * \brief Thread-safe global configuration
   */
  static std::atomic<int> max_recursion_depth_;

  /*!
   * \brief Environment variable name for the maximum recursion depth
   */
  inline constexpr static char kMaxRecursionDepthEnvVar[] = "XGRAMMAR_MAX_RECURSION_DEPTH";

  /*!
   * \brief Default maximum recursion depth
   */
  inline constexpr static int kDefaultMaxRecursionDepth = 10000;

  /*!
   * \brief Maximum reasonable recursion depth
   */
  inline constexpr static int kMaxReasonableDepth = 1000000;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_RECURSION_GUARD_H_
