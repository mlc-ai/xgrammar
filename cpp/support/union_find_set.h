/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/union_find_set.h
 */
#ifndef XGRAMMAR_SUPPORT_UNION_FIND_SET_H_
#define XGRAMMAR_SUPPORT_UNION_FIND_SET_H_
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace xgrammar {
template <typename T>
class UnionFindSet {
 private:
  std::unordered_map<T, T> parent;
  std::unordered_map<T, int> rank;

 public:
  UnionFindSet() = default;

  ~UnionFindSet() = default;

  /*!
    \brief Insert a new element into the union-find set.
    \param value The value to be inserted.
    \return true if the value was successfully inserted, false if it already
    exists.
  */
  bool Make(const T& value) {
    if (parent.find(value) != parent.end()) {
      return false;
    }
    parent[value] = value;
    rank[value] = 0;
    return true;
  }

  /*!
    \brief Union two elements in the union-find set.
    \param a The first element.
    \param b The second element.
    \return true if the union was successful, false if the elements are already
    in the same set.
  */
  bool Union(T a, T b) {
    std::queue<T> queue;
    while (parent[a] != a) {
      queue.push(a);
      a = parent[a];
    }
    while (!queue.empty()) {
      parent[queue.front()] = a;
      queue.pop();
    }
    while (parent[b] != b) {
      queue.push(b);
      b = parent[b];
    }
    while (!queue.empty()) {
      parent[queue.front()] = b;
      queue.pop();
    }
    if (a == b) {
      return false;
    }
    if (rank[a] < rank[b]) {
      parent[a] = b;
      rank[b]++;
    } else {
      parent[b] = a;
      rank[a]++;
    }
    return true;
  }

  /*!
    \brief Find the representative of the set containing the given element.
    \param value The element whose representative is to be found.
    \return The representative of the set containing the element.
  */
  T find(T value) {
    std::queue<T> queue;
    while (parent[value] != value) {
      queue.push(value);
      value = parent[value];
    }
    while (!queue.empty()) {
      parent[queue.front()] = value;
      queue.pop();
    }
    return value;
  }

  /*
    \brief Check if two elements are in the same set.
    \param a The first element.
    \param b The second element.
    \return true if the elements are in the same set, false otherwise.
  */
  bool SameSet(T a, T b) const { return find(a) == find(b); }

  /*!
    \brief Get all the equivalence classes in the union-find set.
    \return A vector of unordered sets, each representing an equivalence class.
  */
  std::vector<std::unordered_set<T>> GetAllSets() const {
    std::vector<std::unordered_set<T>> result;
    std::unordered_map<T, int> which_set;
    for (const auto& [key, value] : parent) {
      if (which_set.find(value) == which_set.end()) {
        which_set[value] = result.size();
        result.push_back(std::unordered_set<T>());
      }
      result[which_set[value]].insert(key);
    }
    return result;
  }

  void Clear() {
    parent.clear();
    rank.clear();
  }
};
}  // namespace xgrammar
#endif  // XGRAMMAR_SUPPORT_UNION_FIND_SET_H_
