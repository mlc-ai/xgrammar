#pragma once
#include <queue>
#include <unordered_map>
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
    \return true if the value was successfully inserted, false if it already exists.
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
    \return true if the union was successful, false if the elements are already in the same set.
  */
  bool Union(T a, T b) {
    std::queue<T> queue;
    while (parent.find(a) != a) {
      queue.push(a);
      a = parent[a];
    }
    while (!queue.empty()) {
      parent[queue.front()] = a;
      queue.pop();
    }
    while (parent.find(b) != b) {
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
  }

  /*!
    \brief Find the representative of the set containing the given element.
    \param value The element whose representative is to be found.
    \return The representative of the set containing the element.
  */
  T find(T value) {
    std::queue<T> queue;
    while (parent.find(value) != value) {
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
  bool SameSet(T a, T b) { return find(a) == find(b); }
};
}  // namespace xgrammar
