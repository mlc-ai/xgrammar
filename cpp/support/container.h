#ifndef XGRAMMAR_SUPPORT_CONTAINER_H_
#define XGRAMMAR_SUPPORT_CONTAINER_H_
#include <vector>

#include "logging.h"

namespace xgrammar {

namespace details {

template <typename Node>
class NodePool {
 public:
  NodePool() = default;

  void reserve(int n) { node_pool_.reserve(n); }

  [[nodiscard]]
  int allocate() {
    if (free_list_.empty()) {
      int node = static_cast<int>(node_pool_.size());
      node_pool_.emplace_back();
      return node;
    } else {
      int node = free_list_.back();
      free_list_.pop_back();
      return node;
    }
  }

  void deallocate(int node) { free_list_.push_back(node); }

  void clear() {
    node_pool_.clear();
    free_list_.clear();
  }

  Node& operator[](int node) { return node_pool_[node]; }

 private:
  std::vector<Node> node_pool_;
  std::vector<int> free_list_;
};

}  // namespace details

template <typename Value>
class List {
 private:
  struct Node {
    int prev;
    int next;
    Value value;
    Node() : prev(0), next(0), value() {}
  };

 public:
  struct iterator {
   public:
    iterator(int n, List& c) : node_(n), list_(&c) {}
    iterator& operator++() {
      node_ = get_node().next;
      return *this;
    }
    iterator operator++(int) {
      iterator tmp = *this;
      ++*this;
      return tmp;
    }
    Value& operator*() { return get_node().value; }
    Value* operator->() { return &get_node().value; }
    bool operator==(const iterator& rhs) const {
      return node_ == rhs.node_;  // compare different container is UB
    }
    bool operator!=(const iterator& rhs) const {
      return node_ != rhs.node_;  // compare different container is UB
    }

   private:
    friend class List;
    Node& get_node() { return list_->node_pool_[node_]; }

    int node_;
    List* list_;
  };

  List(int reserved = 0) {
    node_pool_.reserve(reserved);
    init_guard();
  }

  int push_back(const Value& value) {
    int node = node_pool_.allocate();
    node_pool_[node].value = value;
    insert_before(node, 0);
    return node;
  }

  void move_back(int node) {
    unlink(node);
    insert_before(node, 0);
  }

  iterator erase(iterator it) {
    int node = it.node_;
    int next = node_pool_[node].next;
    unlink(node);
    node_pool_.deallocate(node);
    return iterator(next, *this);
  }

  void clear() {
    node_pool_.clear();
    init_guard();
  }

  iterator begin() { return iterator(node_pool_[0].next, *this); }
  iterator end() { return iterator(0, *this); }

 private:
  void init_guard() {
    int node_id = node_pool_.allocate();
    XGRAMMAR_DCHECK(node_id == 0) << "node 0 should be reserved as guard node";
  }

  void insert_before(int node, int next) {
    int prev = node_pool_[next].prev;
    node_pool_[node].prev = prev;
    node_pool_[node].next = next;
    node_pool_[prev].next = node;
    node_pool_[next].prev = node;
  }

  void unlink(int node) {
    int prev = node_pool_[node].prev;
    int next = node_pool_[node].next;
    node_pool_[prev].next = next;
    node_pool_[next].prev = prev;
  }

  details::NodePool<Node> node_pool_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_CONTAINER_H_
