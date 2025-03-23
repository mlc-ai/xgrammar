#ifndef XGRAMMAR_SUPPORT_CONTAINER_H_
#define XGRAMMAR_SUPPORT_CONTAINER_H_
#include <vector>

namespace xgrammar {

template <typename Value>
class list {
 private:
  struct Node {
    int prev;
    int next;
    Value value;
    Node() : prev(0), next(0), value() {}
  };

  struct iterator {
   public:
    iterator(int n, list& c) : node_(n), list_(&c) {}
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
    friend class list;
    Node& get_node() { return list_->node_pool_[node_]; }

    int node_;
    list* list_;
  };

 public:
  list(int reserved = 0) {
    node_pool_.reserve(reserved);
    free_list_.reserve(reserved);
    node_pool_.emplace_back();  // create a dummy node
  }

  int push_back(const Value& value) {
    int node = allocate();
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
    free_list_.push_back(node);
    return iterator(next, *this);
  }

  void clear() {
    node_pool_.clear();
    free_list_.clear();
    node_pool_.emplace_back();  // create a dummy node
  }

  auto begin() { return iterator(node_pool_[0].next, *this); }
  auto end() { return iterator(0, *this); }

 private:
  int allocate() {
    if (free_list_.empty()) {
      node_pool_.emplace_back();
      return int(node_pool_.size()) - 1;
    } else {
      int node_id = free_list_.back();
      free_list_.pop_back();
      return node_id;
    }
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

  std::vector<Node> node_pool_;
  std::vector<int> free_list_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_CONTAINER_H_
