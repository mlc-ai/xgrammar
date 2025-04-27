/*!
 *  Copyright (c) 2023 by Contributors
 * \file xgrammar/fsm.h
 */
#ifndef XGRAMMAR_FSM_H_
#define XGRAMMAR_FSM_H_

#include <picojson.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "support/csr_array.h"
#include "support/utils.h"

namespace xgrammar {

class CompactFSM;

/*!
 * \brief A finite state machine (FSM) implementation that supports character ranges on transitions.
 */
class FSM {
 public:
  /*! \brief Constructs an FSM with the specified number of nodes. */
  FSM(int num_nodes = 0) : edges_(num_nodes) {}

  /********************** Accessors **********************/

  inline static constexpr int NO_TRANSITION = -1;

  /*!
   * \brief Transitions from a given state based on an input character.
   * \param from The source state to transition from.
   * \param character The input character.
   * \return The target state if a valid transition exists, -1 otherwise.
   */
  int Transition(int from, int16_t character) const {
    auto& edges = edges_[from];
    for (const auto& edge : edges) {
      if (edge.min_ch <= character && edge.max_ch >= character) {
        return edge.target;
      }
    }
    return NO_TRANSITION;
  }

  /*! \brief Returns the start node of the FSM. */
  int StartNode() const { return start_node_; }

  /*!
   * \brief Checks if a given node is an end/accepting state.
   * \param node The node to check.
   * \return True if the node is an end state, false otherwise.
   */
  bool IsEndNode(int node) const {
    return std::any_of(end_nodes_.begin(), end_nodes_.end(), [node](int end_node) {
      return end_node == node;
    });
  }

  /*! \brief Returns the total number of nodes in the FSM. */
  int NumNodes() const { return edges_.size(); }

  /********************** Modifiers **********************/

  /*!
   * \brief Adds a transition edge between states with a character range.
   * \param from The source state.
   * \param to The target state.
   * \param min_ch The minimum character in the range (inclusive).
   * \param max_ch The maximum character in the range (inclusive).
   */
  void AddEdge(int from, int to, int16_t min_ch, int16_t max_ch) {
    edges_[from].push_back({min_ch, max_ch, to});
  }

  /*!
   * \brief Adds a new node to the FSM.
   * \return The index of the newly added node.
   */
  int AddNode() {
    edges_.emplace_back();
    return edges_.size() - 1;
  }

  /*!
   * \brief Sets the start node of the FSM.
   * \param node The node to set as the start node.
   */
  void SetStartNode(int node) { start_node_ = node; }

  /*!
   * \brief Adds an end/accepting node to the FSM.
   * \param node The node to add as an end node.
   */
  void AddEndNode(int node) { end_nodes_.push_back(node); }

  /*! \brief Converts this FSM to a more compact representation. */
  CompactFSM ToCompact();

  friend std::ostream& operator<<(std::ostream& os, const FSM& fsm);

 private:
  /*! \brief Represents a transition edge in the FSM. */
  struct Edge {
    /*! \brief Minimum character in range (inclusive). */
    int16_t min_ch;
    /*! \brief Maximum character in range (inclusive). */
    int16_t max_ch;
    /*! \brief Target state. */
    int32_t target;
    explicit operator int64_t() const {
      auto min_ch_uint = static_cast<uint16_t>(min_ch);
      auto max_ch_uint = static_cast<uint16_t>(max_ch);
      auto target_uint = static_cast<uint32_t>(target);
      auto result_uint = (static_cast<uint64_t>(min_ch_uint) << 48) |
                         (static_cast<uint64_t>(max_ch_uint) << 32) |
                         (static_cast<uint64_t>(target_uint));
      return static_cast<int64_t>(result_uint);
    }
    Edge() = default;
    Edge(int16_t min_ch, int16_t max_ch, int32_t target)
        : min_ch(min_ch), max_ch(max_ch), target(target) {}
    explicit Edge(int64_t packed) {
      auto packed_uint = static_cast<uint64_t>(packed);
      min_ch = static_cast<int16_t>((packed_uint >> 48) & 0xFFFF);
      max_ch = static_cast<int16_t>((packed_uint >> 32) & 0xFFFF);
      target = static_cast<int32_t>(packed_uint & 0xFFFFFFFF);
    }
    bool operator==(const Edge& other) const {
      return min_ch == other.min_ch && max_ch == other.max_ch && target == other.target;
    }
  };

  /*! \brief Adjacency list of edges for each node. */
  std::vector<std::vector<Edge>> edges_;
  /*! \brief Start node index. */
  int start_node_ = -1;
  /*! \brief List of end/accepting nodes. */
  std::vector<int> end_nodes_;

  friend class CompactFSM;
};

/*!
 * \brief A memory-efficient version of FSM that uses CSR format for edge storage.
 */
class CompactFSM {
 public:
  /*! \brief Default constructor. */
  CompactFSM() = default;

  /********************** Accessors **********************/

  inline static constexpr int NO_TRANSITION = -1;

  /*!
   * \brief Transitions from a given state based on an input character.
   * \param from The source state to transition from.
   * \param character The input character.
   * \return The target state if a valid transition exists, -1 otherwise.
   */
  int Transition(int from, int16_t character) const {
    auto edges = edges_[from];
    // TODO(yixin): test correctness for both cases
    if (edges.size() <= 16) {
      for (const auto& edge : edges) {
        if (edge.min_ch > character) {
          return NO_TRANSITION;
        } else if (edge.max_ch >= character) {
          return edge.target;
        }
      }
      return NO_TRANSITION;
    } else {
      auto it = std::lower_bound(
          edges.begin(),
          edges.end(),
          character,
          [](const Edge& edge, int16_t character) { return edge.min_ch <= character; }
      );
      if (it != edges.end() && it->min_ch <= character) {
        return it->target;
      }
      return NO_TRANSITION;
    }
  }

  /*! \brief Returns the start node of the FSM. */
  int StartNode() const { return start_node_; }

  /*!
   * \brief Checks if a given node is an end/accepting state.
   * \param node The node to check.
   * \return True if the node is an end state, false otherwise.
   */
  bool IsEndNode(int node) const {
    return std::any_of(end_nodes_.begin(), end_nodes_.end(), [node](int end_node) {
      return end_node == node;
    });
  }

  /*! \brief Returns the total number of nodes in the FSM. */
  int NumNodes() const { return edges_.Size(); }

  friend std::ostream& operator<<(std::ostream& os, const CompactFSM& fsm);

  friend std::size_t MemorySize(const CompactFSM& self) {
    return MemorySize(self.edges_) + MemorySize(self.end_nodes_);
  }

  picojson::value SerializeToJSON() const {
    picojson::object obj;
    obj["edges"] = edges_.SerializeToJSON();
    obj["start_node"] = picojson::value(static_cast<int64_t>(start_node_));
    picojson::array end_nodes_json;
    for (const auto& end_node : end_nodes_) {
      end_nodes_json.push_back(picojson::value(static_cast<int64_t>(end_node)));
    }
    obj["end_nodes"] = picojson::value(std::move(end_nodes_json));
    return picojson::value(std::move(obj));
  }

  static CompactFSM DeserializeFromJSON(const picojson::value& json) {
    CompactFSM fsm;
    const auto& obj = json.get<picojson::object>();
    fsm.edges_ = CSRArray<Edge>::DeserializeFromJSON(obj.at("edges"));
    fsm.start_node_ = static_cast<int>(obj.at("start_node").get<int64_t>());
    const auto& end_nodes_json = obj.at("end_nodes").get<picojson::array>();
    for (const auto& end_node : end_nodes_json) {
      fsm.end_nodes_.push_back(static_cast<int>(end_node.get<int64_t>()));
    }
    return fsm;
  }

  bool operator==(const CompactFSM& other) const {
    auto ref = [](const CompactFSM& fsm) {
      return std::tie(fsm.start_node_, fsm.end_nodes_, fsm.edges_);
    };
    return ref(*this) == ref(other);
  }

 private:
  using Edge = FSM::Edge;

  /*! \brief Edges stored in CSR format. */
  CSRArray<Edge> edges_;
  /*! \brief Start node index. */
  int start_node_ = -1;
  /*! \brief List of end/accepting nodes. */
  std::vector<int> end_nodes_;

  friend class FSM;
};

inline std::ostream& operator<<(std::ostream& os, const FSM& fsm) {
  os << "FSM(num_nodes=" << fsm.NumNodes() << ", start=" << fsm.StartNode() << ", end=[";
  for (int i = 0; i < static_cast<int>(fsm.end_nodes_.size()); ++i) {
    os << fsm.end_nodes_[i];
    if (i < static_cast<int>(fsm.end_nodes_.size()) - 1) {
      os << ", ";
    }
  }
  os << "], edges=[\n";
  for (int i = 0; i < fsm.NumNodes(); ++i) {
    os << i << ": [";
    const auto& edges = fsm.edges_[i];
    for (int j = 0; j < static_cast<int>(fsm.edges_[i].size()); ++j) {
      const auto& edge = edges[j];
      if (edge.min_ch == edge.max_ch) {
        os << "(" << edge.min_ch << ")->" << edge.target;
      } else {
        os << "(" << edge.min_ch << ", " << edge.max_ch << ")->" << edge.target;
      }
      if (j < static_cast<int>(fsm.edges_[i].size()) - 1) {
        os << ", ";
      }
    }
    os << "]\n";
  }
  os << "])";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const CompactFSM& fsm) {
  os << "CompactFSM(num_nodes=" << fsm.NumNodes() << ", start=" << fsm.StartNode() << ", end=[";
  for (int i = 0; i < static_cast<int>(fsm.end_nodes_.size()); ++i) {
    os << fsm.end_nodes_[i];
    if (i < static_cast<int>(fsm.end_nodes_.size()) - 1) {
      os << ", ";
    }
  }
  os << "], edges=[\n";
  for (int i = 0; i < fsm.NumNodes(); ++i) {
    os << i << ": [";
    const auto& edges = fsm.edges_[i];
    for (int j = 0; j < static_cast<int>(fsm.edges_[i].size()); ++j) {
      const auto& edge = edges[j];
      if (edge.min_ch == edge.max_ch) {
        os << "(" << edge.min_ch << ")->" << edge.target;
      } else {
        os << "(" << edge.min_ch << ", " << edge.max_ch << ")->" << edge.target;
      }
      if (j < static_cast<int>(fsm.edges_[i].size()) - 1) {
        os << ", ";
      }
    }
    os << "]\n";
  }
  os << "])";
  return os;
}

/*!
 * \brief Converts an FSM to its compact representation.
 * \return A CompactFSM with the same transitions but more efficient memory usage.
 */
inline CompactFSM FSM::ToCompact() {
  CompactFSM compact_fsm;
  compact_fsm.start_node_ = start_node_;
  compact_fsm.end_nodes_ = end_nodes_;
  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    std::sort(edges_[i].begin(), edges_[i].end(), [](const Edge& a, const Edge& b) {
      return a.min_ch < b.min_ch;
    });
    compact_fsm.edges_.Insert(edges_[i]);
  }
  return compact_fsm;
}

inline FSM BuildTrie(
    const std::vector<std::string>& patterns, std::vector<int32_t>* end_nodes = nullptr
) {
  FSM fsm(1);
  fsm.SetStartNode(0);
  if (end_nodes) {
    end_nodes->clear();
  }
  for (const auto& pattern : patterns) {
    int current_node = 0;
    for (const auto& ch : pattern) {
      int16_t ch_int16 = static_cast<int16_t>(static_cast<uint8_t>(ch));
      int next_node = fsm.Transition(current_node, ch_int16);
      if (next_node == FSM::NO_TRANSITION) {
        next_node = fsm.AddNode();
        fsm.AddEdge(current_node, next_node, ch_int16, ch_int16);
      }
      current_node = next_node;
    }
    fsm.AddEndNode(current_node);
    if (end_nodes) {
      end_nodes->push_back(current_node);
    }
  }
  return fsm;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_H_
