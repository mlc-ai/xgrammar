/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm.cc
 */
#include "fsm.h"

#include <picojson.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "support/encoding.h"
#include "support/json_serializer.h"
#include "support/logging.h"
#include "support/reflection.h"
#include "support/union_find_set.h"
#include "support/utils.h"
#include "xgrammar/exception.h"

namespace xgrammar {

/****************** FSMImplBase ******************/

template <typename ContainerType>
class FSMImplBase {
  static_assert(
      std::is_same_v<ContainerType, std::vector<std::vector<FSMEdge>>> ||
          std::is_same_v<ContainerType, Compact2DArray<FSMEdge>>,
      "ContainerType must be std::vector<std::vector<FSMEdge>> or Compact2DArray<FSMEdge>"
  );

 public:
  /*! \brief Default constructor. */
  FSMImplBase() = default;

  FSMImplBase(const ContainerType& edges, std::vector<int32_t> edge_aux_data = {})
      : edges_(edges), edge_aux_data_(std::move(edge_aux_data)) {}

  FSMImplBase(ContainerType&& edges, std::vector<int32_t> edge_aux_data = {})
      : edges_(std::move(edges)), edge_aux_data_(std::move(edge_aux_data)) {}

  int NumStates() const { return edges_.size(); }

  std::string EdgesToString(std::optional<std::vector<int>> states = std::nullopt) const;

  const ContainerType& GetEdges() const { return edges_; }

  // For std::vector<std::vector<FSMEdge>>, return const std::vector<FSMEdge>& to avoid copying.
  // For Compact2DArray<FSMEdge>, return Compact2DArray<FSMEdge>::Row since it is just a simple
  // pointer.
  decltype(auto) GetEdges(int state) const { return edges_[state]; }

  void GetEpsilonClosure(std::unordered_set<int>* state_set) const;

  void GetPossibleRules(int state_num, std::unordered_set<int>* rules) const;

  void GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result) const;

  const std::vector<int32_t>& GetEdgeAuxData() const { return edge_aux_data_; }

  void SetEdgeAuxData(std::vector<int32_t> data) { edge_aux_data_ = std::move(data); }

  RepeatEdgeRef GetRepeatEdgeInfo(int32_t idx) const { return {edge_aux_data_.data() + idx}; }

  TokenEdgeRef GetTokenEdgeInfo(int32_t idx) const { return {edge_aux_data_.data() + idx}; }

  ExcludeTokenEdgeRef GetExcludeTokenEdgeInfo(int32_t idx) const {
    return {edge_aux_data_.data() + idx};
  }

 protected:
  ContainerType edges_;
  std::vector<int32_t> edge_aux_data_;
  friend struct member_trait<CompactFSM::Impl>;
};

template <typename ContainerType>
std::string FSMImplBase<ContainerType>::EdgesToString(std::optional<std::vector<int>> states
) const {
  std::string result = "[\n";
  auto f_print_one = [&, this](int i) {
    result += std::to_string(i) + ": [";
    const auto& edges = edges_[i];
    for (int j = 0; j < static_cast<int>(edges.size()); ++j) {
      const auto& edge = edges[j];
      if (edge.min >= 0 && edge.min != edge.max) {
        std::string char_min_str = EscapeString(static_cast<TCodepoint>(edge.min));
        std::string char_max_str = EscapeString(static_cast<TCodepoint>(edge.max));
        result += "[" + char_min_str + "-" + char_max_str + "]->" + std::to_string(edge.target);
      } else if (edge.min >= 0 && edge.min == edge.max) {
        std::string char_str = EscapeString(static_cast<TCodepoint>(edge.min));
        result += "'" + char_str + "'->" + std::to_string(edge.target);
      } else if (edge.min == FSMEdge::EdgeType::kRuleRef) {
        result += "Rule(" + std::to_string(edge.max) + ")->" + std::to_string(edge.target);
      } else if (edge.min == FSMEdge::EdgeType::kEpsilon) {
        result += "Eps->" + std::to_string(edge.target);
      } else if (edge.min == FSMEdge::EdgeType::kEOS) {
        result += "EOS->" + std::to_string(edge.target);
      } else if (edge.min == FSMEdge::EdgeType::kRepeatRef) {
        auto info = GetRepeatEdgeInfo(edge.max);
        result += "Repeat(rule=" + std::to_string(info.RuleId()) +
                  ", min=" + std::to_string(info.Lower()) +
                  ", max=" + std::to_string(info.Upper()) + ")->" + std::to_string(edge.target);
      } else if (edge.min == FSMEdge::EdgeType::kToken) {
        auto info = GetTokenEdgeInfo(edge.max);
        result += "Token(";
        for (int32_t k = 0; k < info.Count(); ++k) {
          if (k > 0) result += ", ";
          result += std::to_string(info.TokenIds()[k]);
        }
        result += ")->" + std::to_string(edge.target);
      } else if (edge.min == FSMEdge::EdgeType::kExcludeToken) {
        auto info = GetExcludeTokenEdgeInfo(edge.max);
        result += "ExcludeToken(";
        for (int32_t k = 0; k < info.Count(); ++k) {
          if (k > 0) result += ", ";
          result += std::to_string(info.TokenIds()[k]);
        }
        result += ")->" + std::to_string(edge.target);
      }
      if (j < static_cast<int>(edges.size()) - 1) {
        result += ", ";
      }
    }
    result += "]\n";
  };
  if (states.has_value()) {
    for (int i : states.value()) {
      f_print_one(i);
    }
  } else {
    for (int i = 0; i < int(NumStates()); ++i) {
      f_print_one(i);
    }
  }
  result += "]";
  return result;
}

template <typename ContainerType>
void FSMImplBase<ContainerType>::GetEpsilonClosure(std::unordered_set<int>* state_set) const {
  std::queue<int> queue;
  for (const auto& state : *state_set) {
    queue.push(state);
  }
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();
    for (const auto& edge : edges_[current]) {
      if (!edge.IsEpsilon()) {
        continue;
      }
      if (state_set->find(edge.target) != state_set->end()) {
        continue;
      }
      state_set->insert(edge.target);
      queue.push(edge.target);
    }
  }
}

template <typename ContainerType>
void FSMImplBase<ContainerType>::GetPossibleRules(int state, std::unordered_set<int>* rules) const {
  rules->clear();
  for (const auto& edge : edges_[state]) {
    if (edge.IsRuleRef()) {
      rules->insert(edge.GetRefRuleId());
    }
  }
}

template <typename ContainerType>
void FSMImplBase<ContainerType>::GetReachableStates(
    const std::vector<int>& from, std::unordered_set<int>* result
) const {
  result->clear();
  std::queue<int> queue;
  for (const auto& state : from) {
    queue.push(state);
    result->insert(state);
  }
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();
    for (const auto& edge : edges_[current]) {
      if (result->find(edge.target) != result->end()) {
        continue;
      }
      result->insert(edge.target);
      queue.push(edge.target);
    }
  }
}

/****************** FSM::Impl ******************/

class FSM::Impl : public FSMImplBase<std::vector<std::vector<FSMEdge>>> {
  using EdgeType = FSMEdge::EdgeType;

 public:
  Impl() = default;

  Impl(int num_states = 0) { edges_.resize(num_states); }

  using FSMImplBase<std::vector<std::vector<FSMEdge>>>::FSMImplBase;

  int GetNextState(int from, int value, EdgeType edge_type) const;

  using FSMImplBase<std::vector<std::vector<FSMEdge>>>::GetEdges;

  std::vector<std::vector<FSMEdge>>& GetEdges() { return edges_; }

  std::vector<FSMEdge>& GetEdges(int state) { return edges_[state]; }

  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      EdgeType edge_type,
      bool from_is_closure
  ) const;

  int AddState() {
    edges_.emplace_back();
    return edges_.size() - 1;
  }

  void AddEdge(int from, int to, int32_t min, int32_t max) {
    XGRAMMAR_DCHECK(from < static_cast<int>(edges_.size()));
    edges_[from].push_back({min, max, to});
  }

  void AddRuleEdge(int from, int to, int32_t rule_id) {
    AddEdge(from, to, FSMEdge::EdgeType::kRuleRef, rule_id);
  }

  void AddEpsilonEdge(int from, int to) { AddEdge(from, to, FSMEdge::EdgeType::kEpsilon, 0); }

  void AddEOSEdge(int from, int to) { AddEdge(from, to, FSMEdge::EdgeType::kEOS, 0); }

  void AddRepeatEdge(int from, int to, int32_t rule_id, int32_t lower, int32_t upper) {
    XGRAMMAR_DCHECK(edges_[from].empty())
        << "A state with a kRepeatRef edge must have no other outgoing edges.";
    XGRAMMAR_DCHECK(edge_aux_data_.size() <= INT32_MAX);
    int32_t aux_index = static_cast<int32_t>(edge_aux_data_.size());
    edge_aux_data_.reserve(edge_aux_data_.size() + 3);
    edge_aux_data_.emplace_back(rule_id);
    edge_aux_data_.emplace_back(lower);
    edge_aux_data_.emplace_back(upper);
    AddEdge(from, to, FSMEdge::EdgeType::kRepeatRef, aux_index);
  }

  void AddTokenEdge(int from, int to, const std::vector<int32_t>& token_ids) {
    XGRAMMAR_DCHECK(!token_ids.empty()) << "Token set must not be empty";
    XGRAMMAR_CHECK(edge_aux_data_.size() <= INT32_MAX)
        << "edge_aux_data_ overflow: too many auxiliary data entries";
    int32_t aux_index = static_cast<int32_t>(edge_aux_data_.size());
    edge_aux_data_.push_back(static_cast<int32_t>(token_ids.size()));
    for (int32_t id : token_ids) {
      edge_aux_data_.push_back(id);
    }
    edges_[from].push_back(FSMEdge(FSMEdge::EdgeType::kToken, aux_index, to));
  }

  void AddExcludeTokenEdge(int from, int to, const std::vector<int32_t>& token_ids) {
    XGRAMMAR_DCHECK(!token_ids.empty()) << "Token exclude set must not be empty";
    XGRAMMAR_CHECK(edge_aux_data_.size() <= INT32_MAX)
        << "edge_aux_data_ overflow: too many auxiliary data entries";
    int32_t aux_index = static_cast<int32_t>(edge_aux_data_.size());
    edge_aux_data_.push_back(static_cast<int32_t>(token_ids.size()));
    for (int32_t id : token_ids) {
      edge_aux_data_.push_back(id);
    }
    edges_[from].push_back(FSMEdge(FSMEdge::EdgeType::kExcludeToken, aux_index, to));
  }

  void AddFSM(const FSM& fsm, std::vector<int>* state_mapping);

  FSM RebuildWithMapping(const std::vector<int>& state_mapping, int new_num_states) const;

  void SortEdges();

  CompactFSM ToCompact();

  friend class FSMWithStartEnd;
};

int FSM::Impl::GetNextState(int from, int value, EdgeType edge_type) const {
  XGRAMMAR_DCHECK(edge_type != EdgeType::kEpsilon)
      << "Should not call GetNextState with edge type kEpsilon.";
  if (edge_type == EdgeType::kCharRange) {
    for (const auto& edge : edges_[from]) {
      if (edge.min >= EdgeType::kCharRange && edge.min <= value && edge.max >= value) {
        return edge.target;
      }
    }
    return FSM::kNoNextState;
  } else if (edge_type == EdgeType::kRuleRef) {
    for (const auto& edge : edges_[from]) {
      if (edge.min == EdgeType::kRuleRef && edge.GetRefRuleId() == value) {
        return edge.target;
      }
    }
    return FSM::kNoNextState;
  } else if (edge_type == EdgeType::kEOS) {
    for (const auto& edge : edges_[from]) {
      if (edge.min == EdgeType::kEOS) {
        return edge.target;
      }
    }
    return FSM::kNoNextState;
  } else if (edge_type == EdgeType::kRepeatRef) {
    // By invariant, a state with kRepeatRef has exactly one outgoing edge.
    XGRAMMAR_DCHECK(edges_[from].size() == 1 && edges_[from][0].IsRepeatRef());
    return edges_[from][0].target;
  } else {
    XGRAMMAR_DCHECK(false) << "Invalid edge type: " << static_cast<int>(edge_type);
  }
  XGRAMMAR_UNREACHABLE();
}

void FSM::Impl::Advance(
    const std::unordered_set<int>& from,
    int value,
    std::unordered_set<int>* result,
    EdgeType edge_type,
    bool from_is_closure
) const {
  XGRAMMAR_DCHECK(edge_type != EdgeType::kEpsilon)
      << "Should not call Advance with edge type kEpsilon.";

  const std::unordered_set<int>* start_closure;
  std::unordered_set<int> start_closure_tmp;

  if (from_is_closure) {
    start_closure = &from;
  } else {
    start_closure_tmp.insert(from.begin(), from.end());
    GetEpsilonClosure(&start_closure_tmp);
    start_closure = &start_closure_tmp;
  }

  result->clear();

  if (edge_type == EdgeType::kCharRange) {
    for (const auto& state : *start_closure) {
      for (const auto& edge : edges_[state]) {
        if (edge.IsCharRange() && edge.min <= value && edge.max >= value) {
          result->insert(edge.target);
        }
      }
    }
  } else if (edge_type == EdgeType::kRuleRef) {
    for (const auto& state : *start_closure) {
      for (const auto& edge : edges_[state]) {
        if (edge.IsRuleRef() && edge.GetRefRuleId() == value) {
          result->insert(edge.target);
        }
      }
    }
  } else if (edge_type == EdgeType::kEOS) {
    for (const auto& state : *start_closure) {
      for (const auto& edge : edges_[state]) {
        if (edge.IsEOS()) {
          result->insert(edge.target);
        }
      }
    }
  } else if (edge_type == EdgeType::kRepeatRef) {
    // By invariant, a state with kRepeatRef has exactly one outgoing edge.
    for (const auto& state : *start_closure) {
      if (!edges_[state].empty() && edges_[state][0].IsRepeatRef()) {
        result->insert(edges_[state][0].target);
      }
    }
  } else {
    XGRAMMAR_DCHECK(false) << "Invalid edge type: " << static_cast<int>(edge_type);
  }

  // Get the epsilon closure of the result.
  GetEpsilonClosure(result);
}

void FSM::Impl::AddFSM(const FSM& fsm, std::vector<int>* state_mapping) {
  int old_num_states = NumStates();
  int32_t aux_offset = static_cast<int32_t>(edge_aux_data_.size());

  const auto& other_aux = fsm.GetEdgeAuxData();
  edge_aux_data_.insert(edge_aux_data_.end(), other_aux.begin(), other_aux.end());

  if (state_mapping != nullptr) {
    state_mapping->clear();
    state_mapping->reserve(fsm.NumStates());
    for (int i = 0; i < fsm.NumStates(); ++i) {
      state_mapping->push_back(i + old_num_states);
    }
  }

  edges_.resize(edges_.size() + fsm.NumStates());

  for (int i = 0; i < fsm.NumStates(); ++i) {
    for (const auto& edge : fsm.GetEdges()[i]) {
      int32_t max_val = edge.max;
      if (edge.IsAuxEdge() && aux_offset > 0) {
        max_val = static_cast<int32_t>(edge.max + aux_offset);
      }
      AddEdge(i + old_num_states, edge.target + old_num_states, edge.min, max_val);
    }
  }
}

FSM FSM::Impl::RebuildWithMapping(const std::vector<int>& state_mapping, int new_num_states) const {
  std::vector<std::vector<FSMEdge>> new_edges(new_num_states);
  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    for (const auto& edge : edges_[i]) {
      if (edge.IsEpsilon() && state_mapping[i] == state_mapping[edge.target]) {
        continue;  // Skip self-loops for epsilon edges.
      }
      new_edges[state_mapping[i]].emplace_back(edge.min, edge.max, state_mapping[edge.target]);
    }
  }
  // aux_indices remain stable since only state ids are remapped
  for (int i = 0; i < new_num_states; ++i) {
    std::sort(new_edges[i].begin(), new_edges[i].end());
    const auto& end_iter = std::unique(new_edges[i].begin(), new_edges[i].end());
    new_edges[i].erase(end_iter, new_edges[i].end());
  }
  return FSM(std::move(new_edges), std::vector<int32_t>(edge_aux_data_));
}

void FSM::Impl::SortEdges() {
  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    std::sort(edges_[i].begin(), edges_[i].end());
  }
}

CompactFSM FSM::Impl::ToCompact() {
  SortEdges();
  Compact2DArray<FSMEdge> edges;
  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    edges.PushBack(edges_[i]);
  }
  return CompactFSM(std::move(edges), std::move(edge_aux_data_));
}

/****************** FSM ******************/

FSM::FSM(int num_states) : pimpl_(std::make_shared<Impl>(num_states)) {}

FSM::FSM(const std::vector<std::vector<FSMEdge>>& edges, std::vector<int32_t> edge_aux_data)
    : pimpl_(std::make_shared<Impl>(edges, std::move(edge_aux_data))) {}

FSM::FSM(std::vector<std::vector<FSMEdge>>&& edges, std::vector<int32_t> edge_aux_data)
    : pimpl_(std::make_shared<Impl>(std::move(edges), std::move(edge_aux_data))) {}

int FSM::NumStates() const { return pimpl_->NumStates(); }

int FSM::AddState() { return pimpl_->AddState(); }

void FSM::AddEdge(int from, int to, int32_t min, int32_t max) {
  pimpl_->AddEdge(from, to, min, max);
}

void FSM::AddEdge(int from, int to, FSMEdge::EdgeType type, int32_t value) {
  pimpl_->AddEdge(from, to, type, value);
}

void FSM::AddEpsilonEdge(int from, int to) { pimpl_->AddEpsilonEdge(from, to); }

void FSM::AddRuleEdge(int from, int to, int32_t rule_id) { pimpl_->AddRuleEdge(from, to, rule_id); }

void FSM::AddEOSEdge(int from, int to) { pimpl_->AddEOSEdge(from, to); }

void FSM::AddRepeatEdge(int from, int to, int32_t rule_id, int32_t lower, int32_t upper) {
  pimpl_->AddRepeatEdge(from, to, rule_id, lower, upper);
}

void FSM::AddTokenEdge(int from, int to, const std::vector<int32_t>& token_ids) {
  pimpl_->AddTokenEdge(from, to, token_ids);
}

void FSM::AddExcludeTokenEdge(int from, int to, const std::vector<int32_t>& token_ids) {
  pimpl_->AddExcludeTokenEdge(from, to, token_ids);
}

const std::vector<int32_t>& FSM::GetEdgeAuxData() const { return pimpl_->GetEdgeAuxData(); }

void FSM::SetEdgeAuxData(std::vector<int32_t> data) { pimpl_->SetEdgeAuxData(std::move(data)); }

RepeatEdgeRef FSM::GetRepeatEdgeInfo(int32_t idx) const { return pimpl_->GetRepeatEdgeInfo(idx); }

TokenEdgeRef FSM::GetTokenEdgeInfo(int32_t idx) const { return pimpl_->GetTokenEdgeInfo(idx); }

ExcludeTokenEdgeRef FSM::GetExcludeTokenEdgeInfo(int32_t idx) const {
  return pimpl_->GetExcludeTokenEdgeInfo(idx);
}

void FSM::AddFSM(const FSM& fsm, std::vector<int>* state_mapping) {
  pimpl_->AddFSM(fsm, state_mapping);
}

std::string FSM::EdgesToString(std::optional<std::vector<int>> states) const {
  return pimpl_->EdgesToString(states);
}

const std::vector<FSMEdge>& FSM::GetEdges(int state) const { return pimpl_->GetEdges(state); }

std::vector<std::vector<FSMEdge>>& FSM::GetEdges() { return pimpl_->GetEdges(); }

const std::vector<std::vector<FSMEdge>>& FSM::GetEdges() const { return pimpl_->GetEdges(); }

std::vector<FSMEdge>& FSM::GetEdges(int state) { return pimpl_->GetEdges(state); }

FSM FSM::Copy() const { return FSM(std::make_shared<Impl>(*pimpl_)); }

int FSM::GetNextState(int from, int value, FSMEdge::EdgeType edge_type) const {
  return pimpl_->GetNextState(from, value, edge_type);
}

void FSM::Advance(
    const std::unordered_set<int>& from,
    int value,
    std::unordered_set<int>* result,
    FSMEdge::EdgeType edge_type,
    bool from_is_closure
) const {
  pimpl_->Advance(from, value, result, edge_type, from_is_closure);
}

void FSM::GetPossibleRules(int state, std::unordered_set<int>* rules) const {
  pimpl_->GetPossibleRules(state, rules);
}

void FSM::GetEpsilonClosure(std::unordered_set<int>* state_set) const {
  pimpl_->GetEpsilonClosure(state_set);
}

void FSM::GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result) const {
  pimpl_->GetReachableStates(from, result);
}

FSM FSM::RebuildWithMapping(const std::vector<int>& state_mapping, int new_num_states) const {
  return pimpl_->RebuildWithMapping(state_mapping, new_num_states);
}

void FSM::SortEdges() { pimpl_->SortEdges(); }

CompactFSM FSM::ToCompact() { return pimpl_->ToCompact(); }

/****************** CompactFSM::Impl ******************/

class CompactFSM::Impl : public FSMImplBase<Compact2DArray<FSMEdge>> {
  using EdgeType = FSMEdge::EdgeType;

 public:
  Impl() = default;

  Impl(const Compact2DArray<FSMEdge>& edges, std::vector<int32_t> edge_aux_data = {})
      : FSMImplBase<Compact2DArray<FSMEdge>>(edges, std::move(edge_aux_data)),
        edge_num_(ComputeEdgeNum(edges_)) {}

  Impl(Compact2DArray<FSMEdge>&& edges, std::vector<int32_t> edge_aux_data = {})
      : FSMImplBase<Compact2DArray<FSMEdge>>(std::move(edges), std::move(edge_aux_data)),
        edge_num_(ComputeEdgeNum(edges_)) {}

  void GetNextStates(int from, int value, EdgeType edge_type, std::vector<int>* target) const;

  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      FSMEdge::EdgeType edge_type,
      bool from_is_closure
  ) const;

  FSM ToFSM() const;

  size_t GetNumEdges() const { return edge_num_; }

  size_t edge_num_ = 0;

  friend std::size_t MemorySize(const Impl& impl) {
    return MemorySize(impl.edges_) + MemorySize(impl.edge_aux_data_) + sizeof(impl.edge_num_);
  }

 private:
  static size_t ComputeEdgeNum(const Compact2DArray<FSMEdge>& edges) {
    size_t edge_num = 0;
    for (int i = 0; i < edges.size(); ++i) {
      edge_num += edges[i].size();
    }
    return edge_num;
  }
};

XGRAMMAR_MEMBER_TABLE(
    CompactFSM::Impl,
    "edges",
    &CompactFSM::Impl::edges_,
    "edge_aux_data",
    &CompactFSM::Impl::edge_aux_data_,
    "edge_num",
    &CompactFSM::Impl::edge_num_
);

void CompactFSM::Impl::GetNextStates(
    int from, int value, EdgeType edge_type, std::vector<int>* targets
) const {
  targets->clear();
  XGRAMMAR_DCHECK(edge_type != EdgeType::kEpsilon)
      << "Should not call GetNextState with edge type kEpsilon.";
  if (edge_type == EdgeType::kCharRange) {
    for (const auto& edge : edges_[from]) {
      if (edge.min < EdgeType::kCharRange) {
        continue;
      } else if (edge.min > value) {
        break;
      } else if (edge.max >= value) {
        targets->push_back(edge.target);
      }
    }
  } else if (edge_type == EdgeType::kRuleRef) {
    for (const auto& edge : edges_[from]) {
      if (edge.min < EdgeType::kRuleRef) {
        continue;
      } else if (edge.min > EdgeType::kRuleRef) {
        break;
      } else if (edge.GetRefRuleId() == value) {
        targets->push_back(edge.target);
      }
    }
  } else if (edge_type == EdgeType::kEOS) {
    for (const auto& edge : edges_[from]) {
      if (edge.min < EdgeType::kEOS) {
        continue;
      } else if (edge.min > EdgeType::kEOS) {
        break;
      } else if (edge.max >= EdgeType::kEOS) {
        targets->push_back(edge.target);
      }
    }
  } else if (edge_type == EdgeType::kRepeatRef) {
    // By invariant, a state with kRepeatRef has exactly one outgoing edge.
    for (const auto& edge : edges_[from]) {
      if (edge.IsRepeatRef()) {
        targets->push_back(edge.target);
        break;
      }
    }
  } else {
    XGRAMMAR_DCHECK(false) << "Invalid edge type: " << static_cast<int>(edge_type);
  }
}

void CompactFSM::Impl::Advance(
    const std::unordered_set<int>& from,
    int value,
    std::unordered_set<int>* result,
    FSMEdge::EdgeType edge_type,
    bool from_is_closure
) const {
  const std::unordered_set<int>* start_closure;
  std::unordered_set<int> start_closure_tmp;

  if (from_is_closure) {
    start_closure = &from;
  } else {
    start_closure_tmp.insert(from.begin(), from.end());
    GetEpsilonClosure(&start_closure_tmp);
    start_closure = &start_closure_tmp;
  }

  result->clear();

  if (edge_type == EdgeType::kCharRange) {
    for (const auto& state : *start_closure) {
      for (const auto& edge : edges_[state]) {
        if (edge.min < EdgeType::kCharRange) {
          continue;
        } else if (edge.min > value) {
          break;
        } else if (edge.max >= value) {
          result->insert(edge.target);
        }
      }
    }
  } else if (edge_type == EdgeType::kRuleRef) {
    for (const auto& state : *start_closure) {
      for (const auto& edge : edges_[state]) {
        if (edge.min < EdgeType::kRuleRef) {
          continue;
        } else if (edge.min > EdgeType::kRuleRef) {
          break;
        } else if (edge.GetRefRuleId() == value) {
          result->insert(edge.target);
        }
      }
    }
  } else if (edge_type == EdgeType::kEOS) {
    for (const auto& state : *start_closure) {
      for (const auto& edge : edges_[state]) {
        if (edge.min < EdgeType::kEOS) {
          continue;
        } else if (edge.min > EdgeType::kEOS) {
          break;
        } else if (edge.max >= EdgeType::kEOS) {
          result->insert(edge.target);
        }
      }
    }
  } else if (edge_type == EdgeType::kRepeatRef) {
    // By invariant, a state with kRepeatRef has exactly one outgoing edge.
    for (const auto& state : *start_closure) {
      for (const auto& edge : edges_[state]) {
        if (edge.IsRepeatRef()) {
          result->insert(edge.target);
          break;
        }
      }
    }
  } else {
    XGRAMMAR_DCHECK(false) << "Invalid edge type: " << static_cast<int>(edge_type);
  }

  // Get the epsilon closure of the result.
  GetEpsilonClosure(result);
}

FSM CompactFSM::Impl::ToFSM() const {
  std::vector<std::vector<FSMEdge>> edges(NumStates());
  for (int i = 0; i < edges_.size(); i++) {
    const auto& row = edges_[i];
    edges[i].insert(edges[i].end(), row.begin(), row.end());
  }
  return FSM(std::move(edges), std::vector<int32_t>(edge_aux_data_));
}

/****************** CompactFSM ******************/

CompactFSM::CompactFSM(const Compact2DArray<FSMEdge>& edges, std::vector<int32_t> edge_aux_data)
    : pimpl_(std::make_shared<Impl>(edges, std::move(edge_aux_data))) {}

CompactFSM::CompactFSM(Compact2DArray<FSMEdge>&& edges, std::vector<int32_t> edge_aux_data)
    : pimpl_(std::make_shared<Impl>(std::move(edges), std::move(edge_aux_data))) {}

int CompactFSM::NumStates() const { return pimpl_->NumStates(); }

const Compact2DArray<FSMEdge>& CompactFSM::GetEdges() const { return pimpl_->GetEdges(); }

Compact2DArray<FSMEdge>::Row CompactFSM::GetEdges(int state) const {
  return pimpl_->GetEdges(state);
}

std::string CompactFSM::EdgesToString(std::optional<std::vector<int>> states) const {
  return pimpl_->EdgesToString(states);
}

void CompactFSM::GetNextStates(
    int from, int value, FSMEdge::EdgeType edge_type, std::vector<int>* targets
) const {
  return pimpl_->GetNextStates(from, value, edge_type, targets);
}

void CompactFSM::Advance(
    const std::unordered_set<int>& from,
    int value,
    std::unordered_set<int>* result,
    FSMEdge::EdgeType edge_type,
    bool from_is_closure
) const {
  pimpl_->Advance(from, value, result, edge_type, from_is_closure);
}

void CompactFSM::GetPossibleRules(int state_num, std::unordered_set<int>* rules) const {
  pimpl_->GetPossibleRules(state_num, rules);
}

void CompactFSM::GetEpsilonClosure(std::unordered_set<int>* state_set) const {
  pimpl_->GetEpsilonClosure(state_set);
}

void CompactFSM::GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result)
    const {
  pimpl_->GetReachableStates(from, result);
}

size_t CompactFSM::GetNumEdges() const { return pimpl_->GetNumEdges(); }

FSM CompactFSM::ToFSM() const { return pimpl_->ToFSM(); }

const std::vector<int32_t>& CompactFSM::GetEdgeAuxData() const { return pimpl_->GetEdgeAuxData(); }

void CompactFSM::SetEdgeAuxData(std::vector<int32_t> data) {
  pimpl_->SetEdgeAuxData(std::move(data));
}

RepeatEdgeRef CompactFSM::GetRepeatEdgeInfo(int32_t idx) const {
  return pimpl_->GetRepeatEdgeInfo(idx);
}

TokenEdgeRef CompactFSM::GetTokenEdgeInfo(int32_t idx) const {
  return pimpl_->GetTokenEdgeInfo(idx);
}

ExcludeTokenEdgeRef CompactFSM::GetExcludeTokenEdgeInfo(int32_t idx) const {
  return pimpl_->GetExcludeTokenEdgeInfo(idx);
}

picojson::value SerializeJSONValue(const CompactFSM& value) {
  return detail::json_serializer::AutoSerializeJSONValuePImpl(value);
}

std::optional<SerializationError> DeserializeJSONValue(
    CompactFSM* result, const picojson::value& value, const std::string& type_name
) {
  return detail::json_serializer::AutoDeserializeJSONValuePImpl(result, value, type_name);
}

struct CompactFSMWithStartEndSerializeHelper {
  CompactFSM fsm;
  int start;
  bool is_dfa;
  std::vector<int32_t> end_index;
  size_t edge_num;

  CompactFSMWithStartEndSerializeHelper(const CompactFSMWithStartEnd& compact_fsm_with_se)
      : fsm(compact_fsm_with_se.fsm_),
        start(compact_fsm_with_se.start_),
        is_dfa(compact_fsm_with_se.is_dfa_),
        end_index(compact_fsm_with_se.ends_),
        edge_num(compact_fsm_with_se.edge_num_) {}

  CompactFSMWithStartEndSerializeHelper() = default;
};

XGRAMMAR_MEMBER_ARRAY(
    CompactFSMWithStartEndSerializeHelper,
    &CompactFSMWithStartEndSerializeHelper::fsm,
    &CompactFSMWithStartEndSerializeHelper::start,
    &CompactFSMWithStartEndSerializeHelper::end_index,
    &CompactFSMWithStartEndSerializeHelper::is_dfa,
    &CompactFSMWithStartEndSerializeHelper::edge_num
);

picojson::value SerializeJSONValue(const CompactFSMWithStartEnd& value) {
  return AutoSerializeJSONValue(CompactFSMWithStartEndSerializeHelper(value));
}
std::optional<SerializationError> DeserializeJSONValue(
    CompactFSMWithStartEnd* result, const picojson::value& value, const std::string& type_name
) {
  CompactFSMWithStartEndSerializeHelper tmp;
  auto err = AutoDeserializeJSONValue(&tmp, value, type_name);
  if (err.has_value()) {
    return err;
  }
  result->fsm_ = std::move(tmp.fsm);
  result->start_ = tmp.start;
  result->is_dfa_ = tmp.is_dfa;
  result->edge_num_ = tmp.edge_num;
  result->SetEndStates(std::move(tmp.end_index));
  return std::nullopt;
}

struct CompactFSMWithStartEndWithSizeSerializeHelper {
  CompactFSMWithStartEnd fsm;
  size_t edge_num;
  size_t node_num;

  CompactFSMWithStartEndWithSizeSerializeHelper(
      const CompactFSMWithStartEndWithSize& compact_fsm_with_size
  )
      : fsm(compact_fsm_with_size.fsm_),
        edge_num(compact_fsm_with_size.edge_num_),
        node_num(compact_fsm_with_size.node_num_) {}

  CompactFSMWithStartEndWithSizeSerializeHelper() = default;
};

XGRAMMAR_MEMBER_ARRAY(
    CompactFSMWithStartEndWithSizeSerializeHelper,
    &CompactFSMWithStartEndWithSizeSerializeHelper::fsm,
    &CompactFSMWithStartEndWithSizeSerializeHelper::edge_num,
    &CompactFSMWithStartEndWithSizeSerializeHelper::node_num
);

picojson::value SerializeJSONValue(const CompactFSMWithStartEndWithSize& value) {
  return AutoSerializeJSONValue(CompactFSMWithStartEndWithSizeSerializeHelper(value));
}

std::optional<SerializationError> DeserializeJSONValue(
    CompactFSMWithStartEndWithSize* result,
    const picojson::value& value,
    const std::string& type_name
) {
  CompactFSMWithStartEndWithSizeSerializeHelper tmp;
  auto err = AutoDeserializeJSONValue(&tmp, value, type_name);
  if (err.has_value()) {
    return err;
  }
  result->fsm_ = std::move(tmp.fsm);
  result->edge_num_ = tmp.edge_num;
  result->node_num_ = tmp.node_num;
  return std::nullopt;
}

/****************** FSMWithStartEnd ******************/

std::string FSMWithStartEnd::ToString() const {
  std::string result;
  result += "FSM(num_states=" + std::to_string(NumStates()) + ", start=" + std::to_string(start_) +
            ", end=[";

  std::unordered_set<int> reachable_states;
  GetReachableStates(&reachable_states);
  std::vector<int> reachable_states_vec(reachable_states.begin(), reachable_states.end());
  std::sort(reachable_states_vec.begin(), reachable_states_vec.end());

  bool first = true;
  for (auto end : ends_) {
    if (!first) {
      result += ", ";
    }
    first = false;
    result += std::to_string(end);
  }

  result += "], edges=" + fsm_.EdgesToString(reachable_states_vec) + ")";
  return result;
}

std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm) {
  os << fsm.ToString();
  return os;
}

FSMWithStartEnd FSMWithStartEnd::Copy() const {
  return FSMWithStartEnd(fsm_.Copy(), start_, ends_, is_dfa_);
}

FSMWithStartEnd FSMWithStartEnd::RebuildWithMapping(
    const std::vector<int>& state_mapping, int new_num_states
) const {
  FSM new_fsm = fsm_.RebuildWithMapping(state_mapping, new_num_states);
  auto new_start = state_mapping[start_];
  std::vector<int32_t> new_ends;
  new_ends.reserve(ends_.size());
  for (auto end : ends_) {
    new_ends.push_back(state_mapping[end]);
  }
  return FSMWithStartEnd(new_fsm, new_start, std::move(new_ends));
}

CompactFSMWithStartEnd FSMWithStartEnd::ToCompact() {
  return CompactFSMWithStartEnd(fsm_.ToCompact(), start_, ends_, is_dfa_);
}

FSMWithStartEndWithSize FSMWithStartEnd::AddToCompleteFSM(
    FSM* complete_fsm, std::vector<int>* state_mapping
) {
  XGRAMMAR_DCHECK(state_mapping != nullptr) << "state_mapping cannot be nullptr";
  complete_fsm->AddFSM(fsm_, state_mapping);
  int new_start = (*state_mapping)[start_];
  // Map the end states to the states in the complete FSM. The mapping is monotonic, so the
  // sorted invariant is preserved. The sparse representation is important here: this method is
  // called once per rule, and the ends of each returned view must not scale with the size of the
  // complete FSM, otherwise the total cost is O(num_rules * num_total_states).
  std::vector<int32_t> new_ends;
  new_ends.reserve(ends_.size());
  for (auto end : ends_) {
    new_ends.push_back((*state_mapping)[end]);
  }

  int num_edges = 0;
  for (int i = 0; i < fsm_.NumStates(); ++i) {
    num_edges += fsm_.GetEdges(i).size();
  }

  int num_nodes = fsm_.NumStates();

  auto fsm_with_se = FSMWithStartEnd(*complete_fsm, new_start, std::move(new_ends), is_dfa_);

  return FSMWithStartEndWithSize(fsm_with_se, num_edges, num_nodes);
}

FSMWithStartEnd FSMWithStartEnd::Star() const {
  FSM fsm = fsm_.Copy();
  auto new_start = fsm.AddState();
  for (auto end : ends_) {
    fsm.AddEpsilonEdge(end, new_start);
  }
  fsm.AddEpsilonEdge(new_start, start_);
  return FSMWithStartEnd(fsm, new_start, {new_start});
}

FSMWithStartEnd FSMWithStartEnd::Plus() const {
  FSM fsm = fsm_.Copy();
  for (auto end : ends_) {
    fsm.AddEpsilonEdge(end, start_);
  }
  return FSMWithStartEnd(fsm, start_, ends_);
}

FSMWithStartEnd FSMWithStartEnd::Optional() const {
  FSM fsm = fsm_.Copy();
  if (!ends_.empty()) {
    fsm.AddEpsilonEdge(start_, ends_.front());
  }
  return FSMWithStartEnd(fsm, start_, ends_);
}

Result<FSMWithStartEnd> FSMWithStartEnd::Not(int max_result_num_states) const {
  // Check if the FSM contains any rule references.
  if (!IsLeaf()) {
    XGRAMMAR_LOG(FATAL) << "Not operation is not supported for FSM with rule references.";
  }
  FSMWithStartEnd result;
  if (is_dfa_) {
    result = Copy();
  } else {
    Result<FSMWithStartEnd> dfa_result = ToDFA(max_result_num_states);
    if (dfa_result.IsErr()) {
      return dfa_result;
    }
    result = std::move(dfa_result).Unwrap();
  }
  // Reverse all the final states.
  std::vector<int32_t> new_final_states;
  for (int i = 0; i < result.NumStates(); ++i) {
    if (!result.IsEndState(i)) {
      new_final_states.push_back(i);  // Mark all states as final except the original final states.
    }
  }

  // Add a new final state that accepts all characters.
  int accept_all_new_state = result.AddState();
  new_final_states.push_back(accept_all_new_state);

  std::bitset<256> char_set;
  for (int i = 0; i < result.NumStates(); i++) {
    char_set.reset();
    // Collect all characters that are not accepted by the original FSM.
    for (const auto& edge : result.GetFsm().GetEdges(i)) {
      if (edge.IsCharRange()) {
        for (int j = edge.min; j <= edge.max; ++j) {
          char_set.set(j);
        }
      }
    }
    // Add edges for characters that are not accepted.
    for (int left_bound = 0; left_bound < 256; ++left_bound) {
      if (char_set[left_bound]) {
        continue;  // Skip characters that are accepted.
      }
      int right_bound = left_bound + 1;
      while (right_bound < 256 && !char_set[right_bound]) {
        ++right_bound;
      }
      result.GetFsm().AddEdge(i, accept_all_new_state, left_bound, right_bound - 1);
      left_bound = right_bound;
    }
  }

  result.SetEndStates(new_final_states);
  return ResultOk(result);
}

FSMWithStartEnd FSMWithStartEnd::Union(const std::vector<FSMWithStartEnd>& fsms) {
  // Put all the FSMs in parallel.
  // Allocate a new start state. Start state will be linked to the start states of all the FSMs.
  // The end states of the new FSM will be the union of the end states of all the FSMs.
  if (fsms.size() == 1) {
    return fsms[0];
  }
  XGRAMMAR_DCHECK(fsms.size() > 1) << "Union of 0 FSMs is not allowed.";

  FSM fsm(1);
  int start = 0;
  std::vector<int32_t> ends;

  std::vector<int> state_mapping;

  for (const auto& fsm_with_se : fsms) {
    fsm.AddFSM(fsm_with_se.GetFsm(), &state_mapping);
    fsm.AddEpsilonEdge(start, state_mapping[fsm_with_se.GetStart()]);
    for (auto end : fsm_with_se.GetEnds()) {
      ends.push_back(state_mapping[end]);
    }
  }

  return FSMWithStartEnd(fsm, start, std::move(ends));
}

FSMWithStartEnd FSMWithStartEnd::Concat(const std::vector<FSMWithStartEnd>& fsms) {
  // For each FSM, link the end states to the start state of the next FSM.
  // Set the start state of the first FSM as the start state of the result.
  // Set the end states of the last FSM as the end states of the result.
  if (fsms.size() == 1) {
    return fsms[0];
  }
  XGRAMMAR_DCHECK(fsms.size() > 1) << "Concatenation of 0 FSMs is not allowed.";

  FSM fsm;
  int start = 0;
  std::vector<int32_t> ends;

  std::vector<int> state_mapping;
  std::vector<int> previous_ends;

  for (int i = 0; i < static_cast<int>(fsms.size()); ++i) {
    fsm.AddFSM(fsms[i].GetFsm(), &state_mapping);
    if (i == 0) {
      start = state_mapping[fsms[i].GetStart()];
    } else {
      auto this_start = state_mapping[fsms[i].GetStart()];
      for (const auto& end : previous_ends) {
        fsm.AddEpsilonEdge(end, this_start);
      }
    }
    if (i == static_cast<int>(fsms.size()) - 1) {
      for (auto end : fsms[i].GetEnds()) {
        ends.push_back(state_mapping[end]);
      }
    } else {
      previous_ends.clear();
      previous_ends.reserve(fsms[i].GetEnds().size());
      for (auto end : fsms[i].GetEnds()) {
        previous_ends.push_back(state_mapping[end]);
      }
    }
  }

  return FSMWithStartEnd(fsm, start, std::move(ends));
}

Result<FSMWithStartEnd> FSMWithStartEnd::Intersect(
    const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs, int max_result_num_states
) {
  if (!lhs.IsLeaf() || !rhs.IsLeaf()) {
    return ResultErr("Intersect only support leaf fsm!");
  }
  auto lhs_dfa_raw = lhs.ToDFA();
  auto rhs_dfa_raw = rhs.ToDFA();

  if (lhs_dfa_raw.IsErr()) {
    return lhs_dfa_raw;
  }
  if (rhs_dfa_raw.IsErr()) {
    return rhs_dfa_raw;
  }

  auto lhs_dfa = std::move(lhs_dfa_raw).Unwrap();
  auto rhs_dfa = std::move(rhs_dfa_raw).Unwrap();
  // Initialize the result FSM.
  FSM result_fsm(0);
  FSMWithStartEnd result(result_fsm, 0, std::vector<int32_t>(), true);
  std::unordered_map<std::pair<int, int>, int> state_map;
  std::unordered_set<std::pair<int, int>> visited;
  std::queue<std::pair<int, int>> queue;
  queue.push({lhs_dfa.GetStart(), rhs_dfa.GetStart()});
  result.AddState();
  state_map[{lhs_dfa.GetStart(), rhs_dfa.GetStart()}] = 0;
  while (!queue.empty()) {
    auto [lhs_state, rhs_state] = std::move(queue.front());
    if (lhs_dfa.IsEndState(lhs_state) && rhs_dfa.IsEndState(rhs_state)) {
      result.AddEndState(state_map[{lhs_state, rhs_state}]);
    }
    queue.pop();
    for (const auto& lhs_edge : lhs_dfa.GetFsm().GetEdges(lhs_state)) {
      for (const auto& rhs_edge : rhs_dfa.GetFsm().GetEdges(rhs_state)) {
        XGRAMMAR_DCHECK(lhs_edge.IsCharRange() && rhs_edge.IsCharRange());
        // Check if the edges intersect.
        if (lhs_edge.min > rhs_edge.max || rhs_edge.min > lhs_edge.max) {
          continue;  // No intersection.
        }
        int min_value = std::max(lhs_edge.min, rhs_edge.min);
        int max_value = std::min(lhs_edge.max, rhs_edge.max);
        if (state_map.find(std::make_pair(lhs_edge.target, rhs_edge.target)) == state_map.end()) {
          state_map[{lhs_edge.target, rhs_edge.target}] = result.AddState();
          queue.push({lhs_edge.target, rhs_edge.target});
        }
        int target_state = state_map[{lhs_edge.target, rhs_edge.target}];
        result.GetFsm().AddEdge(
            state_map[{lhs_state, rhs_state}], target_state, min_value, max_value
        );
      }
    }
  }
  return ResultOk(std::move(result));
}

bool FSMWithStartEnd::IsDFA() {
  if (is_dfa_) {
    return true;
  }
  std::bitset<256> character_transitions;
  std::unordered_set<int> rule_transitions;
  for (const auto& edges : fsm_->GetEdges()) {
    character_transitions.reset();
    rule_transitions.clear();
    for (const auto& edge : edges) {
      if (edge.IsEpsilon()) {
        return false;  // Epsilon transitions are not allowed in DFA.
      }
      if (edge.IsCharRange()) {
        for (int i = edge.min; i <= edge.max; ++i) {
          if (character_transitions[i]) {
            return false;  // Duplicate character transition.
          }
          character_transitions.set(i);
        }
        continue;
      }
      if (edge.IsRuleRef()) {
        if (rule_transitions.find(edge.GetRefRuleId()) != rule_transitions.end()) {
          return false;  // Duplicate rule transition.
        }
        rule_transitions.insert(edge.GetRefRuleId());
      }
      // kRepeatRef: by invariant, a state with kRepeatRef has exactly one edge, always
      // deterministic.
    }
  }
  is_dfa_ = true;
  return true;
}

FSMWithStartEnd FSMWithStartEnd::SimplifyEpsilon(int max_num_states) const {
  if (is_dfa_) {
    return *this;
  }
  if (NumStates() > max_num_states) {
    return *this;
  }

  UnionFindSet<int> union_find_set;
  std::vector<int> in_degree(NumStates(), 0);
  std::vector<std::pair<int32_t, int32_t>> epsilon_edges;

  std::vector<bool> has_exclude_token(NumStates(), false);
  for (int i = 0; i < NumStates(); i++) {
    for (const auto& edge : fsm_->GetEdges(i)) {
      if (edge.IsExcludeToken()) {
        has_exclude_token[i] = true;
        break;
      }
    }
  }

  for (int i = 0; i < NumStates(); i++) {
    const auto& edges = fsm_->GetEdges(i);
    for (const auto& edge : edges) {
      in_degree[edge.target]++;
      if (edge.IsEpsilon()) {
        // a -- epsilon --> b, and a doesn't have other outward edges. Do not merge an
        // accepting a into a non-accepting b: the merged state would be accepting, so other
        // paths reaching b would be wrongly accepted. (If b is accepting, a effectively
        // accepts already via the epsilon edge, so merging is safe.)
        if (edges.size() == 1 && !has_exclude_token[i] && !has_exclude_token[edge.target] &&
            (!IsEndState(i) || IsEndState(edge.target))) {
          union_find_set.Add(i);
          union_find_set.Add(edge.target);
          union_find_set.Union(i, edge.target);
          in_degree[edge.target]--;  // Remove the inward edge since a and b are merged.
        } else {
          // Otherwise, we store it to check for the second merge rule.
          epsilon_edges.emplace_back(i, edge.target);
        }
      }
    }
  }

  // Build the equivalent graph.
  std::vector<int> equiv_node(NumStates());
  for (int i = 0; i < NumStates(); i++) {
    if (union_find_set.Count(i)) {
      equiv_node[i] = union_find_set.Find(i);
      if (equiv_node[i] == i) {
        continue;
      }
      in_degree[equiv_node[i]] += in_degree[i];
    } else {
      equiv_node[i] = i;
    }
  }

  // a --> epsilon --> b, and b doesn't have other inward edges.
  for (const auto& [from_raw, to_raw] : epsilon_edges) {
    const int& from = equiv_node[from_raw];
    const int& to = equiv_node[to_raw];
    if (in_degree[to] == 1 && equiv_node[GetStart()] != to && !has_exclude_token[from_raw] &&
        !has_exclude_token[to_raw]) {
      union_find_set.Add(from);
      union_find_set.Add(to);
      union_find_set.Union(from, to);
    }
  }

  // Merge the states.
  auto eq_classes = union_find_set.GetAllSets();
  if (eq_classes.empty()) {
    return *this;
  }

  std::vector<int> new_to_old(NumStates(), -1);
  for (size_t i = 0; i < eq_classes.size(); i++) {
    for (const auto& state : eq_classes[i]) {
      new_to_old[state] = i;
    }
  }

  int cnt = eq_classes.size();
  for (int i = 0; i < NumStates(); i++) {
    if (new_to_old[i] == -1) {
      new_to_old[i] = cnt;
      cnt++;
    }
  }
  return RebuildWithMapping(new_to_old, cnt);
}

FSMWithStartEnd FSMWithStartEnd::MergeEquivalentStates(int max_result_num_states) const {
  if (max_result_num_states < NumStates()) {
    return *this;
  }
  // No merge is possible with fewer than 4 states (need >=2 sources sharing a target,
  // or >=2 sinks sharing a source).
  if (NumStates() < 4) {
    return Copy();
  }
  FSMWithStartEnd result = Copy();
  result.GetFsm()->SortEdges();

  // Keep equivalence classes over stable state IDs, revisit only neighborhoods changed by the
  // previous round, and rebuild the physical FSM once after reaching a fixed point.
  struct EndpointEdge {
    int peer;  // source in incoming edges, target in outgoing edges
    int32_t min;
    int32_t max;

    bool operator<(const EndpointEdge& other) const {
      return std::tie(peer, min, max) < std::tie(other.peer, other.min, other.max);
    }

    bool operator==(const EndpointEdge& other) const {
      return std::tie(peer, min, max) == std::tie(other.peer, other.min, other.max);
    }
  };

  int original_num_states = 0;
  bool input_edges_are_canonical = false;
  Compact2DArray<EndpointEdge> original_incoming_edges;
  while (true) {
    original_num_states = result.NumStates();
    input_edges_are_canonical = true;
    bool may_have_non_leaf_merge = false;
    std::vector<int32_t> original_incoming_row_sizes(original_num_states, 0);
    for (int source = 0; source < original_num_states; ++source) {
      const auto& edges = result.GetFsm().GetEdges(source);
      for (int32_t edge_index = 0; edge_index < static_cast<int32_t>(edges.size()); ++edge_index) {
        const auto& edge = edges[edge_index];
        ++original_incoming_row_sizes[edge.target];
        if ((edge.IsEpsilon() && edge.target == source) ||
            (edge_index > 0 && edge == edges[edge_index - 1])) {
          input_edges_are_canonical = false;
        }
        if (edge_index > 0 && edges[edge_index - 1].min == edge.min &&
            edges[edge_index - 1].max == edge.max && edges[edge_index - 1].target != edge.target) {
          int previous_target = edges[edge_index - 1].target;
          bool merges_only_leaves =
              result.GetFsm().GetEdges(previous_target).empty() &&
              result.GetFsm().GetEdges(edge.target).empty() &&
              result.IsEndState(previous_target) == result.IsEndState(edge.target);
          may_have_non_leaf_merge |= !merges_only_leaves;
        }
      }
    }

    original_incoming_edges.ResetWithRowSizes(original_incoming_row_sizes);
    std::vector<int32_t> original_incoming_write_positions(original_num_states, 0);
    for (int source = 0; source < original_num_states; ++source) {
      for (const auto& edge : result.GetFsm().GetEdges(source)) {
        original_incoming_edges.MutableRowAt(edge.target
        )[original_incoming_write_positions[edge.target]++] = {source, edge.min, edge.max};
      }
    }

    std::vector<EndpointEdge> incoming_edge_scratch;
    for (int state = 0; !may_have_non_leaf_merge && state < original_num_states; ++state) {
      auto incoming_row = original_incoming_edges[state];
      if (incoming_row.size() < 2) {
        continue;
      }
      incoming_edge_scratch.assign(incoming_row.begin(), incoming_row.end());
      std::sort(
          incoming_edge_scratch.begin(),
          incoming_edge_scratch.end(),
          [](const EndpointEdge& lhs, const EndpointEdge& rhs) {
            return std::tie(lhs.min, lhs.max, lhs.peer) < std::tie(rhs.min, rhs.max, rhs.peer);
          }
      );
      for (int32_t index = 1; index < static_cast<int32_t>(incoming_edge_scratch.size()); ++index) {
        if (incoming_edge_scratch[index - 1].min == incoming_edge_scratch[index].min &&
            incoming_edge_scratch[index - 1].max == incoming_edge_scratch[index].max &&
            incoming_edge_scratch[index - 1].peer != incoming_edge_scratch[index].peer) {
          may_have_non_leaf_merge = true;
          break;
        }
      }
    }

    if (may_have_non_leaf_merge) {
      break;
    }

    std::vector<int32_t> leaf_states[2];
    for (int state = 0; state < original_num_states; ++state) {
      if (result.GetFsm().GetEdges(state).empty()) {
        leaf_states[result.IsEndState(state)].push_back(state);
      }
    }
    if (leaf_states[0].size() < 2 && leaf_states[1].size() < 2) {
      // No local rule can apply, so no later merge can be exposed.
      return result;
    }

    UnionFindSet<int> union_find_set;
    for (const auto& states : leaf_states) {
      for (size_t index = 1; index < states.size(); ++index) {
        union_find_set.Add(states[0]);
        union_find_set.Add(states[index]);
        union_find_set.Union(states[0], states[index]);
      }
    }
    auto equivalent_classes = union_find_set.GetAllSets();
    std::vector<int> old_to_new(original_num_states, -1);
    for (size_t index = 0; index < equivalent_classes.size(); ++index) {
      for (int state : equivalent_classes[index]) {
        old_to_new[state] = index;
      }
    }
    int new_num_states = equivalent_classes.size();
    for (int state = 0; state < original_num_states; ++state) {
      if (old_to_new[state] == -1) {
        old_to_new[state] = new_num_states++;
      }
    }
    result = result.RebuildWithMapping(old_to_new, new_num_states);
    result.GetFsm()->SortEdges();
  }
  constexpr int kNumStateDataArrays = 15;
  std::vector<int> state_data(static_cast<size_t>(original_num_states) * kNumStateDataArrays, 0);
  size_t state_data_offset = 0;
  auto AllocateStateData = [&] {
    int* data = state_data.data() + state_data_offset;
    state_data_offset += original_num_states;
    return data;
  };
  int* parent = AllocateStateData();
  int* class_sizes = AllocateStateData();
  int* first_members = AllocateStateData();
  int* last_members = AllocateStateData();
  int* next_members = AllocateStateData();
  int* current_ids = AllocateStateData();
  int* outgoing_peer_generations = AllocateStateData();
  int* incoming_peer_generations = AllocateStateData();
  int* single_outgoing_peers = AllocateStateData();
  int* single_incoming_peers = AllocateStateData();
  int* dirty_state_generations = AllocateStateData();
  int* round_parents = AllocateStateData();
  int* round_class_sizes = AllocateStateData();
  int* round_component_indices = AllocateStateData();
  int* merged_state_generations = AllocateStateData();
  XGRAMMAR_DCHECK(state_data_offset == state_data.size());
  std::fill(class_sizes, class_sizes + original_num_states, 1);
  std::fill(next_members, next_members + original_num_states, -1);
  std::fill(outgoing_peer_generations, outgoing_peer_generations + original_num_states, -1);
  std::fill(incoming_peer_generations, incoming_peer_generations + original_num_states, -1);
  std::fill(single_outgoing_peers, single_outgoing_peers + original_num_states, -1);
  std::fill(single_incoming_peers, single_incoming_peers + original_num_states, -1);
  std::fill(round_parents, round_parents + original_num_states, -1);
  std::fill(round_component_indices, round_component_indices + original_num_states, -1);
  std::vector<bool> is_end_state(original_num_states);
  for (int state = 0; state < original_num_states; ++state) {
    parent[state] = state;
    first_members[state] = state;
    last_members[state] = state;
    current_ids[state] = state;
    is_end_state[state] = result.IsEndState(state);
  }

  auto FindRoot = [&](int state) {
    int root = state;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[state] != state) {
      int next = parent[state];
      parent[state] = root;
      state = next;
    }
    return root;
  };

  bool has_rebuilt = false;
  auto CollectOutgoingEdges = [&](int state, std::vector<EndpointEdge>* edges) {
    int root = FindRoot(state);
    edges->clear();
    for (int member = first_members[root]; member != -1; member = next_members[member]) {
      for (const auto& edge : result.GetFsm().GetEdges(member)) {
        int target = FindRoot(edge.target);
        if (has_rebuilt && edge.IsEpsilon() && target == root) {
          continue;
        }
        edges->push_back({target, edge.min, edge.max});
      }
    }
    std::sort(edges->begin(), edges->end());
    if (has_rebuilt) {
      edges->erase(std::unique(edges->begin(), edges->end()), edges->end());
    }
  };

  auto CollectIncomingEdges = [&](int state, std::vector<EndpointEdge>* edges) {
    int root = FindRoot(state);
    edges->clear();
    for (int member = first_members[root]; member != -1; member = next_members[member]) {
      for (const auto& edge : original_incoming_edges[member]) {
        int source = FindRoot(edge.peer);
        if (has_rebuilt && edge.min == FSMEdge::EdgeType::kEpsilon && source == root) {
          continue;
        }
        edges->push_back({source, edge.min, edge.max});
      }
    }
    std::sort(edges->begin(), edges->end());
    if (has_rebuilt) {
      edges->erase(std::unique(edges->begin(), edges->end()), edges->end());
    }
  };

  int graph_generation = 0;

  auto GetSingleOutgoingPeer = [&](int state) {
    int root = FindRoot(state);
    if (outgoing_peer_generations[root] == graph_generation) {
      return single_outgoing_peers[root];
    }
    outgoing_peer_generations[root] = graph_generation;
    int single_peer = -1;
    for (int member = first_members[root]; member != -1; member = next_members[member]) {
      for (const auto& edge : result.GetFsm().GetEdges(member)) {
        int target = FindRoot(edge.target);
        if (has_rebuilt && edge.IsEpsilon() && target == root) {
          continue;
        }
        if (single_peer == -1) {
          single_peer = target;
        } else if (single_peer != target) {
          single_peer = -2;
          break;
        }
      }
      if (single_peer == -2) {
        break;
      }
    }
    single_outgoing_peers[root] = single_peer;
    return single_peer;
  };

  auto GetSingleIncomingPeer = [&](int state) {
    int root = FindRoot(state);
    if (incoming_peer_generations[root] == graph_generation) {
      return single_incoming_peers[root];
    }
    incoming_peer_generations[root] = graph_generation;
    int single_peer = -1;
    for (int member = first_members[root]; member != -1; member = next_members[member]) {
      for (const auto& edge : original_incoming_edges[member]) {
        int source = FindRoot(edge.peer);
        if (has_rebuilt && edge.min == FSMEdge::EdgeType::kEpsilon && source == root) {
          continue;
        }
        if (single_peer == -1) {
          single_peer = source;
        } else if (single_peer != source) {
          single_peer = -2;
          break;
        }
      }
      if (single_peer == -2) {
        break;
      }
    }
    single_incoming_peers[root] = single_peer;
    return single_peer;
  };

  struct MergeCandidate {
    int state;
    int32_t edge_begin;
    int32_t edge_end;
    bool is_end;
  };

  auto AppendCandidateGroups = [&](std::vector<MergeCandidate>* candidates,
                                   const std::vector<EndpointEdge>& edges,
                                   std::vector<std::vector<int>>* merge_groups,
                                   std::vector<bool>* states_merged_by_case_1) {
    auto CompareEdgeRanges = [&](const MergeCandidate& lhs, const MergeCandidate& rhs) {
      int32_t lhs_index = lhs.edge_begin;
      int32_t rhs_index = rhs.edge_begin;
      while (lhs_index < lhs.edge_end && rhs_index < rhs.edge_end) {
        auto lhs_range = std::tie(edges[lhs_index].min, edges[lhs_index].max);
        auto rhs_range = std::tie(edges[rhs_index].min, edges[rhs_index].max);
        if (lhs_range < rhs_range) {
          return -1;
        }
        if (rhs_range < lhs_range) {
          return 1;
        }
        ++lhs_index;
        ++rhs_index;
      }
      if (lhs_index == lhs.edge_end && rhs_index == rhs.edge_end) {
        return 0;
      }
      return lhs_index == lhs.edge_end ? -1 : 1;
    };

    std::sort(
        candidates->begin(),
        candidates->end(),
        [&](const MergeCandidate& lhs, const MergeCandidate& rhs) {
          if (lhs.is_end != rhs.is_end) {
            return lhs.is_end < rhs.is_end;
          }
          int range_comparison = CompareEdgeRanges(lhs, rhs);
          if (range_comparison != 0) {
            return range_comparison < 0;
          }
          return current_ids[lhs.state] < current_ids[rhs.state];
        }
    );

    int32_t group_begin = 0;
    while (group_begin < static_cast<int32_t>(candidates->size())) {
      int32_t group_end = group_begin + 1;
      while (group_end < static_cast<int32_t>(candidates->size()) &&
             (*candidates)[group_begin].is_end == (*candidates)[group_end].is_end &&
             CompareEdgeRanges((*candidates)[group_begin], (*candidates)[group_end]) == 0) {
        ++group_end;
      }
      if (group_end - group_begin > 1) {
        auto& group = merge_groups->emplace_back();
        group.reserve(group_end - group_begin);
        for (int32_t index = group_begin; index < group_end; ++index) {
          int state = (*candidates)[index].state;
          group.push_back(state);
          if (states_merged_by_case_1 != nullptr) {
            (*states_merged_by_case_1)[state] = true;
          }
        }
      }
      group_begin = group_end;
    }
  };

  std::vector<int> dirty_states(original_num_states);
  std::vector<int> active_states(original_num_states);
  for (int state = 0; state < original_num_states; ++state) {
    dirty_states[state] = state;
    active_states[state] = state;
  }
  int dirty_state_generation = 0;
  int merged_state_generation = 0;
  int leaf_representatives[2] = {-1, -1};
  int active_num_states = original_num_states;
  std::vector<EndpointEdge> center_edges;
  std::vector<EndpointEdge> neighbor_edges;
  std::vector<MergeCandidate> candidates;

  while (!dirty_states.empty()) {
    ++graph_generation;
    ++dirty_state_generation;
    for (int state : dirty_states) {
      dirty_state_generations[FindRoot(state)] = dirty_state_generation;
    }
    dirty_states.clear();
    for (int state : active_states) {
      if (dirty_state_generations[state] == dirty_state_generation) {
        dirty_states.push_back(state);
      }
    }

    std::vector<std::vector<int>> merge_groups;
    std::vector<bool> states_merged_by_case_1(original_num_states, false);

    // Case 1: Like ab | ac | ad, merge the successors into a(b | c | d).
    for (int center : dirty_states) {
      CollectOutgoingEdges(center, &center_edges);
      candidates.clear();
      int32_t edge_begin = 0;
      while (edge_begin < static_cast<int32_t>(center_edges.size())) {
        int target = center_edges[edge_begin].peer;
        int32_t edge_end = edge_begin + 1;
        while (edge_end < static_cast<int32_t>(center_edges.size()) &&
               center_edges[edge_end].peer == target) {
          ++edge_end;
        }
        if (GetSingleIncomingPeer(target) == center) {
          candidates.push_back({target, edge_begin, edge_end, is_end_state[target]});
        }
        edge_begin = edge_end;
      }
      AppendCandidateGroups(&candidates, center_edges, &merge_groups, &states_merged_by_case_1);
    }

    // Case 2: Like ba | ca | da, merge the precursors into (b | c | d)a. States already selected
    // by Case 1 use stale outgoing neighborhoods until this logical round is applied.
    for (int center : dirty_states) {
      CollectIncomingEdges(center, &center_edges);
      candidates.clear();
      int32_t edge_begin = 0;
      while (edge_begin < static_cast<int32_t>(center_edges.size())) {
        int source = center_edges[edge_begin].peer;
        int32_t edge_end = edge_begin + 1;
        while (edge_end < static_cast<int32_t>(center_edges.size()) &&
               center_edges[edge_end].peer == source) {
          ++edge_end;
        }
        if (!states_merged_by_case_1[source] && GetSingleOutgoingPeer(source) == center) {
          candidates.push_back({source, edge_begin, edge_end, is_end_state[source]});
        }
        edge_begin = edge_end;
      }
      AppendCandidateGroups(&candidates, center_edges, &merge_groups, nullptr);
    }

    // Leaf states are the only candidates whose merge group has no adjacent center.
    std::vector<int> leaf_candidates[2];
    for (int state : dirty_states) {
      if (GetSingleOutgoingPeer(state) == -1) {
        leaf_candidates[is_end_state[state]].push_back(state);
      }
    }
    for (int end_status = 0; end_status < 2; ++end_status) {
      if (leaf_representatives[end_status] != -1) {
        int representative = FindRoot(leaf_representatives[end_status]);
        if (is_end_state[representative] == static_cast<bool>(end_status) &&
            GetSingleOutgoingPeer(representative) == -1) {
          leaf_candidates[end_status].push_back(representative);
        }
      }
      std::sort(leaf_candidates[end_status].begin(), leaf_candidates[end_status].end());
      leaf_candidates[end_status].erase(
          std::unique(leaf_candidates[end_status].begin(), leaf_candidates[end_status].end()),
          leaf_candidates[end_status].end()
      );
      std::sort(
          leaf_candidates[end_status].begin(),
          leaf_candidates[end_status].end(),
          [&](int lhs, int rhs) { return current_ids[lhs] < current_ids[rhs]; }
      );
      if (!leaf_candidates[end_status].empty()) {
        leaf_representatives[end_status] = leaf_candidates[end_status][0];
      }
      if (leaf_candidates[end_status].size() > 1) {
        merge_groups.push_back(leaf_candidates[end_status]);
      }
    }

    if (merge_groups.empty()) {
      break;
    }

    // Combine overlapping groups found in the same logical round.
    std::vector<int> round_touched_states;
    auto FindRoundRoot = [&](int state) {
      int root = state;
      while (round_parents[root] != root) {
        root = round_parents[root];
      }
      while (round_parents[state] != state) {
        int next = round_parents[state];
        round_parents[state] = root;
        state = next;
      }
      return root;
    };
    for (const auto& group : merge_groups) {
      for (int state : group) {
        if (round_parents[state] == -1) {
          round_parents[state] = state;
          round_class_sizes[state] = 1;
          round_touched_states.push_back(state);
        }
      }
      for (size_t index = 1; index < group.size(); ++index) {
        int lhs_root = FindRoundRoot(group[0]);
        int rhs_root = FindRoundRoot(group[index]);
        if (lhs_root == rhs_root) {
          continue;
        }
        if (round_class_sizes[lhs_root] < round_class_sizes[rhs_root]) {
          std::swap(lhs_root, rhs_root);
        }
        round_parents[rhs_root] = lhs_root;
        round_class_sizes[lhs_root] += round_class_sizes[rhs_root];
      }
    }

    std::vector<std::vector<int>> equivalent_classes;
    for (int state : active_states) {
      if (round_parents[state] == -1) {
        continue;
      }
      int round_root = FindRoundRoot(state);
      if (round_component_indices[round_root] == -1) {
        round_component_indices[round_root] = equivalent_classes.size();
        equivalent_classes.emplace_back();
      }
      equivalent_classes[round_component_indices[round_root]].push_back(state);
    }

    ++merged_state_generation;
    std::vector<int> affected_states;
    for (const auto& equivalent_class : equivalent_classes) {
      for (int state : equivalent_class) {
        merged_state_generations[state] = merged_state_generation;
        affected_states.push_back(state);
        CollectIncomingEdges(state, &neighbor_edges);
        for (const auto& edge : neighbor_edges) {
          affected_states.push_back(edge.peer);
        }
        CollectOutgoingEdges(state, &neighbor_edges);
        for (const auto& edge : neighbor_edges) {
          affected_states.push_back(edge.peer);
        }
      }
    }

    std::vector<int> merged_roots;
    merged_roots.reserve(equivalent_classes.size());
    for (const auto& equivalent_class : equivalent_classes) {
      int merged_root = equivalent_class[0];
      for (int state : equivalent_class) {
        if (class_sizes[state] > class_sizes[merged_root]) {
          merged_root = state;
        }
      }
      for (int state : equivalent_class) {
        XGRAMMAR_DCHECK(is_end_state[state] == is_end_state[merged_root]);
        if (state == merged_root) {
          continue;
        }
        parent[state] = merged_root;
        class_sizes[merged_root] += class_sizes[state];
        next_members[last_members[merged_root]] = first_members[state];
        last_members[merged_root] = last_members[state];
      }
      merged_roots.push_back(merged_root);
    }

    std::vector<int> next_active_states;
    next_active_states.reserve(
        active_num_states - round_touched_states.size() + equivalent_classes.size()
    );
    next_active_states.insert(next_active_states.end(), merged_roots.begin(), merged_roots.end());
    for (int state : active_states) {
      if (merged_state_generations[state] != merged_state_generation) {
        next_active_states.push_back(state);
      }
    }
    active_states = std::move(next_active_states);
    active_num_states = active_states.size();
    for (int current_id = 0; current_id < active_num_states; ++current_id) {
      current_ids[active_states[current_id]] = current_id;
    }

    for (const auto& equivalent_class : equivalent_classes) {
      int round_root = FindRoundRoot(equivalent_class[0]);
      round_component_indices[round_root] = -1;
    }
    for (int state : round_touched_states) {
      round_parents[state] = -1;
    }

    bool first_rebuild = !has_rebuilt;
    has_rebuilt = true;
    dirty_states.clear();
    if (first_rebuild && !input_edges_are_canonical) {
      dirty_states = active_states;
    } else {
      affected_states.insert(affected_states.end(), merged_roots.begin(), merged_roots.end());
      for (int state : affected_states) {
        dirty_states.push_back(FindRoot(state));
      }
    }
  }

  if (!has_rebuilt) {
    return result;
  }

  std::vector<int> original_to_final(original_num_states);
  for (int state = 0; state < original_num_states; ++state) {
    original_to_final[state] = current_ids[FindRoot(state)];
  }
  return result.RebuildWithMapping(original_to_final, active_num_states);
}

Result<FSMWithStartEnd> FSMWithStartEnd::MinimizeDFA(int max_num_states) const {
  FSMWithStartEnd now_fsm(FSM(0), 0, std::vector<int32_t>(), true);
  if (NumStates() > max_num_states) {
    return ResultErr("The number of states exceeds the limit.");
  }
  // To perform the algorithm, we must make sure the FSM is
  // a DFA.
  if (!is_dfa_) {
    Result<FSMWithStartEnd> dfa_raw = ToDFA(max_num_states);
    if (dfa_raw.IsErr()) {
      return dfa_raw;
    }
    now_fsm = std::move(dfa_raw).Unwrap();
  } else {
    now_fsm = Copy();
  }

  // Initialize the precursors of nodes.
  std::vector<std::vector<std::pair<std::pair<int32_t, int32_t>, int>>> precursors;
  precursors.resize(now_fsm.NumStates());
  for (int i = 0; i < now_fsm.NumStates(); ++i) {
    const auto& edges = now_fsm.GetFsm().GetEdges(i);
    for (const auto& edge : edges) {
      XGRAMMAR_DCHECK(!edge.IsEpsilon());
      precursors[edge.target].push_back(std::make_pair(std::make_pair(edge.min, edge.max), i));
    }
  }

  // Initialize the partitions and working set.
  std::vector<std::unordered_set<int>> partitions;
  std::vector<std::unordered_set<int>> working_set;
  std::unordered_set<int> final_states;
  std::unordered_set<int> non_final_states;
  for (int i = 0; i < now_fsm.NumStates(); ++i) {
    if (now_fsm.IsEndState(i)) {
      final_states.insert(i);
    } else {
      non_final_states.insert(i);
    }
  }
  partitions.push_back(final_states);
  partitions.push_back(non_final_states);
  working_set.push_back(std::move(final_states));
  working_set.push_back(std::move(non_final_states));

  while (!working_set.empty()) {
    std::map<std::pair<int32_t, int32_t>, std::unordered_set<int>> possible_transitions;
    auto current_partition = std::move(working_set.back());
    working_set.pop_back();

    // Get the possible transitions from the current partition.
    for (const auto& state : current_partition) {
      const auto& precursor_map = precursors[state];
      for (const auto& precursor : precursor_map) {
        if (possible_transitions.find(precursor.first) == possible_transitions.end()) {
          possible_transitions[precursor.first] = std::unordered_set<int>();
        }
        possible_transitions[precursor.first].insert(precursor.second);
      }
    }

    // Check each possible transition.
    std::vector<int> intersection;
    std::vector<int> difference;
    for (const auto& [transition, precursors] : possible_transitions) {
      for (size_t i = 0; i < partitions.size(); i++) {
        const auto& partition = partitions[i];
        intersection.clear();  // partition \cap precursors
        difference.clear();    // partition - precursors
        for (const auto& partition_state : partition) {
          if (precursors.find(partition_state) != precursors.end()) {
            intersection.push_back(partition_state);
          } else {
            difference.push_back(partition_state);
          }
        }

        // the states in the partition is not equivalent. We need to
        // update the working set and the partitions.
        if ((!intersection.empty()) && (!difference.empty())) {
          bool in_working_set = false;
          for (size_t i = 0; i < working_set.size(); i++) {
            if (partition == working_set[i]) {
              in_working_set = true;
              working_set[i].clear();
              for (const auto& state : intersection) {
                working_set[i].insert(state);
              }
              working_set.emplace_back();
              for (const auto& state : difference) {
                working_set.back().insert(state);
              }
              break;
            }
          }
          if (!in_working_set) {
            const auto& smaller_set =
                difference.size() < intersection.size() ? difference : intersection;
            working_set.emplace_back();
            for (const auto& state : smaller_set) {
              working_set.back().insert(state);
            }
          }
          partitions[i].clear();
          for (const auto& state : intersection) {
            partitions[i].insert(state);
          }
          partitions.emplace_back();
          for (const auto& state : difference) {
            partitions.back().insert(state);
          }
        }
      }
    }
  }
  std::vector<int> state_mapping(now_fsm.NumStates(), -1);
  for (size_t i = 0; i < partitions.size(); ++i) {
    for (const auto& state : partitions[i]) {
      state_mapping[state] = i;
    }
  }
  int new_num_states = partitions.size();
  return ResultOk(now_fsm.RebuildWithMapping(state_mapping, new_num_states));
}

Result<FSMWithStartEnd> FSMWithStartEnd::ToDFA(int max_num_states) const {
  if (NumStates() > max_num_states) {
    return ResultErr("The number of states exceeds the limit.");
  }
  FSMWithStartEnd dfa(FSM(0), 0, std::vector<int32_t>(), true);
  std::vector<std::unordered_set<int>> closures;
  std::unordered_set<int> rules;
  std::unordered_set<int32_t> repeat_aux_indices;
  int now_process = 0;
  std::unordered_set<int> closure;
  closure.insert(start_);
  fsm_.GetEpsilonClosure(&closure);
  closures.push_back(closure);
  while (now_process < static_cast<int>(closures.size())) {
    rules.clear();
    repeat_aux_indices.clear();
    std::unordered_set<int32_t> token_aux_indices;
    std::unordered_set<int32_t> exclude_token_aux_indices;
    std::set<int> interval_ends;
    std::bitset<256> allowed_characters;
    dfa.AddState();
    // Check if the closure is a final state.
    for (const auto& state : closures[now_process]) {
      if (IsEndState(state)) {
        dfa.AddEndState(now_process);
      }
      const auto& edges = fsm_->GetEdges(state);
      for (const auto& edge : edges) {
        if (edge.IsCharRange()) {
          interval_ends.insert(edge.min);
          interval_ends.insert(edge.max + 1);
          for (int i = edge.min; i <= edge.max; ++i) {
            allowed_characters.set(i);
          }
          continue;
        } else if (edge.IsRuleRef()) {
          rules.insert(edge.GetRefRuleId());
        } else if (edge.IsRepeatRef()) {
          repeat_aux_indices.insert(edge.GetAuxIndex());
        } else if (edge.IsToken()) {
          token_aux_indices.insert(edge.GetAuxIndex());
        } else if (edge.IsExcludeToken()) {
          exclude_token_aux_indices.insert(edge.GetAuxIndex());
        }
      }
    }
    // This part is to get the all possible intervals.
    // Which can help reduce the transitions.
    using Interval = std::pair<int, int>;
    std::vector<Interval> intervals;
    intervals.reserve(interval_ends.size());
    int last = -1;
    for (const auto& end : interval_ends) {
      if (last == -1) {
        last = end;
        continue;
      }
      bool allowed = true;
      for (int i = last; i < end; ++i) {
        if (!allowed_characters[i]) {
          allowed = false;
          break;
        }
      }
      if (allowed) {
        intervals.emplace_back(last, end - 1);
      }
      last = end;
    }
    for (const auto& interval : intervals) {
      std::unordered_set<int> next_closure;
      for (const auto& state : closures[now_process]) {
        const auto& edges = fsm_->GetEdges(state);
        for (const auto& edge : edges) {
          if (edge.IsCharRange()) {
            if (interval.first >= edge.min && interval.second <= edge.max) {
              if (next_closure.find(edge.target) == next_closure.end()) {
                std::unordered_set<int> epsilon_closure;
                epsilon_closure.insert(edge.target);
                fsm_.GetEpsilonClosure(&epsilon_closure);
                next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
              }
            }
          }
        }
      }
      bool flag = false;
      for (int j = 0; j < static_cast<int>(closures.size()); j++) {
        if (closures[j] == next_closure) {
          dfa.GetFsm().AddEdge(now_process, j, interval.first, interval.second);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa.GetFsm().AddEdge(now_process, closures.size(), interval.first, interval.second);
        closures.push_back(next_closure);
      }
    }
    for (auto rule : rules) {
      std::unordered_set<int> next_closure;
      for (const auto& state : closures[now_process]) {
        const auto& edges = fsm_.GetEdges(state);
        for (const auto& edge : edges) {
          if (edge.IsRuleRef()) {
            if (rule == edge.GetRefRuleId()) {
              if (next_closure.find(edge.target) == next_closure.end()) {
                std::unordered_set<int> epsilon_closure;
                epsilon_closure.insert(edge.target);
                fsm_.GetEpsilonClosure(&epsilon_closure);
                next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
              }
            }
          }
        }
      }
      bool flag = false;
      for (int j = 0; j < static_cast<int>(closures.size()); j++) {
        if (closures[j] == next_closure) {
          dfa.GetFsm().AddRuleEdge(now_process, j, rule);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa.GetFsm().AddRuleEdge(now_process, closures.size(), rule);
        closures.push_back(next_closure);
      }
    }

    for (auto aux_idx : repeat_aux_indices) {
      std::unordered_set<int> next_closure;
      for (const auto& state : closures[now_process]) {
        const auto& edges = fsm_.GetEdges(state);
        for (const auto& edge : edges) {
          if (edge.IsRepeatRef() && edge.GetAuxIndex() == aux_idx) {
            if (next_closure.find(edge.target) == next_closure.end()) {
              std::unordered_set<int> epsilon_closure;
              epsilon_closure.insert(edge.target);
              fsm_.GetEpsilonClosure(&epsilon_closure);
              next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
            }
          }
        }
      }
      bool flag = false;
      for (int j = 0; j < static_cast<int>(closures.size()); j++) {
        if (closures[j] == next_closure) {
          dfa.GetFsm().AddEdge(now_process, j, FSMEdge::EdgeType::kRepeatRef, aux_idx);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa.GetFsm().AddEdge(now_process, closures.size(), FSMEdge::EdgeType::kRepeatRef, aux_idx);
        closures.push_back(next_closure);
      }
    }

    for (auto aux_idx : token_aux_indices) {
      std::unordered_set<int> next_closure;
      for (const auto& state : closures[now_process]) {
        const auto& edges = fsm_.GetEdges(state);
        for (const auto& edge : edges) {
          if (edge.IsToken() && edge.GetAuxIndex() == aux_idx) {
            if (next_closure.find(edge.target) == next_closure.end()) {
              std::unordered_set<int> epsilon_closure;
              epsilon_closure.insert(edge.target);
              fsm_.GetEpsilonClosure(&epsilon_closure);
              next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
            }
          }
        }
      }
      bool flag = false;
      for (int j = 0; j < static_cast<int>(closures.size()); j++) {
        if (closures[j] == next_closure) {
          dfa.GetFsm().AddEdge(now_process, j, FSMEdge::EdgeType::kToken, aux_idx);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa.GetFsm().AddEdge(now_process, closures.size(), FSMEdge::EdgeType::kToken, aux_idx);
        closures.push_back(next_closure);
      }
    }

    for (auto aux_idx : exclude_token_aux_indices) {
      std::unordered_set<int> next_closure;
      for (const auto& state : closures[now_process]) {
        const auto& edges = fsm_.GetEdges(state);
        for (const auto& edge : edges) {
          if (edge.IsExcludeToken() && edge.GetAuxIndex() == aux_idx) {
            if (next_closure.find(edge.target) == next_closure.end()) {
              std::unordered_set<int> epsilon_closure;
              epsilon_closure.insert(edge.target);
              fsm_.GetEpsilonClosure(&epsilon_closure);
              next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
            }
          }
        }
      }
      bool flag = false;
      for (int j = 0; j < static_cast<int>(closures.size()); j++) {
        if (closures[j] == next_closure) {
          dfa.GetFsm().AddEdge(now_process, j, FSMEdge::EdgeType::kExcludeToken, aux_idx);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa.GetFsm().AddEdge(
            now_process, closures.size(), FSMEdge::EdgeType::kExcludeToken, aux_idx
        );
        closures.push_back(next_closure);
      }
    }

    now_process++;
  }
  dfa.GetFsm().SetEdgeAuxData(std::vector<int32_t>(fsm_.GetEdgeAuxData()));
  dfa.is_dfa_ = true;
  return ResultOk(dfa);
}

/****************** CompactFSMWithStartEnd ******************/

std::string CompactFSMWithStartEnd::ToString() const {
  std::string result;
  result += "CompactFSM(num_states=" + std::to_string(NumStates()) +
            ", start=" + std::to_string(start_) + ", end=[";

  std::unordered_set<int> reachable_states;
  GetReachableStates(&reachable_states);
  std::vector<int> reachable_states_vec(reachable_states.begin(), reachable_states.end());
  std::sort(reachable_states_vec.begin(), reachable_states_vec.end());
  bool first = true;
  for (auto end : ends_) {
    if (reachable_states.count(end)) {
      if (!first) {
        result += ", ";
      }
      first = false;
      result += std::to_string(end);
    }
  }

  result += "], edges=" + fsm_.EdgesToString(reachable_states_vec) + ")";
  return result;
}

std::ostream& operator<<(std::ostream& os, const CompactFSMWithStartEnd& fsm) {
  os << fsm.ToString();
  return os;
}

std::size_t MemorySize(const CompactFSM& self) { return MemorySize(*self.ImplPtr()); }

std::size_t MemorySize(const CompactFSMWithStartEnd& self) {
  // The underlying CompactFSM is not counted here: CompactFSMWithStartEnd is a view, and many
  // views usually share one CompactFSM (e.g. the per-rule FSMs of a grammar all point to the
  // grammar's complete FSM, which is counted by the grammar itself). Counting it per view would
  // multiply the shared FSM's size by the number of views.
  return MemorySize(self.ends_);
}

FSMWithStartEnd CompactFSMWithStartEnd::ToFSM() const {
  return FSMWithStartEnd(fsm_.ToFSM(), start_, ends_);
}

}  // namespace xgrammar
