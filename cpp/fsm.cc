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

  /*! \brief Copy constructor. */
  FSMImplBase(const ContainerType& edges) : edges_(edges) {}

  /*! \brief Move constructor. */
  FSMImplBase(ContainerType&& edges) : edges_(std::move(edges)) {}

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

 protected:
  ContainerType edges_;
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

  void AddEdge(int from, int to, int16_t min, int16_t max) {
    XGRAMMAR_DCHECK(from < static_cast<int>(edges_.size()));
    edges_[from].push_back({min, max, to});
  }

  void AddRuleEdge(int from, int to, int16_t rule_id) {
    AddEdge(from, to, FSMEdge::EdgeType::kRuleRef, rule_id);
  }

  void AddEpsilonEdge(int from, int to) { AddEdge(from, to, FSMEdge::EdgeType::kEpsilon, 0); }

  void AddEOSEdge(int from, int to) { AddEdge(from, to, FSMEdge::EdgeType::kEOS, 0); }

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
      if (edge.min == EdgeType::kRuleRef && edge.max == value) {
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
  } else {
    XGRAMMAR_DCHECK(false) << "Invalid edge type: " << static_cast<int>(edge_type);
  }

  // Get the epsilon closure of the result.
  GetEpsilonClosure(result);
}

void FSM::Impl::AddFSM(const FSM& fsm, std::vector<int>* state_mapping) {
  int old_num_states = NumStates();

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
      AddEdge(i + old_num_states, edge.target + old_num_states, edge.min, edge.max);
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
  for (int i = 0; i < new_num_states; ++i) {
    std::sort(new_edges[i].begin(), new_edges[i].end());
    const auto& end_iter = std::unique(new_edges[i].begin(), new_edges[i].end());
    new_edges[i].erase(end_iter, new_edges[i].end());
  }
  return FSM(std::move(new_edges));
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
  return CompactFSM(edges);
}

/****************** FSM ******************/

FSM::FSM(int num_states) : pimpl_(std::make_shared<Impl>(num_states)) {}

FSM::FSM(const std::vector<std::vector<FSMEdge>>& edges) : pimpl_(std::make_shared<Impl>(edges)) {}

FSM::FSM(std::vector<std::vector<FSMEdge>>&& edges)
    : pimpl_(std::make_shared<Impl>(std::move(edges))) {}

int FSM::NumStates() const { return pimpl_->NumStates(); }

int FSM::AddState() { return pimpl_->AddState(); }

void FSM::AddEdge(int from, int to, int16_t min, int16_t max) {
  pimpl_->AddEdge(from, to, min, max);
}

void FSM::AddEpsilonEdge(int from, int to) { pimpl_->AddEpsilonEdge(from, to); }

void FSM::AddRuleEdge(int from, int to, int16_t rule_id) { pimpl_->AddRuleEdge(from, to, rule_id); }

void FSM::AddEOSEdge(int from, int to) { pimpl_->AddEOSEdge(from, to); }

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
  using FSMImplBase<Compact2DArray<FSMEdge>>::FSMImplBase;

  void GetNextStates(int from, int value, EdgeType edge_type, std::vector<int>* target) const;

  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      FSMEdge::EdgeType edge_type,
      bool from_is_closure
  ) const;

  FSM ToFSM() const;

  friend std::size_t MemorySize(const Impl& impl) { return MemorySize(impl.edges_); }
};

XGRAMMAR_MEMBER_ARRAY(CompactFSM::Impl, &CompactFSM::Impl::edges_);

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
      } else if (edge.max == value) {
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
        } else if (edge.max == value) {
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
  return FSM(edges);
}

/****************** CompactFSM ******************/

CompactFSM::CompactFSM(const Compact2DArray<FSMEdge>& edges)
    : pimpl_(std::make_shared<Impl>(edges)) {}

CompactFSM::CompactFSM(Compact2DArray<FSMEdge>&& edges)
    : pimpl_(std::make_shared<Impl>(std::move(edges))) {}

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

FSM CompactFSM::ToFSM() const { return pimpl_->ToFSM(); }

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

  CompactFSMWithStartEndSerializeHelper(const CompactFSMWithStartEnd& compact_fsm_with_se)
      : fsm(compact_fsm_with_se.fsm_),
        start(compact_fsm_with_se.start_),
        is_dfa(compact_fsm_with_se.is_dfa_) {
    end_index.reserve(compact_fsm_with_se.NumStates());
    for (int i = 0; i < static_cast<int>(compact_fsm_with_se.ends_.size()); ++i) {
      if (compact_fsm_with_se.ends_[i]) {
        end_index.push_back(i);
      }
    }
  }

  CompactFSMWithStartEndSerializeHelper() = default;
};

XGRAMMAR_MEMBER_ARRAY(
    CompactFSMWithStartEndSerializeHelper,
    &CompactFSMWithStartEndSerializeHelper::fsm,
    &CompactFSMWithStartEndSerializeHelper::start,
    &CompactFSMWithStartEndSerializeHelper::end_index,
    &CompactFSMWithStartEndSerializeHelper::is_dfa
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
  const auto& end_index = tmp.end_index;
  result->ends_.resize(result->fsm_.NumStates(), false);
  for (const auto& idx : end_index) {
    result->ends_[idx] = true;
  }
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
  for (int i = 0; i < NumStates(); ++i) {
    if (!IsEndState(i)) {
      continue;
    }
    if (!first) {
      result += ", ";
    }
    first = false;
    result += std::to_string(i);
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
  std::vector<bool> new_ends(new_num_states, false);
  for (int end = 0; end < NumStates(); ++end) {
    if (IsEndState(end)) {
      new_ends[state_mapping[end]] = true;
    }
  }
  return FSMWithStartEnd(new_fsm, new_start, new_ends);
}

CompactFSMWithStartEnd FSMWithStartEnd::ToCompact() {
  return CompactFSMWithStartEnd(fsm_.ToCompact(), start_, ends_);
}

FSMWithStartEnd FSMWithStartEnd::AddToCompleteFSM(
    FSM* complete_fsm, std::vector<int>* state_mapping
) {
  XGRAMMAR_DCHECK(state_mapping != nullptr) << "state_mapping cannot be nullptr";
  complete_fsm->AddFSM(fsm_, state_mapping);
  int new_start = (*state_mapping)[start_];
  std::vector<bool> new_ends(complete_fsm->NumStates(), false);
  for (int end = 0; end < NumStates(); ++end) {
    if (IsEndState(end)) {
      new_ends[(*state_mapping)[end]] = true;
    }
  }
  return FSMWithStartEnd(*complete_fsm, new_start, new_ends, is_dfa_);
}

FSMWithStartEnd FSMWithStartEnd::Star() const {
  FSM fsm = fsm_.Copy();
  auto new_start = fsm.AddState();
  for (int end = 0; end < NumStates(); ++end) {
    if (IsEndState(end)) {
      fsm.AddEpsilonEdge(end, new_start);
    }
  }
  fsm.AddEpsilonEdge(new_start, start_);
  std::vector<bool> is_end(NumStates() + 1, false);
  is_end[new_start] = true;
  return FSMWithStartEnd(fsm, new_start, is_end);
}

FSMWithStartEnd FSMWithStartEnd::Plus() const {
  FSM fsm = fsm_.Copy();
  for (int end = 0; end < NumStates(); ++end) {
    if (IsEndState(end)) {
      fsm.AddEpsilonEdge(end, start_);
    }
  }
  return FSMWithStartEnd(fsm, start_, ends_);
}

FSMWithStartEnd FSMWithStartEnd::Optional() const {
  FSM fsm = fsm_.Copy();
  for (int end = 0; end < NumStates(); ++end) {
    if (IsEndState(end)) {
      fsm.AddEpsilonEdge(start_, end);
      break;
    }
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
  std::vector<bool> new_final_states(result.NumStates() + 1, false);
  for (int i = 0; i < result.NumStates(); ++i) {
    if (!result.IsEndState(i)) {
      new_final_states[i] = true;  // Mark all states as final except the original final states.
    }
  }

  // Add a new final state that accepts all characters.
  int accept_all_new_state = result.AddState();
  new_final_states[accept_all_new_state] = true;

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
  std::vector<bool> ends(1, false);

  std::vector<int> state_mapping;

  for (const auto& fsm_with_se : fsms) {
    fsm.AddFSM(fsm_with_se.GetFsm(), &state_mapping);
    fsm.AddEpsilonEdge(start, state_mapping[fsm_with_se.GetStart()]);
    for (int state = 0; state < fsm_with_se.NumStates(); ++state) {
      ends.push_back(fsm_with_se.IsEndState(state));
    }
  }

  return FSMWithStartEnd(fsm, start, ends);
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
  std::vector<bool> ends;

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
      ends.resize(fsm.NumStates(), false);
      for (int end = 0; end < fsms[i].NumStates(); ++end) {
        if (fsms[i].IsEndState(end)) {
          ends[state_mapping[end]] = true;
        }
      }
    } else {
      previous_ends.clear();
      previous_ends.reserve(fsms[i].GetFsm().NumStates());
      for (int end = 0; end < fsms[i].NumStates(); ++end) {
        if (fsms[i].IsEndState(end)) {
          previous_ends.push_back(state_mapping[end]);
        }
      }
    }
  }

  return FSMWithStartEnd(fsm, start, ends);
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
  FSMWithStartEnd result(result_fsm, 0, std::vector<bool>(), true);
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
  for (int i = 0; i < NumStates(); i++) {
    const auto& edges = fsm_->GetEdges(i);
    for (const auto& edge : edges) {
      in_degree[edge.target]++;
      if (edge.IsEpsilon()) {
        if (edges.size() == 1) {
          // a -- epsilon --> b, and a doesn't have other outward edges.
          union_find_set.Add(i);
          union_find_set.Add(edge.target);
          union_find_set.Union(i, edge.target);
          in_degree[edge.target]--;  // Remove the inward edge since a and b are merged.
        } else {
          // a has other outward edges, we store it to check for another case.
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
    if (in_degree[to] == 1 && equiv_node[GetStart()] != to) {
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

FSMWithStartEnd FSMWithStartEnd::MergeEquivalentSuccessors(int max_result_num_states) const {
  if (max_result_num_states < NumStates()) {
    return *this;
  }
  bool changed = true;
  FSMWithStartEnd result = Copy();
  result.GetFsm()->SortEdges();
  UnionFindSet<int> union_find_set;
  while (changed) {
    union_find_set.Clear();
    std::vector<std::unordered_map<int, std::vector<FSMEdge>>> previous_states(result.NumStates());
    std::vector<std::unordered_map<int, std::vector<FSMEdge>>> next_states(result.NumStates());
    // Initialize the previous states.
    for (int i = 0; i < result.NumStates(); i++) {
      const auto& edges = result.GetFsm().GetEdges(i);
      for (const auto& edge : edges) {
        if (previous_states[edge.target].find(i) == previous_states[edge.target].end()) {
          previous_states[edge.target][i] = std::vector<FSMEdge>();
        }
        previous_states[edge.target][i].push_back(edge);
        if (next_states[i].find(edge.target) == next_states[i].end()) {
          next_states[i][edge.target] = std::vector<FSMEdge>();
        }
        next_states[i][edge.target].push_back(edge);
      }
    }
    // Case 1: Like ab | ac | ad, then they can be merged into a(b | c | d).
    bool is_equiv_successor = false;
    for (int i = 0; i < static_cast<int>(previous_states.size()); i++) {
      if (previous_states[i].size() != 1 || union_find_set.Count(i)) {
        continue;
      }
      const auto& previous_state = previous_states[i].begin()->first;
      const auto& edges_to_i = previous_states[i].begin()->second;
      const auto& siblings = next_states[previous_state];
      for (const auto& [sibling, edges_to_sibling] : siblings) {
        if (sibling <= i || previous_states[sibling].size() != 1 ||
            result.IsEndState(sibling) != result.IsEndState(i)) {
          continue;
        }
        bool is_equiv = true;

        // Check if the edges are the same.
        if (edges_to_i.size() != edges_to_sibling.size()) {
          break;  // Different edges, not equivalent.
        }
        for (int i = 0; i < static_cast<int>(edges_to_i.size()); i++) {
          if (edges_to_i[i].min != edges_to_sibling[i].min ||
              edges_to_i[i].max != edges_to_sibling[i].max) {
            is_equiv = false;
            break;  // Different edge ranges, not equivalent.
          }
        }

        // Merge the nodes.
        if (is_equiv) {
          union_find_set.Add(i);
          union_find_set.Add(sibling);
          union_find_set.Union(i, sibling);
          is_equiv_successor = true;
        }
      }
    }

    // Case 2: Like ba | ca | da, then they can be merged into (b | c | d)a.
    bool is_equiv_precursor = false;
    std::vector<int32_t> no_successor_end_states;
    std::vector<int32_t> no_successor_non_end_states;

    for (int i = 0; i < static_cast<int>(next_states.size()); i++) {
      if (next_states[i].empty()) {
        if (result.IsEndState(i)) {
          no_successor_end_states.push_back(i);
        } else {
          no_successor_non_end_states.push_back(i);
        }
        continue;  // Skip states with no successors.
      }
      if (next_states[i].size() != 1 || union_find_set.Count(i)) {
        continue;  // Skip states with multiple successors.
      }
      const auto& next_state = next_states[i].begin()->first;
      const auto& node_edges = result.GetFsm().GetEdges(i);
      const auto& siblings = previous_states[next_state];
      for (const auto& [sibling, edges_to_sibling] : siblings) {
        if (sibling <= i || next_states[sibling].size() != 1 ||
            result.IsEndState(i) != result.IsEndState(sibling)) {
          continue;
        }
        const auto& sibling_node_edges = result.GetFsm().GetEdges(sibling);
        if (sibling_node_edges.size() != node_edges.size()) {
          continue;  // Different number of edges, not equivalent.
        }
        bool is_equiv = true;
        for (int i = 0; i < static_cast<int>(sibling_node_edges.size()); i++) {
          if (sibling_node_edges[i].min != node_edges[i].min ||
              sibling_node_edges[i].max != node_edges[i].max) {
            is_equiv = false;
            break;
          }
        }

        if (is_equiv) {
          union_find_set.Add(i);
          union_find_set.Add(sibling);
          union_find_set.Union(i, sibling);
          is_equiv_successor = true;
        }
      }
    }

    if (no_successor_end_states.size() > 1) {
      // Merge all end states with no successors.
      for (size_t i = 1; i < no_successor_end_states.size(); ++i) {
        union_find_set.Add(no_successor_end_states[0]);
        union_find_set.Add(no_successor_end_states[i]);
        union_find_set.Union(no_successor_end_states[0], no_successor_end_states[i]);
        is_equiv_precursor = true;
      }
    }

    if (no_successor_non_end_states.size() > 1) {
      // Merge all non-end states with no successors.
      for (size_t i = 1; i < no_successor_non_end_states.size(); ++i) {
        union_find_set.Add(no_successor_non_end_states[0]);
        union_find_set.Add(no_successor_non_end_states[i]);
        union_find_set.Union(no_successor_non_end_states[0], no_successor_non_end_states[i]);
        is_equiv_precursor = true;
      }
    }

    changed = is_equiv_successor || is_equiv_precursor;
    if (changed) {
      auto eq_classes = union_find_set.GetAllSets();
      std::vector<int> old_to_new(result.NumStates(), -1);
      for (size_t i = 0; i < eq_classes.size(); i++) {
        for (const auto& state : eq_classes[i]) {
          old_to_new[state] = i;
        }
      }
      int cnt = eq_classes.size();
      for (int i = 0; i < result.NumStates(); i++) {
        if (old_to_new[i] == -1) {
          old_to_new[i] = cnt;
          cnt++;
        }
      }
      result = result.RebuildWithMapping(old_to_new, cnt);
      result.GetFsm()->SortEdges();
    }
  }
  return result;
}

Result<FSMWithStartEnd> FSMWithStartEnd::MinimizeDFA(int max_num_states) const {
  FSMWithStartEnd now_fsm(FSM(0), 0, std::vector<bool>(), true);
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
  std::vector<std::vector<std::pair<std::pair<int16_t, int16_t>, int>>> precursors;
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
    std::map<std::pair<int16_t, int16_t>, std::unordered_set<int>> possible_transitions;
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
  FSMWithStartEnd dfa(FSM(0), 0, std::vector<bool>(), true);
  std::vector<std::unordered_set<int>> closures;
  std::unordered_set<int> rules;
  int now_process = 0;
  std::unordered_set<int> closure;
  closure.insert(start_);
  fsm_.GetEpsilonClosure(&closure);
  closures.push_back(closure);
  while (now_process < static_cast<int>(closures.size())) {
    rules.clear();
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
    now_process++;
  }
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
  for (int end = 0; end < NumStates(); end++) {
    if (reachable_states.count(end) && IsEndState(end)) {
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
  return MemorySize(self.fsm_) + MemorySize(self.ends_);
}

FSMWithStartEnd CompactFSMWithStartEnd::ToFSM() const {
  return FSMWithStartEnd(fsm_.ToFSM(), start_, ends_);
}

}  // namespace xgrammar
