/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm.cc
 */
#include "fsm.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "picojson.h"
#include "support/encoding.h"
#include "support/json.h"
#include "support/logging.h"
#include "support/reflection.h"
#include "support/union_find_set.h"

namespace xgrammar {

/****************** FSMImplBase ******************/

template <typename ContainerType>
class FSMImplBase {
  static_assert(
      std::is_same_v<ContainerType, std::vector<std::vector<FSMEdge>>> ||
          std::is_same_v<ContainerType, CSRArray<FSMEdge>>,
      "ContainerType must be std::vector<std::vector<FSMEdge>> or CSRArray<FSMEdge>"
  );

 public:
  /*! \brief Default constructor. */
  FSMImplBase() = default;

  /*! \brief Copy constructor. */
  FSMImplBase(const ContainerType& edges) : edges_(edges) {}

  /*! \brief Move constructor. */
  FSMImplBase(ContainerType&& edges) : edges_(std::move(edges)) {}

  int NumStates() const { return edges_.size(); }

  std::string PrintEdges() const;

  const ContainerType& GetEdges() const { return edges_; }

  // For std::vector<std::vector<FSMEdge>>, return const std::vector<FSMEdge>& to avoid copying.
  // For CSRArray<FSMEdge>, return CSRArray<FSMEdge>::Row since it is just a simple pointer.
  decltype(auto) GetEdges(int state) const { return edges_[state]; }

  void GetEpsilonClosure(std::unordered_set<int>* state_set) const;

  void GetPossibleRules(const int& state_num, std::unordered_set<int>* rules) const;

  void GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result) const;

 protected:
  ContainerType edges_;
  friend struct member_trait<CompactFSM::Impl>;
};

template <typename ContainerType>
std::string FSMImplBase<ContainerType>::PrintEdges() const {
  std::string result = "[\n";
  for (int i = 0; i < int(NumStates()); ++i) {
    result += std::to_string(i) + ": [";
    const auto& edges = edges_[i];
    for (int j = 0; j < static_cast<int>(edges.size()); ++j) {
      const auto& edge = edges[j];
      if (edge.min == -1 && edge.max == -1) {
        result += "eps->" + std::to_string(edge.target);
      } else if (edge.min == -1 && edge.max >= 0) {
        result += "rule(" + std::to_string(edge.max) + ")->" + std::to_string(edge.target);
      } else if (edge.min == edge.max) {
        std::string char_str = PrintAsEscapedUTF8(static_cast<TCodepoint>(edge.min));
        result += "(" + char_str + ")->" + std::to_string(edge.target);
      } else {
        std::string char_min_str = PrintAsEscapedUTF8(static_cast<TCodepoint>(edge.min));
        std::string char_max_str = PrintAsEscapedUTF8(static_cast<TCodepoint>(edge.max));
        result += "(" + char_min_str + ", " + char_max_str + ")->" + std::to_string(edge.target);
      }
      if (j < static_cast<int>(edges_.size()) - 1) {
        result += ", ";
      }
    }
    result += "]\n";
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
void FSMImplBase<ContainerType>::GetPossibleRules(const int& state, std::unordered_set<int>* rules)
    const {
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
 public:
  Impl() = default;

  Impl(int num_states = 0) { edges_.resize(num_states); }

  using FSMImplBase<std::vector<std::vector<FSMEdge>>>::FSMImplBase;

  int GetNextState(int from, int16_t character) const;

  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      bool value_is_rule,
      bool from_is_closure
  ) const;

  int AddState() {
    edges_.emplace_back();
    return edges_.size() - 1;
  }

  void AddEdge(int from, int to, int16_t min_ch, int16_t max_ch) {
    XGRAMMAR_DCHECK(
        from < static_cast<int>(edges_.size()) && to <= static_cast<int>(edges_.size())
    );
    edges_[from].push_back({min_ch, max_ch, to});
  }

  void AddEpsilonEdge(int from, int to) { AddEdge(from, to, -1, -1); }

  void AddFSM(const FSM& fsm, std::unordered_map<int, int>* state_mapping);

  FSM RebuildWithMapping(std::unordered_map<int, int>& state_mapping, int new_num_states);

  CompactFSM ToCompact();
};

int FSM::Impl::GetNextState(int from, int16_t character) const {
  for (const auto& edge : edges_[from]) {
    if (edge.min != -1 && edge.min <= character && edge.max >= character) {
      return edge.target;
    }
  }
  return FSM::kNoNextState;
}

void FSM::Impl::Advance(
    const std::unordered_set<int>& from,
    int value,
    std::unordered_set<int>* result,
    bool value_is_rule,
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
  for (const auto& state : *start_closure) {
    if (value_is_rule) {
      for (const auto& edge : edges_[state]) {
        if (edge.IsRuleRef() && edge.GetRefRuleId() == value) {
          result->insert(edge.target);
        }
      }
    } else {
      for (const auto& edge : edges_[state]) {
        if (edge.IsCharRange() && edge.min <= value && edge.max >= value) {
          result->insert(edge.target);
        }
      }
    }
  }

  // Get the epsilon closure of the result.
  GetEpsilonClosure(result);
}

void FSM::Impl::AddFSM(const FSM& fsm, std::unordered_map<int, int>* state_mapping) {
  int old_num_states = NumStates();

  if (state_mapping != nullptr) {
    state_mapping->clear();
    for (int i = 0; i < fsm.NumStates(); ++i) {
      state_mapping->insert({i, i + old_num_states});
    }
  }

  edges_.resize(edges_.size() + fsm.NumStates());

  for (int i = 0; i < fsm.NumStates(); ++i) {
    for (const auto& edge : fsm.GetEdges()[i]) {
      AddEdge(i + old_num_states, edge.target + old_num_states, edge.min, edge.max);
    }
  }
}

FSM FSM::Impl::RebuildWithMapping(std::unordered_map<int, int>& state_mapping, int new_num_states) {
  std::vector<std::set<FSMEdge>> new_edges_set(new_num_states);
  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    for (const auto& edge : edges_[i]) {
      new_edges_set[state_mapping[i]].insert(FSMEdge(edge.min, edge.max, state_mapping[edge.target])
      );
    }
  }
  std::vector<std::vector<FSMEdge>> new_edges(new_num_states);
  for (int i = 0; i < new_num_states; ++i) {
    new_edges[i].insert(new_edges[i].end(), new_edges_set[i].begin(), new_edges_set[i].end());
  }
  return FSM(std::move(new_edges));
}

CompactFSM FSM::Impl::ToCompact() {
  CSRArray<FSMEdge> edges;
  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    std::sort(edges_[i].begin(), edges_[i].end());
    edges.Insert(edges_[i]);
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

void FSM::AddEdge(int from, int to, int16_t min_ch, int16_t max_ch) {
  pimpl_->AddEdge(from, to, min_ch, max_ch);
}

void FSM::AddRuleEdge(int from, int to, int16_t rule_id) {
  // Rule edges are represented as a special case of character range edges.
  pimpl_->AddEdge(from, to, -1, rule_id);
}

void FSM::AddEpsilonEdge(int from, int to) { pimpl_->AddEpsilonEdge(from, to); }

void FSM::AddFSM(const FSM& fsm, std::unordered_map<int, int>* state_mapping) {
  pimpl_->AddFSM(fsm, state_mapping);
}

std::string FSM::PrintEdges() const { return pimpl_->PrintEdges(); }

const std::vector<FSMEdge>& FSM::GetEdges(int state) const { return pimpl_->GetEdges(state); }

const std::vector<std::vector<FSMEdge>>& FSM::GetEdges() const { return pimpl_->GetEdges(); }

FSM FSM::Copy() const { return FSM(std::make_shared<Impl>(*pimpl_)); }

int FSM::GetNextState(int from, int16_t character) const {
  return pimpl_->GetNextState(from, character);
}

void FSM::Advance(
    const std::unordered_set<int>& from,
    int value,
    std::unordered_set<int>* result,
    bool value_is_rule,
    bool from_is_closure
) const {
  pimpl_->Advance(from, value, result, value_is_rule, from_is_closure);
}

void FSM::GetPossibleRules(const int& state, std::unordered_set<int>* rules) const {
  pimpl_->GetPossibleRules(state, rules);
}

void FSM::GetEpsilonClosure(std::unordered_set<int>* state_set) const {
  pimpl_->GetEpsilonClosure(state_set);
}

void FSM::GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result) const {
  pimpl_->GetReachableStates(from, result);
}

FSM FSM::RebuildWithMapping(std::unordered_map<int, int>& state_mapping, int new_num_states) {
  return pimpl_->RebuildWithMapping(state_mapping, new_num_states);
}

CompactFSM FSM::ToCompact() { return pimpl_->ToCompact(); }

/****************** CompactFSM::Impl ******************/

class CompactFSM::Impl : public FSMImplBase<CSRArray<FSMEdge>> {
 public:
  using FSMImplBase<CSRArray<FSMEdge>>::FSMImplBase;

  int GetNextState(int from, int16_t character) const;

  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      bool value_is_rule,
      bool from_is_closure
  ) const;

  FSM ToFSM() const;

  friend std::size_t MemorySize(const Impl& self) { return MemorySize(self.edges_); }
};

XGRAMMAR_MEMBER_ARRAY(CompactFSM::Impl, &CompactFSM::Impl::edges_);

int CompactFSM::Impl::GetNextState(int from, int16_t character) const {
  for (const auto& edge : edges_[from]) {
    if (edge.min == -1) {
      continue;
    } else if (edge.min <= character && edge.max >= character) {
      return edge.target;
    } else if (edge.min > character) {
      return CompactFSM::kNoNextState;
    }
  }
  return CompactFSM::kNoNextState;
}

void CompactFSM::Impl::Advance(
    const std::unordered_set<int>& from,
    int value,
    std::unordered_set<int>* result,
    bool value_is_rule,
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
  for (const auto& state : *start_closure) {
    if (value_is_rule) {
      for (const auto& edge : edges_[state]) {
        if (edge.min == -1 && edge.max == value) {
          result->insert(edge.target);
        } else if (edge.min >= 0) {
          break;
        }
      }
    } else {
      for (const auto& edge : edges_[state]) {
        if (edge.min == -1) {
          continue;
        } else if (edge.min <= value && edge.max >= value) {
          result->insert(edge.target);
        } else if (edge.min > value) {
          break;
        }
      }
    }
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

CompactFSM::CompactFSM(const CSRArray<FSMEdge>& edges) : pimpl_(std::make_shared<Impl>(edges)) {}

CompactFSM::CompactFSM(CSRArray<FSMEdge>&& edges)
    : pimpl_(std::make_shared<Impl>(std::move(edges))) {}

int CompactFSM::NumStates() const { return pimpl_->NumStates(); }

const CSRArray<FSMEdge>& CompactFSM::GetEdges() const { return pimpl_->GetEdges(); }

CSRArray<FSMEdge>::Row CompactFSM::GetEdges(int state) const { return pimpl_->GetEdges(state); }

std::string CompactFSM::PrintEdges() const { return pimpl_->PrintEdges(); }

int CompactFSM::GetNextState(int from, int16_t character) const {
  return pimpl_->GetNextState(from, character);
}

void CompactFSM::Advance(
    const std::unordered_set<int>& from,
    int value,
    std::unordered_set<int>* result,
    bool value_is_rule,
    bool from_is_closure
) const {
  pimpl_->Advance(from, value, result, value_is_rule, from_is_closure);
}

void CompactFSM::GetPossibleRules(const int& state_num, std::unordered_set<int>* rules) const {
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

std::size_t MemorySize(const CompactFSM& self) { return MemorySize(*self.pimpl_); }

/****************** FSMWithStartEnd ******************/

std::string FSMWithStartEnd::Print() const {
  std::string result;
  result += "FSM(num_states=" + std::to_string(NumStates()) + ", start=" + std::to_string(start_) +
            ", end=[";
  for (const auto& end : ends_) {
    XGRAMMAR_DCHECK(end >= 0 && end < NumStates())
        << "End state " << end << " is out of bounds for FSM with " << NumStates() << " states.";
    result += std::to_string(end) + ", ";
  }
  result += "], edges=" + fsm_.PrintEdges() + ")";
  return result;
}

FSMWithStartEnd FSMWithStartEnd::Copy() const {
  return FSMWithStartEnd(fsm_.Copy(), start_, ends_, is_dfa_);
}

std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm) {
  os << fsm.Print();
  return os;
}

FSMWithStartEnd FSMWithStartEnd::Star() const {
  FSM fsm = fsm_.Copy();
  auto new_start = fsm.AddState();
  for (const auto& end : ends_) {
    fsm.AddEpsilonEdge(end, new_start);
  }
  fsm.AddEpsilonEdge(new_start, start_);
  return FSMWithStartEnd(fsm, new_start, {new_start});
}

FSMWithStartEnd FSMWithStartEnd::Plus() const {
  FSM fsm = fsm_.Copy();
  for (const auto& end : ends_) {
    fsm.AddEpsilonEdge(end, start_);
  }
  return FSMWithStartEnd(fsm, start_, ends_);
}

FSMWithStartEnd FSMWithStartEnd::Optional() const {
  FSM fsm = fsm_.Copy();
  fsm.AddEpsilonEdge(start_, *ends_.begin());
  return FSMWithStartEnd(fsm, start_, ends_);
}

FSMWithStartEnd FSMWithStartEnd::Not() const {
  FSMWithStartEnd result = is_dfa_ ? Copy() : ToDFA();
  int state_cnt = result.NumStates();
  // Reverse all the final states.
  std::unordered_set<int> final_states;
  for (int i = 0; i < result->NumStates(); ++i) {
    if (!result.IsEndState(i)) {
      final_states.insert(i);
    }
  }

  // Add all the rules in the alphabet.
  std::unordered_set<int> rules;
  for (const auto& edges : result->GetEdges()) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        rules.insert(edge.GetRefRuleId());
      }
    }
  }

  // Add a new state to avoid the blocking.
  result->AddState();
  final_states.insert(result.NumStates() - 1);
  for (auto rule : rules) {
    result->AddRuleEdge(result.NumStates() - 1, result.NumStates() - 1, rule);
  }
  result->AddEdge(result.NumStates() - 1, result.NumStates() - 1, 0, 0xFF);
  result.AddEndState(result.NumStates() - 1);

  for (int i = 0; i < result.NumStates(); i++) {
    const auto& state_edges = result->GetEdges(i);
    std::vector<bool> char_has_edges(0x100, false);
    std::unordered_set<int> rule_has_edges;
    for (const auto& edge : state_edges) {
      if (edge.IsCharRange()) {
        for (int i = edge.min; i <= edge.max; ++i) {
          char_has_edges[i] = true;
        }
      }
      if (edge.IsRuleRef()) {
        rule_has_edges.insert(edge.GetRefRuleId());
      }
    }

    // Add the left characters to the new state.
    int interval_start = -1;
    for (int j = 0; j < 0x100; ++j) {
      if (!char_has_edges[j]) {
        // The char doesn't have any edges. Thus, we can accept it in the
        // complement FSM.
        if (interval_start == -1) {
          interval_start = j;
        }
      } else {
        if (interval_start != -1) {
          // state_cnt is the state to accept all such characters.
          result->AddEdge(i, state_cnt, interval_start, j - 1);
          interval_start = -1;
        }
      }
    }
    if (interval_start != -1) {
      result->AddEdge(i, state_cnt, interval_start, 0xFF);
    }

    // Add the left rules to the new state.
    for (auto rule : rules) {
      if (rule_has_edges.find(rule) == rule_has_edges.end()) {
        result->AddRuleEdge(result.NumStates() - 1, state_cnt, rule);
      }
    }
  }
  result.SetEndStates(final_states);
  return result;
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
  std::unordered_set<int> ends;

  std::unordered_map<int, int> state_mapping;

  for (const auto& fsm_with_se : fsms) {
    fsm.AddFSM(fsm_with_se.GetFSM(), &state_mapping);
    fsm.AddEpsilonEdge(start, state_mapping[fsm_with_se.GetStart()]);
    for (const auto& end : fsm_with_se.GetEnds()) {
      ends.insert(state_mapping[end]);
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
  std::unordered_set<int> ends;

  std::unordered_map<int, int> state_mapping;
  std::vector<int> previous_ends;

  for (int i = 0; i < static_cast<int>(fsms.size()); ++i) {
    fsm.AddFSM(fsms[i].GetFSM(), &state_mapping);
    if (i == 0) {
      start = state_mapping[fsms[i].GetStart()];
    } else {
      auto this_start = state_mapping[fsms[i].GetStart()];
      for (const auto& end : previous_ends) {
        fsm.AddEpsilonEdge(end, this_start);
      }
    }
    if (i == static_cast<int>(fsms.size()) - 1) {
      for (const auto& end : fsms[i].GetEnds()) {
        ends.insert(state_mapping[end]);
      }
    } else {
      previous_ends.clear();
      previous_ends.reserve(fsms[i].GetEnds().size());
      for (const auto& end : fsms[i].GetEnds()) {
        previous_ends.push_back(state_mapping[end]);
      }
    }
  }

  return FSMWithStartEnd(fsm, start, ends);
}

Result<FSMWithStartEnd> FSMWithStartEnd::Intersect(
    const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs, const int& num_of_states_limited
) {
  if (!lhs.IsLeaf() || !rhs.IsLeaf()) {
    return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Intersect only support leaf fsm!")
    );
  }
  auto lhs_dfa = lhs.ToDFA();
  auto rhs_dfa = rhs.ToDFA();
  std::unordered_set<int> rules_lhs;
  std::unordered_set<int> rules;
  std::set<int> interval_ends;
  std::vector<std::pair<int, int>> intervals;
  // This part is to build the equivalent alphabet.
  for (const auto& edges : lhs_dfa->GetEdges()) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        rules_lhs.insert(edge.GetRefRuleId());
      } else if (edge.IsCharRange()) {
        interval_ends.insert(edge.min);
        interval_ends.insert(edge.max + 1);
      }
    }
  }
  for (const auto& edges : rhs_dfa->GetEdges()) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        if (rules_lhs.find(edge.GetRefRuleId()) != rules_lhs.end()) {
          rules.insert(edge.GetRefRuleId());
        }
      } else if (edge.IsCharRange()) {
        interval_ends.insert(edge.min);
        interval_ends.insert(edge.max + 1);
      }
    }
  }
  for (auto it = interval_ends.begin(); it != interval_ends.end(); ++it) {
    auto next_it = std::next(it);
    if (next_it != interval_ends.end()) {
      intervals.emplace_back(*it, *next_it - 1);
    }
  }

  // Initialize the result FSM.
  FSM result_fsm(0);
  FSMWithStartEnd result(result_fsm, 0, {}, true);
  std::unordered_map<std::pair<int, int>, int> state_map;
  std::unordered_set<std::pair<int, int>> visited;
  std::queue<std::pair<int, int>> queue;
  queue.push({lhs_dfa.GetStart(), rhs_dfa.GetStart()});
  result->AddState();
  state_map[{lhs_dfa.GetStart(), rhs_dfa.GetStart()}] = 0;
  while (!queue.empty()) {
    if (int(state_map.size()) > num_of_states_limited) {
      return Result<FSMWithStartEnd>::Err(
          std::make_shared<Error>("Intersection have too many states!")
      );
    }
    auto state = queue.front();
    queue.pop();
    if (visited.find(state) != visited.end()) {
      continue;
    }
    visited.insert(state);
    int lhs_state = state.first;
    int rhs_state = state.second;
    for (const auto& interval : intervals) {
      for (const auto& lhs_edge : lhs_dfa->GetEdges(lhs_state)) {
        if (!lhs_edge.IsCharRange()) {
          continue;
        }
        if (lhs_edge.min > interval.first || lhs_edge.max < interval.second) {
          continue;
        }
        for (const auto& rhs_edge : rhs_dfa->GetEdges(rhs_state)) {
          if (!rhs_edge.IsCharRange()) {
            continue;
          }
          if (rhs_edge.min > interval.first || rhs_edge.max < interval.second) {
            continue;
          }
          auto next_state = std::make_pair(lhs_edge.target, rhs_edge.target);
          if (state_map.find(next_state) == state_map.end()) {
            state_map[next_state] = state_map.size();
            queue.push(next_state);
            result->AddState();
          }
          result->AddEdge(
              state_map[{lhs_state, rhs_state}],
              state_map[next_state],
              interval.first,
              interval.second
          );
          break;
        }
      }
    }
    for (const auto& rule : rules) {
      for (const auto& lhs_edge : lhs_dfa->GetEdges(lhs_state)) {
        if (!lhs_edge.IsRuleRef()) {
          continue;
        }
        if (lhs_edge.GetRefRuleId() != rule) {
          continue;
        }
        for (const auto& rhs_edge : rhs_dfa->GetEdges(rhs_state)) {
          if (!rhs_edge.IsRuleRef()) {
            continue;
          }
          if (rhs_edge.GetRefRuleId() != rule) {
            continue;
          }
          auto next_state = std::make_pair(lhs_edge.target, rhs_edge.target);
          if (state_map.find(next_state) == state_map.end()) {
            state_map[next_state] = state_map.size();
            queue.push(next_state);
            result->AddState();
          }
          result->AddRuleEdge(state_map[{lhs_state, rhs_state}], state_map[next_state], rule);
          break;
        }
      }
    }
  }
  for (const auto& state : visited) {
    if (lhs_dfa.IsEndState(state.first) && rhs_dfa.IsEndState(state.second)) {
      result.AddEndState(state_map[state]);
    }
  }
  return Result<FSMWithStartEnd>::Ok(result);
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

FSMWithStartEnd FSMWithStartEnd::SimplifyEpsilon() const {
  if (is_dfa_) {
    return *this;
  }
  UnionFindSet<int> union_find_set;
  std::unordered_map<int, std::unordered_set<int>> previous_states;
  std::unordered_set<int> has_epsilon;

  // Initialize the previous states, and find all the states that have
  // epsilon edges.
  for (int i = 0; i < NumStates(); i++) {
    const auto& edges = fsm_->GetEdges(i);
    for (const auto& edge : edges) {
      if (previous_states.find(edge.target) == previous_states.end()) {
        previous_states[edge.target] = std::unordered_set<int>();
      }
      previous_states[edge.target].insert(i);
      if (edge.IsEpsilon()) {
        if (edges.size() != 1 || edge.target == GetStart()) {
          has_epsilon.insert(i);
        } else {
          // a -- epsilon --> b, and a doesn't have other outward edges.
          union_find_set.Make(i);
          union_find_set.Make(edge.target);
          union_find_set.Union(i, edge.target);
        }
      }
    }
  }

  // a --> epsilon --> b, and b doesn't have other inward edges.
  for (const auto& state : has_epsilon) {
    const auto& edges = fsm_->GetEdges(state);
    for (const auto& edge : edges) {
      if (!edge.IsEpsilon()) {
        continue;
      }
      // Have other inward states.
      if (previous_states[edge.target].size() != 1 || edge.target == GetStart()) {
        continue;
      }
      bool has_other_edge = false;
      for (const auto& second_edge : edges) {
        if (second_edge.IsEpsilon()) {
          continue;
        }
        if (second_edge.target == edge.target) {
          has_other_edge = true;
          break;
        }
      }
      // The state can be merged.
      if (!has_other_edge) {
        union_find_set.Make(state);
        union_find_set.Make(edge.target);
        union_find_set.Union(state, edge.target);
      }
    }
  }

  // Merge the states.
  auto eq_classes = union_find_set.GetAllSets();
  if (eq_classes.empty()) {
    return *this;
  }

  std::unordered_map<int, int> new_to_old;
  for (size_t i = 0; i < eq_classes.size(); i++) {
    for (const auto& state : eq_classes[i]) {
      new_to_old[state] = i;
    }
  }
  int cnt = eq_classes.size();
  for (int i = 0; i < NumStates(); i++) {
    if (new_to_old.find(i) == new_to_old.end()) {
      new_to_old[i] = cnt;
      cnt++;
    }
  }
  FSMWithStartEnd result = Copy();
  return result.RebuildWithMapping(new_to_old, cnt);
}

FSMWithStartEnd FSMWithStartEnd::MergeEquivalentSuccessors() const {
  bool changed = true;
  FSMWithStartEnd result = Copy();
  UnionFindSet<int> union_find_set;
  while (changed) {
    union_find_set.Clear();
    std::unordered_map<int, std::unordered_set<int>> previous_states;
    // Initialize the previous states.
    for (int i = 0; i < result->NumStates(); i++) {
      const auto& edges = result->GetEdges(i);
      for (const auto& edge : edges) {
        if (previous_states.find(edge.target) == previous_states.end()) {
          previous_states[edge.target] = std::unordered_set<int>();
        }
        previous_states[edge.target].insert(i);
      }
    }
    // Case 1: Like ab | ac | ad, then they can be merged into a(b | c | d).
    changed = false;
    bool change_case1 = false;
    for (const auto& edges : result->GetEdges()) {
      for (size_t i = 0; i < edges.size(); i++) {
        for (size_t j = i + 1; j < edges.size(); j++) {
          if (IsEndState(edges[i].target) != IsEndState(edges[j].target)) {
            continue;
          }
          if (edges[i].target == edges[j].target) {
            continue;
          }
          if (edges[i].max != edges[j].max || edges[i].min != edges[j].min) {
            continue;
          }
          if (previous_states[edges[i].target].size() != 1 ||
              previous_states[edges[j].target].size() != 1) {
            continue;
          }
          union_find_set.Make(edges[i].target);
          union_find_set.Make(edges[j].target);
          union_find_set.Union(edges[i].target, edges[j].target);
          change_case1 = true;
        }
      }
    }
    if (change_case1) {
      auto eq_classes = union_find_set.GetAllSets();
      std::unordered_map<int, int> old_to_new;
      for (size_t i = 0; i < eq_classes.size(); i++) {
        for (const auto& state : eq_classes[i]) {
          old_to_new[state] = i;
        }
      }
      int cnt = eq_classes.size();
      for (int i = 0; i < result->NumStates(); i++) {
        if (old_to_new.find(i) == old_to_new.end()) {
          old_to_new[i] = cnt;
          cnt++;
        }
      }
      result = result.RebuildWithMapping(old_to_new, cnt);
    }
    union_find_set.Clear();
    // Case 2: Like ba | ca | da, then they can be merged into (b | c | d)a.
    bool change_case2 = false;
    for (int i = 0; i < result->NumStates(); i++) {
      for (int j = i + 1; j < result->NumStates(); j++) {
        bool equivalent = true;
        for (const auto& edge_i : result->GetEdges(i)) {
          bool same = false;
          for (const auto& edge_j : result->GetEdges(j)) {
            if (edge_i.min == edge_j.min && edge_i.max == edge_j.max &&
                edge_i.target == edge_j.target) {
              same = true;
              break;
            }
          }
          if (!same) {
            equivalent = false;
            break;
          }
        }
        if (!equivalent) {
          continue;
        }
        for (const auto& edge_j : result->GetEdges(j)) {
          bool same = false;
          for (const auto& edge_i : result->GetEdges(i)) {
            if (edge_i.min == edge_j.min && edge_i.max == edge_j.max &&
                edge_i.target == edge_j.target) {
              same = true;
              break;
            }
          }
          if (!same) {
            equivalent = false;
            break;
          }
        }
        if (equivalent) {
          union_find_set.Make(i);
          union_find_set.Make(j);
          union_find_set.Union(i, j);
          change_case2 = true;
        }
      }
    }
    if (change_case2) {
      auto eq_classes = union_find_set.GetAllSets();
      std::unordered_map<int, int> old_to_new;
      for (size_t i = 0; i < eq_classes.size(); i++) {
        for (const auto& state : eq_classes[i]) {
          old_to_new[state] = i;
        }
      }
      int cnt = eq_classes.size();
      for (int i = 0; i < result->NumStates(); i++) {
        if (old_to_new.find(i) == old_to_new.end()) {
          old_to_new[i] = cnt;
          cnt++;
        }
      }
      result = result.RebuildWithMapping(old_to_new, cnt);
    }
    changed = change_case1 || change_case2;
  }
  return result;
}

FSMWithStartEnd FSMWithStartEnd::MinimizeDFA() const {
  FSMWithStartEnd now_fsm(FSM(0), 0, {}, true);

  // To perform the algorithm, we must make sure the FSM is
  // a DFA.
  if (!is_dfa_) {
    now_fsm = ToDFA();
  } else {
    now_fsm = Copy();
  }
  // Initialize the set.
  std::list<std::unordered_set<int>> blocks;
  std::list<std::unordered_set<int>> queue;
  std::unordered_set<int> not_end;
  for (int i = 0; i < now_fsm->NumStates(); i++) {
    if (!now_fsm.IsEndState(i)) {
      not_end.insert(i);
    }
  }
  queue.push_back(not_end);
  queue.push_back(now_fsm.GetEnds());
  blocks.push_back(not_end);
  blocks.push_back(now_fsm.GetEnds());
  std::set<int> interval_ends;
  std::unordered_set<std::pair<int, int>> intervals;
  std::unordered_set<int> rules;
  std::unordered_map<int, std::unordered_set<int>> previous_mapping;
  for (int i = 0; i < now_fsm->NumStates(); i++) {
    const auto& edges = now_fsm->GetEdges(i);
    for (const auto& edge : edges) {
      if (previous_mapping.find(edge.target) == previous_mapping.end()) {
        previous_mapping[edge.target] = std::unordered_set<int>();
      }
      previous_mapping[edge.target].insert(i);
      if (edge.IsCharRange()) {
        interval_ends.insert(edge.min);
        interval_ends.insert(edge.max + 1);
        continue;
      }
      if (edge.IsRuleRef()) {
        rules.insert(edge.GetRefRuleId());
      }
    }
  }
  for (auto it = interval_ends.begin(); it != interval_ends.end(); ++it) {
    auto next_it = std::next(it);
    if (next_it != interval_ends.end()) {
      intervals.insert(std::make_pair(*it, *next_it - 1));
    }
  }

  while (!queue.empty()) {
    // Initial the alphabet.
    auto block_x = *queue.begin();
    queue.erase(queue.begin());
    std::unordered_set<int> prev_nodes;
    for (const auto& node : block_x) {
      if (previous_mapping.find(node) != previous_mapping.end()) {
        prev_nodes.insert(previous_mapping[node].begin(), previous_mapping[node].end());
      }
    }
    // Check the intervals.
    std::list<std::unordered_set<int>> blocks_copy = blocks;
    for (const auto& interval : intervals) {
      std::unordered_set<int> from_block;
      for (const auto& node : prev_nodes) {
        const auto& edges = now_fsm->GetEdges(node);
        for (const auto& edge : edges) {
          if (block_x.find(edge.target) == block_x.end()) {
            continue;
          }
          if (edge.IsCharRange()) {
            if (interval.first >= edge.min && interval.second <= edge.max) {
              from_block.insert(node);
            }
          }
        }
      }
      for (const auto& block : blocks_copy) {
        std::unordered_set<int> intersection;
        for (const auto& prev : from_block) {
          if (block.find(prev) != block.end()) {
            intersection.insert(prev);
          }
        }
        // The intersection is empty, or the intersection == block.
        if (intersection.empty() || intersection.size() == block.size()) {
          continue;
        }
        std::unordered_set<int> difference;
        for (const auto& node : block) {
          if (intersection.find(node) == intersection.end()) {
            difference.insert(node);
          }
        }
        blocks.remove(block);
        blocks.remove(intersection);
        blocks.remove(difference);
        blocks.push_back(intersection);
        blocks.push_back(difference);
        bool found = false;
        for (auto iter = queue.begin(); iter != queue.end(); ++iter) {
          if (*iter == block) {
            found = true;
            break;
          }
        }
        if (found) {
          queue.remove(block);
          queue.push_back(intersection);
          queue.push_back(difference);
        } else {
          queue.push_back(intersection.size() < difference.size() ? intersection : difference);
        }
      }
    }
    // Do the same thing for the rules.
    blocks_copy = blocks;
    for (const auto& rule : rules) {
      std::unordered_set<int> from_block;
      for (const auto& node : prev_nodes) {
        const auto& edges = now_fsm->GetEdges(node);
        for (const auto& edge : edges) {
          if (block_x.find(edge.target) == block_x.end()) {
            continue;
          }
          if (edge.IsRuleRef()) {
            if (rule == edge.GetRefRuleId()) {
              from_block.insert(node);
            }
          }
        }
      }
      for (const auto& block : blocks_copy) {
        std::unordered_set<int> intersection;
        for (const auto& prev : from_block) {
          if (block.find(prev) != block.end()) {
            intersection.insert(prev);
          }
        }
        // The intersection is empty, or the intersection == block.
        if (intersection.empty() || intersection.size() == block.size()) {
          continue;
        }
        std::unordered_set<int> difference;
        for (const auto& node : from_block) {
          if (intersection.find(node) == intersection.end()) {
            difference.insert(node);
          }
        }
        blocks.remove(block);
        blocks.remove(intersection);
        blocks.remove(difference);
        blocks.push_back(intersection);
        blocks.push_back(difference);
        bool found = false;
        for (auto iter = queue.begin(); iter != queue.end(); ++iter) {
          if (*iter == block) {
            found = true;
            break;
          }
        }
        if (found) {
          queue.remove(block);
          queue.push_back(intersection);
          queue.push_back(difference);
        } else {
          queue.push_back(intersection.size() < difference.size() ? intersection : difference);
        }
      }
    }
  }

  std::unordered_map<int, int> old_to_new;
  int cnt = 0;
  for (const auto& block : blocks) {
    for (const auto& node : block) {
      old_to_new[node] = cnt;
    }
    cnt++;
  }
  FSMWithStartEnd new_fsm(FSM(0), old_to_new[now_fsm.GetStart()], {}, true);
  for (int i = 0; i < cnt; i++) {
    new_fsm->AddState();
  }
  for (const auto& end : now_fsm.GetEnds()) {
    new_fsm.AddEndState(old_to_new[end]);
  }
  std::unordered_set<int> been_built;
  for (int i = 0; i < now_fsm->NumStates(); i++) {
    if (been_built.find(old_to_new[i]) != been_built.end()) {
      continue;
    }
    been_built.insert(old_to_new[i]);
    for (const auto& edge : now_fsm->GetEdges(i)) {
      new_fsm->AddEdge(old_to_new[i], old_to_new[edge.target], edge.min, edge.max);
    }
  }
  return new_fsm;
}

FSMWithStartEnd FSMWithStartEnd::ToDFA() const {
  FSMWithStartEnd dfa(FSM(0), 0, {}, true);
  std::vector<std::unordered_set<int>> closures;
  std::unordered_set<int> rules;
  for (const auto& edges : fsm_->GetEdges()) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        rules.insert(edge.GetRefRuleId());
      }
    }
  }
  int now_process = 0;
  std::unordered_set<int> closure;
  closure.insert(start_);
  fsm_.GetEpsilonClosure(&closure);
  closures.push_back(closure);
  while (now_process < static_cast<int>(closures.size())) {
    std::set<int> interval_ends;
    dfa->AddState();
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
          continue;
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
      intervals.emplace_back(last, end - 1);
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
          dfa->AddEdge(now_process, j, interval.first, interval.second);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa->AddEdge(now_process, closures.size(), interval.first, interval.second);
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
          dfa->AddRuleEdge(now_process, j, rule);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa->AddRuleEdge(now_process, closures.size(), rule);
        closures.push_back(next_closure);
      }
    }
    now_process++;
  }
  return dfa;
}

FSMWithStartEnd FSMWithStartEnd::RebuildWithMapping(
    std::unordered_map<int, int>& state_mapping, int new_num_states
) {
  FSM new_fsm = fsm_.RebuildWithMapping(state_mapping, new_num_states);
  auto new_start = state_mapping[start_];
  std::unordered_set<int> new_ends;
  for (const auto& end : ends_) {
    new_ends.insert(state_mapping[end]);
  }

  return FSMWithStartEnd(new_fsm, new_start, new_ends);
}

CompactFSMWithStartEnd FSMWithStartEnd::ToCompact() {
  return CompactFSMWithStartEnd(fsm_.ToCompact(), start_, ends_);
}

/****************** CompactFSMWithStartEnd ******************/

std::string CompactFSMWithStartEnd::Print() const {
  std::string result;
  result += "CompactFSM(num_states=" + std::to_string(NumStates()) +
            ", start=" + std::to_string(start_) + ", end=[";
  for (const auto& end : ends_) {
    result += std::to_string(end) + ", ";
  }
  result += "], edges=" + fsm_.PrintEdges() + ")";
  return result;
}

std::ostream& operator<<(std::ostream& os, const CompactFSMWithStartEnd& fsm) {
  os << fsm.Print();
  return os;
}

std::size_t MemorySize(const CompactFSMWithStartEnd& self) {
  return MemorySize(self.fsm_) + MemorySize(self.ends_);
}

FSMWithStartEnd CompactFSMWithStartEnd::ToFSM() const {
  return FSMWithStartEnd(fsm_.ToFSM(), start_, ends_);
}

picojson::value CompactFSM::SerializeJSONValue() const { return AutoSerializeJSONValue(**this); }

void DeserializeJSONValue(CompactFSM& fsm, const picojson::value& v) {
  if (!fsm.pimpl_) {
    fsm.pimpl_ = std::make_unique<CompactFSM::Impl>();
  }
  return AutoDeserializeJSONValue(*fsm, v);
}

}  // namespace xgrammar
