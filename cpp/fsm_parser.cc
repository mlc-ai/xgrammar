#include "fsm_parser.h"

#include <cstdint>
#include <vector>

#include "fsm.h"
#include "support/csr_array.h"
#include "support/logging.h"

namespace xgrammar {

bool EarleyParserWithFSM::Advance(uint8_t ch) {
  tmp_states_visited_in_queue_.clear();
  tmp_states_to_be_added_.clear();
  tmp_accept_stop_token_ = false;
  const auto& latest_states = scanable_state_history_[scanable_state_history_.Size() - 1];

  // Scan the current states.
  Scan(latest_states, ch);

  if (tmp_process_state_queue_.empty()) {
    return false;
  }

  XGRAMMAR_LOG(FATAL) << "Not implemented yet";
  // TODO(linzhang): implement the rest of the Earley parser
}

void EarleyParserWithFSM::Scan(const CSRArray<FSMState>::Row& current_states, uint8_t ch) {
  std::vector<int> tmp_input_container;
  std::vector<int> tmp_next_states;
  tmp_input_container.resize(1);

  for (const auto& state : current_states) {
    // Check the fsm_id is valid.
    XGRAMMAR_DCHECK(state.fsm_id >= 0 && state.fsm_id < fsms_.size())
        << "The fsm id is out of bound. The fsm id is " << state.fsm_id;

    // Advance the FSM. The current states are an epsilon closure, which is guaranteed by the
    // last step.
    const auto& fsm = fsms_[state.fsm_id];
    fsm.fsm.Advance(tmp_input_container, ch, &tmp_next_states, true, false);
    for (const auto& next_node : tmp_next_states) {
      FSMState new_state(state.fsm_id, next_node, state.input_pos);
      Enque(new_state);
    }
  }
}

bool EarleyParserWithFSM::Predict(const FSMState& state) {
  bool scanable = false;
  XGRAMMAR_DCHECK(state.fsm_id >= 0 && state.fsm_id < fsms_.size())
      << "The fsm id is out of bound. The fsm id is " << state.fsm_id;
  XGRAMMAR_DCHECK(state.node_id >= 0 && state.node_id < fsms_[state.fsm_id].fsm.edges.size())
      << "The node id is out of bound. The node id is " << state.node_id;
  const auto& state_edges = fsms_[state.fsm_id].fsm.edges[state.node_id];
  for (const auto& edge : state_edges) {
    // Check if the state can predict a new rule.
    if (!edge.IsRuleRef()) {
      if (edge.IsCharRange()) {
        scanable = true;
      }
      continue;
    }
    // The state can predict a new rule. Thus, we need to check if the rule has
    // been predicted before. If not, then we need to expand the rule, i.e. get the
    // epsilon closure of the start state.
    // TODO(linzhang): implement the next predict.
  }
  return scanable;
}

}  // namespace xgrammar
