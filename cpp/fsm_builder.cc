/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.cc
 */
#include "fsm_builder.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "fsm.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

namespace {

constexpr TCodepoint kMaxUnicodeCodepoint = 0x10FFFF;
constexpr TCodepoint kHighSurrogateStart = 0xD800;
constexpr TCodepoint kLowSurrogateEnd = 0xDFFF;
constexpr uint8_t kContinuationByteMin = 0x80;
constexpr uint8_t kContinuationByteMax = 0xBF;

using UTF8Bytes = std::array<uint8_t, 4>;

struct EncodedCodepoint {
  UTF8Bytes bytes{};
  int length = 0;
};

EncodedCodepoint EncodeCodepoint(TCodepoint codepoint) {
  std::string utf8 = CharToUTF8(codepoint);
  EncodedCodepoint result;
  result.length = static_cast<int>(utf8.size());
  for (int index = 0; index < result.length; ++index) {
    result.bytes[index] = static_cast<uint8_t>(utf8[index]);
  }
  return result;
}

void AddAnyContinuationBytes(FSMWithStartEnd* fsm, int from, int to, int remaining_byte_count) {
  int current_state = from;
  for (int index = 0; index < remaining_byte_count; ++index) {
    int next_state = index + 1 == remaining_byte_count ? to : fsm->AddState();
    fsm->GetFsm().AddEdge(current_state, next_state, kContinuationByteMin, kContinuationByteMax);
    current_state = next_state;
  }
}

void AddUTF8SequenceRange(
    FSMWithStartEnd* fsm,
    int from,
    int to,
    const UTF8Bytes& minimum,
    const UTF8Bytes& maximum,
    int byte_count,
    int byte_index = 0
) {
  if (byte_index + 1 == byte_count) {
    fsm->GetFsm().AddEdge(from, to, minimum[byte_index], maximum[byte_index]);
    return;
  }

  if (minimum[byte_index] == maximum[byte_index]) {
    int next_state = fsm->AddState();
    fsm->GetFsm().AddEdge(from, next_state, minimum[byte_index], maximum[byte_index]);
    AddUTF8SequenceRange(fsm, next_state, to, minimum, maximum, byte_count, byte_index + 1);
    return;
  }

  UTF8Bytes lower_maximum = minimum;
  std::fill(
      lower_maximum.begin() + byte_index + 1,
      lower_maximum.begin() + byte_count,
      kContinuationByteMax
  );
  int lower_state = fsm->AddState();
  fsm->GetFsm().AddEdge(from, lower_state, minimum[byte_index], minimum[byte_index]);
  AddUTF8SequenceRange(fsm, lower_state, to, minimum, lower_maximum, byte_count, byte_index + 1);

  if (minimum[byte_index] + 1 < maximum[byte_index]) {
    int middle_state = fsm->AddState();
    fsm->GetFsm().AddEdge(from, middle_state, minimum[byte_index] + 1, maximum[byte_index] - 1);
    AddAnyContinuationBytes(fsm, middle_state, to, byte_count - byte_index - 1);
  }

  UTF8Bytes upper_minimum = maximum;
  std::fill(
      upper_minimum.begin() + byte_index + 1,
      upper_minimum.begin() + byte_count,
      kContinuationByteMin
  );
  int upper_state = fsm->AddState();
  fsm->GetFsm().AddEdge(from, upper_state, maximum[byte_index], maximum[byte_index]);
  AddUTF8SequenceRange(fsm, upper_state, to, upper_minimum, maximum, byte_count, byte_index + 1);
}

void AddCodepointRange(
    FSMWithStartEnd* fsm, int from, int to, TCodepoint minimum, TCodepoint maximum
) {
  static constexpr std::array<std::pair<TCodepoint, TCodepoint>, 5> kValidCodepointIntervals = {
      std::pair<TCodepoint, TCodepoint>{0x000000, 0x00007F},
      {0x000080, 0x0007FF},
      {0x000800, 0x00D7FF},
      {0x00E000, 0x00FFFF},
      {0x010000, kMaxUnicodeCodepoint}
  };

  for (const auto& [interval_minimum, interval_maximum] : kValidCodepointIntervals) {
    TCodepoint range_minimum = std::max(minimum, interval_minimum);
    TCodepoint range_maximum = std::min(maximum, interval_maximum);
    if (range_minimum > range_maximum) {
      continue;
    }
    EncodedCodepoint encoded_minimum = EncodeCodepoint(range_minimum);
    EncodedCodepoint encoded_maximum = EncodeCodepoint(range_maximum);
    XGRAMMAR_DCHECK(encoded_minimum.length == encoded_maximum.length);
    AddUTF8SequenceRange(
        fsm, from, to, encoded_minimum.bytes, encoded_maximum.bytes, encoded_minimum.length
    );
  }
}

Result<std::vector<TCodepoint>> ParseRegexCodepoints(const std::string& regex) {
  std::vector<TCodepoint> result;
  for (size_t offset = 0; offset < regex.size();) {
    auto [codepoint, byte_count] = ParseNextUTF8(regex.c_str() + offset);
    if (codepoint == CharHandlingError::kInvalidUTF8 || byte_count <= 0 ||
        offset + byte_count > regex.size()) {
      return ResultErr("The regex is not a valid UTF-8 string.");
    }
    int canonical_byte_count = codepoint <= 0x7F     ? 1
                               : codepoint <= 0x7FF  ? 2
                               : codepoint <= 0xFFFF ? 3
                                                     : 4;
    if (byte_count != canonical_byte_count || codepoint > kMaxUnicodeCodepoint ||
        (codepoint >= kHighSurrogateStart && codepoint <= kLowSurrogateEnd)) {
      return ResultErr("The regex contains an invalid Unicode codepoint.");
    }
    result.push_back(codepoint);
    offset += byte_count;
  }
  return ResultOk(std::move(result));
}

}  // namespace

class RegexIR {
 public:
  struct Leaf;

  struct Symbol;

  struct Union;

  struct Bracket;

  struct Repeat;

  static constexpr int kRepeatNoUpperBound = -1;

  using State = std::variant<Leaf, Symbol, Union, Bracket, Repeat>;

  struct CodepointRange {
    TCodepoint minimum;
    TCodepoint maximum;
  };

  // An atomic regex element matching exactly one Unicode codepoint.
  struct Leaf {
    std::vector<CodepointRange> ranges;
  };

  // This struct is used to store the symbol in regex, i.e.
  // +, *, ?
  enum class RegexSymbol {
    star,
    plus,
    optional,
  };

  struct Bracket {
    std::vector<State> states;
  };

  struct Symbol {
    RegexSymbol symbol;
    std::vector<State> state;
  };

  // This struct is used to represent a union symbol.
  struct Union {
    std::vector<State> states;
  };

  struct Repeat {
    std::vector<State> states;
    int lower_bound = 0;
    int upper_bound = 0;
  };

  struct LookAhead {
    bool is_positive;
    std::vector<State> states;
  };

  // This struct is used to represent a bracket in regex.
  std::vector<State> states;

  /*!
    \brief Constructs a NFA from the regex IR.
  */
  Result<FSMWithStartEnd> Build() const;

  /*!
    \brief the visit function for the variant.
  */
  Result<FSMWithStartEnd> visit(const Leaf& state) const;

  Result<FSMWithStartEnd> visit(const Symbol& state) const;

  Result<FSMWithStartEnd> visit(const Union& state) const;

  Result<FSMWithStartEnd> visit(const Bracket& state) const;

  Result<FSMWithStartEnd> visit(const Repeat& state) const;

  Result<FSMWithStartEnd> visit(const LookAhead& state) const;

 private:
  /*!
   * \brief Construct an FSM from a regex leaf.
   * \param leaf The Unicode codepoint ranges accepted by the leaf.
   * \return The FSM with start and end states.
   */
  static FSMWithStartEnd BuildLeafFSM(const Leaf& leaf);

  /*!
   * \brief Parse an escape sequence into a regex leaf.
   */
  static Result<Leaf> ParseEscape(
      const std::vector<TCodepoint>& regex, int* index, bool in_character_class
  );

  /*!
   * \brief Parse a character class into a regex leaf.
   */
  static Result<Leaf> ParseCharacterClass(const std::vector<TCodepoint>& regex, int* index);

  static std::vector<CodepointRange> NormalizeRanges(std::vector<CodepointRange> ranges);

  static std::vector<CodepointRange> NegateRanges(std::vector<CodepointRange> ranges);

  /*!
   * \brief Check repeat in regex. i.e {...} and {...,...}
   * \param regex The regex string.
   * \param start The start position of the repeat. i.e. regex[start] == '{'.
   * After the function, start will be the position of '}'.
   * \return The repeat range.
   */
  static Result<std::pair<int, int>> CheckRepeat(const std::vector<TCodepoint>& regex, int& start);

  friend class RegexFSMBuilder;
};

Result<std::pair<int, int>> RegexIR::CheckRepeat(const std::vector<TCodepoint>& regex, int& start) {
  if (regex[start] != '{') {
    return ResultErr("Invalid repeat format1");
  }
  int lower_bound = 0;
  int upper_bound = RegexIR::kRepeatNoUpperBound;
  XGRAMMAR_DCHECK(regex[start] == '{');
  start++;
  while (start < static_cast<int>(regex.size()) && regex[start] == ' ') {
    start++;
  }
  bool has_lower_bound = false;
  while (start < static_cast<int>(regex.size()) && regex[start] >= '0' && regex[start] <= '9') {
    has_lower_bound = true;
    if (lower_bound > (std::numeric_limits<int>::max() - (regex[start] - '0')) / 10) {
      return ResultErr("Repeat lower bound is too large");
    }
    lower_bound = lower_bound * 10 + regex[start] - '0';
    start++;
  }
  if (!has_lower_bound) {
    return ResultErr("Invalid repeat format2");
  }
  while (start < static_cast<int>(regex.size()) && regex[start] == ' ') {
    start++;
  }
  if (start >= static_cast<int>(regex.size())) {
    return ResultErr("Invalid repeat format");
  }
  // The format is {n}
  if (regex[start] == '}') {
    upper_bound = lower_bound;
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  if (regex[start] != ',') {
    return ResultErr("Invalid repeat format3");
  }
  XGRAMMAR_DCHECK(regex[start] == ',');
  start++;
  while (start < static_cast<int>(regex.size()) && regex[start] == ' ') {
    start++;
  }
  if (start >= static_cast<int>(regex.size())) {
    return ResultErr("Invalid repeat format");
  }
  // The format is {n,}
  if (regex[start] == '}') {
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  bool has_upper_bound = false;
  upper_bound = 0;
  while (start < static_cast<int>(regex.size()) && regex[start] >= '0' && regex[start] <= '9') {
    has_upper_bound = true;
    if (upper_bound > (std::numeric_limits<int>::max() - (regex[start] - '0')) / 10) {
      return ResultErr("Repeat upper bound is too large");
    }
    upper_bound = upper_bound * 10 + regex[start] - '0';
    start++;
  }
  if (!has_upper_bound) {
    return ResultErr("Invalid repeat format4");
  }
  if (upper_bound < lower_bound) {
    return ResultErr("Invalid repeat: the lower bound is larger than the upper bound");
  }
  while (start < static_cast<int>(regex.size()) && regex[start] == ' ') {
    start++;
  }
  if (start >= static_cast<int>(regex.size()) || regex[start] != '}') {
    return ResultErr("Invalid repeat format5");
  }
  XGRAMMAR_DCHECK(regex[start] == '}');
  return ResultOk(std::make_pair(lower_bound, upper_bound));
}

Result<FSMWithStartEnd> RegexIR::Build() const {
  if (states.empty()) {
    FSM empty_fsm(1);
    FSMWithStartEnd result(empty_fsm, 0, {0}, false);
    return ResultOk(std::move(result));
  }
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& state : states) {
    auto visited = std::visit([&](auto&& arg) { return visit(arg); }, state);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.size() > 1) {
    return ResultOk(FSMWithStartEnd::Concat(fsm_list));
  } else {
    // If there is only one FSM, return it directly.
    return ResultOk(std::move(fsm_list[0]));
  }
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Leaf& state) const {
  FSMWithStartEnd result = BuildLeafFSM(state);
  return ResultOk(std::move(result));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Union& state) const {
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.size() <= 1) {
    return ResultErr("Invalid union");
  }
  return ResultOk(FSMWithStartEnd::Union(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Symbol& state) const {
  if (state.state.size() != 1) {
    return ResultErr("Invalid symbol");
  }
  Result<FSMWithStartEnd> child_result =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, state.state[0]);
  if (child_result.IsErr()) {
    return child_result;
  }
  auto child = std::move(child_result).Unwrap();

  switch (state.symbol) {
    case RegexIR::RegexSymbol::plus: {
      return ResultOk(child.Plus());
    }
    case RegexIR::RegexSymbol::star: {
      return ResultOk(child.Star());
    }
    case RegexIR::RegexSymbol::optional: {
      return ResultOk(child.Optional());
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Unknown regex symbol: " << static_cast<int>(state.symbol);
      XGRAMMAR_UNREACHABLE();
    }
  }
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Bracket& state) const {
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.empty()) {
    return ResultErr("Invalid bracket");
  }
  return ResultOk(FSMWithStartEnd::Concat(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Repeat& state) const {
  if (state.states.size() != 1) {
    return ResultErr("Invalid repeat");
  }
  bool has_upper_bound = state.upper_bound != RegexIR::kRepeatNoUpperBound;
  if (has_upper_bound && state.upper_bound == 0) {
    // {0} / {0,0}: matches exactly the empty string. The general path below cannot express
    // this: it starts from one copy of the child whose end states stay accepting.
    FSM empty_fsm(1);
    return ResultOk(FSMWithStartEnd(empty_fsm, 0, {0}, false));
  }
  Result<FSMWithStartEnd> child_result =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, state.states[0]);
  if (child_result.IsErr()) {
    return child_result;
  }
  FSMWithStartEnd child = std::move(child_result).Unwrap();
  FSMWithStartEnd result = child.Copy();
  std::unordered_set<int> new_ends;

  if (state.lower_bound <= 1 && (!has_upper_bound || state.upper_bound >= 1)) {
    // A single copy is accepting when the lower bound is at most 1.
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        new_ends.insert(end);
      }
    }
  }

  // Add a fresh accepting start state so that zero repetitions match. A fresh state is
  // required: making the original start accepting would also accept strings that merely
  // loop back to the start inside the first copy.
  auto allow_zero_repetitions = [](FSMWithStartEnd* fsm) {
    int new_start = fsm->AddState();
    fsm->GetFsm().AddEpsilonEdge(new_start, fsm->GetStart());
    fsm->SetStartState(new_start);
    fsm->AddEndState(new_start);
  };

  // Handling {n,}
  if (!has_upper_bound) {
    for (int i = 2; i < state.lower_bound; i++) {
      result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    }
    int end_state_of_lower_bound_fsm = -1;
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        end_state_of_lower_bound_fsm = end;
        break;
      }
    }
    XGRAMMAR_DCHECK(end_state_of_lower_bound_fsm != -1)
        << "No end state found in the lower bound FSM.";
    result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        result.GetFsm().AddEpsilonEdge(end, end_state_of_lower_bound_fsm);
      }
    }
    for (const auto& end : new_ends) {
      result.AddEndState(end);
    }
    if (state.lower_bound == 0) {
      allow_zero_repetitions(&result);
    }
    return ResultOk(std::move(result));
  }
  // Handling {n, m} or {n}
  for (int i = 2; i <= state.upper_bound; i++) {
    result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    if (i >= state.lower_bound) {
      for (int end = 0; end < result.NumStates(); ++end) {
        if (result.IsEndState(end)) {
          new_ends.insert(end);
        }
      }
    }
  }
  for (const auto& end : new_ends) {
    result.AddEndState(end);
  }
  if (state.lower_bound == 0) {
    allow_zero_repetitions(&result);
  }
  return ResultOk(std::move(result));
}

FSMWithStartEnd RegexIR::BuildLeafFSM(const Leaf& leaf) {
  FSMWithStartEnd result(FSM(2), 0, {1});
  for (const auto& range : leaf.ranges) {
    AddCodepointRange(&result, 0, 1, range.minimum, range.maximum);
  }
  return result;
}

std::vector<RegexIR::CodepointRange> RegexIR::NormalizeRanges(std::vector<CodepointRange> ranges) {
  std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
    return std::make_pair(lhs.minimum, lhs.maximum) < std::make_pair(rhs.minimum, rhs.maximum);
  });
  std::vector<CodepointRange> result;
  for (const auto& range : ranges) {
    if (result.empty() || range.minimum > result.back().maximum + 1) {
      result.push_back(range);
    } else {
      result.back().maximum = std::max(result.back().maximum, range.maximum);
    }
  }
  return result;
}

std::vector<RegexIR::CodepointRange> RegexIR::NegateRanges(std::vector<CodepointRange> ranges) {
  ranges = NormalizeRanges(std::move(ranges));
  std::vector<CodepointRange> result;
  TCodepoint next_minimum = 0;
  for (const auto& range : ranges) {
    if (next_minimum < range.minimum) {
      result.push_back({next_minimum, range.minimum - 1});
    }
    if (range.maximum == kMaxUnicodeCodepoint) {
      return result;
    }
    next_minimum = range.maximum + 1;
  }
  if (next_minimum <= kMaxUnicodeCodepoint) {
    result.push_back({next_minimum, kMaxUnicodeCodepoint});
  }
  return result;
}

Result<RegexIR::Leaf> RegexIR::ParseEscape(
    const std::vector<TCodepoint>& regex, int* index, bool in_character_class
) {
  int start = *index;
  if (start + 1 >= static_cast<int>(regex.size())) {
    return ResultErr("Escape sequence is not finished.");
  }
  TCodepoint escaped = regex[start + 1];

  auto make_leaf = [](TCodepoint codepoint) {
    return Leaf{{CodepointRange{codepoint, codepoint}}};
  };
  auto make_ranges = [](std::initializer_list<CodepointRange> ranges) {
    return Leaf{std::vector<CodepointRange>(ranges)};
  };

  if (escaped == 'd') {
    *index = start + 1;
    return ResultOk(make_ranges({{'0', '9'}}));
  }
  if (escaped == 'D') {
    *index = start + 1;
    return ResultOk(Leaf{NegateRanges({{'0', '9'}})});
  }
  if (escaped == 'w') {
    *index = start + 1;
    return ResultOk(make_ranges({{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}}));
  }
  if (escaped == 'W') {
    *index = start + 1;
    return ResultOk(Leaf{NegateRanges({{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}})});
  }
  if (escaped == 's') {
    *index = start + 1;
    return ResultOk(make_ranges(
        {{'\t', '\t'},
         {'\n', '\n'},
         {'\v', '\v'},
         {'\f', '\f'},
         {'\r', '\r'},
         {' ', ' '},
         {0xA0, 0xA0}}
    ));
  }
  if (escaped == 'S') {
    *index = start + 1;
    return ResultOk(Leaf{NegateRanges(
        {{'\t', '\t'},
         {'\n', '\n'},
         {'\v', '\v'},
         {'\f', '\f'},
         {'\r', '\r'},
         {' ', ' '},
         {0xA0, 0xA0}}
    )});
  }
  if (!in_character_class && ((escaped >= '1' && escaped <= '9') || escaped == 'k')) {
    return ResultErr("Backreference is not supported yet.");
  }
  if (!in_character_class && (escaped == 'p' || escaped == 'P')) {
    return ResultErr("Unicode character class escape sequence is not supported yet.");
  }
  if (!in_character_class && (escaped == 'b' || escaped == 'B')) {
    return ResultErr("Word boundary is not supported yet.");
  }

  static const std::unordered_map<TCodepoint, TCodepoint> kSimpleEscapes = {
      {'\'', '\''},
      {'"', '"'},
      {'?', '?'},
      {'\\', '\\'},
      {'/', '/'},
      {'a', '\a'},
      {'b', '\b'},
      {'f', '\f'},
      {'n', '\n'},
      {'r', '\r'},
      {'t', '\t'},
      {'v', '\v'},
      {'0', '\0'},
      {'e', '\x1B'}
  };
  if (auto iterator = kSimpleEscapes.find(escaped); iterator != kSimpleEscapes.end()) {
    *index = start + 1;
    return ResultOk(make_leaf(iterator->second));
  }

  uint64_t parsed_codepoint = 0;
  if (escaped == 'u' && start + 2 < static_cast<int>(regex.size()) && regex[start + 2] == '{') {
    int current = start + 3;
    int digit_count = 0;
    while (current < static_cast<int>(regex.size()) && regex[current] <= 0x7F &&
           HexCharToInt(static_cast<char>(regex[current])) != -1) {
      if (++digit_count > 6) {
        return ResultErr("Invalid Unicode escape sequence.");
      }
      parsed_codepoint = parsed_codepoint * 16 + HexCharToInt(static_cast<char>(regex[current]));
      current++;
    }
    if (digit_count == 0 || current >= static_cast<int>(regex.size()) || regex[current] != '}') {
      return ResultErr("Invalid Unicode escape sequence.");
    }
    *index = current;
  } else if (escaped == 'u' || escaped == 'U') {
    int digit_count = escaped == 'u' ? 4 : 8;
    if (start + digit_count + 1 >= static_cast<int>(regex.size())) {
      return ResultErr("Escape sequence is not finished.");
    }
    for (int offset = 0; offset < digit_count; ++offset) {
      TCodepoint digit_codepoint = regex[start + 2 + offset];
      int digit = digit_codepoint <= 0x7F ? HexCharToInt(static_cast<char>(digit_codepoint)) : -1;
      if (digit == -1) {
        return ResultErr("Invalid Unicode escape sequence.");
      }
      parsed_codepoint = parsed_codepoint * 16 + digit;
    }
    *index = start + digit_count + 1;
  } else if (escaped == 'x') {
    int current = start + 2;
    int digit_count = 0;
    while (current < static_cast<int>(regex.size()) && regex[current] <= 0x7F &&
           HexCharToInt(static_cast<char>(regex[current])) != -1) {
      int digit = HexCharToInt(static_cast<char>(regex[current]));
      if (parsed_codepoint > (static_cast<uint64_t>(kMaxUnicodeCodepoint) - digit) / 16) {
        return ResultErr("Invalid Unicode codepoint in escape sequence.");
      }
      parsed_codepoint = parsed_codepoint * 16 + digit;
      digit_count++;
      current++;
    }
    if (digit_count == 0) {
      return ResultErr("Invalid hexadecimal escape sequence.");
    }
    *index = current - 1;
  } else if (escaped == 'c') {
    if (start + 2 >= static_cast<int>(regex.size()) ||
        !((regex[start + 2] >= 'A' && regex[start + 2] <= 'Z') ||
          (regex[start + 2] >= 'a' && regex[start + 2] <= 'z'))) {
      return ResultErr("Invalid control character escape sequence.");
    }
    parsed_codepoint = regex[start + 2] % 32;
    *index = start + 2;
  } else {
    *index = start + 1;
    return ResultOk(make_leaf(escaped));
  }

  if (parsed_codepoint > static_cast<uint64_t>(kMaxUnicodeCodepoint) ||
      (parsed_codepoint >= static_cast<uint64_t>(kHighSurrogateStart) &&
       parsed_codepoint <= static_cast<uint64_t>(kLowSurrogateEnd))) {
    return ResultErr("Invalid Unicode codepoint in escape sequence.");
  }
  return ResultOk(make_leaf(static_cast<TCodepoint>(parsed_codepoint)));
}

Result<RegexIR::Leaf> RegexIR::ParseCharacterClass(
    const std::vector<TCodepoint>& regex, int* index
) {
  XGRAMMAR_DCHECK(regex[*index] == '[');
  int current = *index + 1;
  bool is_negative = current < static_cast<int>(regex.size()) && regex[current] == '^';
  if (is_negative) {
    current++;
  }
  if (current >= static_cast<int>(regex.size())) {
    return ResultErr("Empty character class is not allowed in regex.");
  }
  if (regex[current] == ']') {
    if (!is_negative) {
      return ResultErr("Empty character class is not allowed in regex.");
    }
    *index = current;
    return ResultOk(Leaf{{CodepointRange{0, kMaxUnicodeCodepoint}}});
  }

  auto parse_atom = [&](int* position) -> Result<Leaf> {
    if (*position >= static_cast<int>(regex.size()) || regex[*position] == ']') {
      return ResultErr("Character class range is not finished.");
    }
    if (regex[*position] == '\n' || regex[*position] == '\r') {
      return ResultErr("Character class should not contain newline.");
    }
    if (regex[*position] == '\\') {
      int escape_index = *position;
      auto escape_result = ParseEscape(regex, &escape_index, true);
      if (escape_result.IsErr()) {
        return escape_result;
      }
      *position = escape_index + 1;
      return escape_result;
    }
    TCodepoint codepoint = regex[*position];
    (*position)++;
    return ResultOk(Leaf{{CodepointRange{codepoint, codepoint}}});
  };

  std::vector<CodepointRange> ranges;
  while (current < static_cast<int>(regex.size()) && regex[current] != ']') {
    auto left_result = parse_atom(&current);
    if (left_result.IsErr()) {
      return left_result;
    }
    Leaf left = std::move(left_result).Unwrap();
    bool has_range_separator = current + 1 < static_cast<int>(regex.size()) &&
                               regex[current] == '-' && regex[current + 1] != ']';
    if (has_range_separator && left.ranges.size() == 1 &&
        left.ranges[0].minimum == left.ranges[0].maximum) {
      int right_position = current + 1;
      auto right_result = parse_atom(&right_position);
      if (right_result.IsErr()) {
        return right_result;
      }
      Leaf right = std::move(right_result).Unwrap();
      if (right.ranges.size() == 1 && right.ranges[0].minimum == right.ranges[0].maximum) {
        if (left.ranges[0].minimum > right.ranges[0].minimum) {
          return ResultErr("Character class range has a larger start than end.");
        }
        ranges.push_back({left.ranges[0].minimum, right.ranges[0].minimum});
        current = right_position;
        continue;
      }
    }
    ranges.insert(ranges.end(), left.ranges.begin(), left.ranges.end());
  }

  if (current >= static_cast<int>(regex.size()) || regex[current] != ']') {
    return ResultErr("Unclosed character class.");
  }
  *index = current;
  ranges = NormalizeRanges(std::move(ranges));
  if (is_negative) {
    ranges = NegateRanges(std::move(ranges));
  }
  return ResultOk(Leaf{std::move(ranges)});
}

Result<FSMWithStartEnd> RegexFSMBuilder::Build(const std::string& regex) {
  auto codepoint_result = ParseRegexCodepoints(regex);
  if (codepoint_result.IsErr()) {
    return ResultErr(std::move(codepoint_result).UnwrapErr());
  }
  std::vector<TCodepoint> regex_codepoints = std::move(codepoint_result).Unwrap();
  RegexIR ir;
  using IRState = std::variant<RegexIR::State, char>;
  // We use a stack to store the states.
  std::stack<IRState> stack;
  for (int i = 0; i < static_cast<int>(regex_codepoints.size()); i++) {
    TCodepoint current_codepoint = regex_codepoints[i];
    if (i == 0 && current_codepoint == '^') {
      continue;
    }
    if (i == static_cast<int>(regex_codepoints.size()) - 1 && current_codepoint == '$') {
      continue;
    }
    if (current_codepoint == '[') {
      auto leaf_result = RegexIR::ParseCharacterClass(regex_codepoints, &i);
      if (leaf_result.IsErr()) {
        return ResultErr(std::move(leaf_result).UnwrapErr());
      }
      stack.push(std::move(leaf_result).Unwrap());
      continue;
    }
    if (current_codepoint == ']') {
      return ResultErr("Invalid middle bracket!");
    }
    if (current_codepoint == '+' || current_codepoint == '*' || current_codepoint == '?') {
      if (stack.empty()) {
        return ResultErr("Invalid regex: no state before operator!");
      }
      auto state = stack.top();
      if (std::holds_alternative<char>(state)) {
        return ResultErr("Invalid regex: no state before operator!");
      }
      stack.pop();
      auto child = std::get<RegexIR::State>(state);
      RegexIR::Symbol symbol;
      symbol.state.push_back(child);
      switch (current_codepoint) {
        case '+': {
          symbol.symbol = RegexIR::RegexSymbol::plus;
          break;
        }
        case '*': {
          symbol.symbol = RegexIR::RegexSymbol::star;
          break;
        }
        case '?': {
          symbol.symbol = RegexIR::RegexSymbol::optional;
          break;
        }
      }
      stack.push(symbol);
      if (i + 1 < static_cast<int>(regex_codepoints.size()) && regex_codepoints[i + 1] == '?') {
        i++;
      }
      if (i + 1 < static_cast<int>(regex_codepoints.size()) &&
          (regex_codepoints[i + 1] == '{' || regex_codepoints[i + 1] == '*' ||
           regex_codepoints[i + 1] == '+' || regex_codepoints[i + 1] == '?')) {
        return ResultErr("Two consecutive repetition modifiers are not allowed.");
      }
      continue;
    }
    if (current_codepoint == '(' || current_codepoint == '|') {
      stack.push(static_cast<char>(current_codepoint));
      if (i < static_cast<int>(regex_codepoints.size()) - 2 && current_codepoint == '(' &&
          regex_codepoints[i + 1] == '?' && regex_codepoints[i + 2] == ':') {
        i += 2;
        continue;
      }
      if (i < static_cast<int>(regex_codepoints.size()) - 2 && current_codepoint == '(' &&
          regex_codepoints[i + 1] == '?' &&
          (regex_codepoints[i + 2] == '!' || regex_codepoints[i + 2] == '=')) {
        return ResultErr("Lookahead is not supported yet.");
      }
      continue;
    }
    if (current_codepoint == ')') {
      std::stack<IRState> states;
      bool paired = false;
      bool unioned = false;
      while ((!stack.empty()) && (!paired)) {
        auto state = stack.top();
        stack.pop();
        if (std::holds_alternative<char>(state)) {
          char c = std::get<char>(state);
          if (c == '(') {
            paired = true;
            break;
          }
          if (c == '|') {
            unioned = true;
          }
          states.push(state);
        } else {
          states.push(state);
        }
      }
      if (!paired) {
        return ResultErr("Invalid regex: no paired bracket!" + std::to_string(__LINE__));
      }
      if (states.empty()) {
        continue;
      }
      if (!unioned) {
        RegexIR::Bracket bracket;
        while (!states.empty()) {
          auto state = states.top();
          states.pop();
          auto child = std::get<RegexIR::State>(state);
          bracket.states.push_back(child);
        }
        stack.push(bracket);
      } else {
        RegexIR::Union union_state;
        RegexIR::Bracket bracket;
        while (!states.empty()) {
          auto state = states.top();
          states.pop();
          if (std::holds_alternative<char>(state)) {
            char c = std::get<char>(state);
            if (c == '|') {
              union_state.states.push_back(bracket);
              bracket.states.clear();
              continue;
            }
            return ResultErr("Invalid regex: no paired bracket!" + std::to_string(__LINE__));
          }
          if (std::holds_alternative<RegexIR::State>(state)) {
            auto child = std::get<RegexIR::State>(state);
            bracket.states.push_back(child);
            continue;
          }
          return ResultErr("Invalid regex: no paired bracket!" + std::to_string(__LINE__));
        }
        union_state.states.push_back(bracket);
        stack.push(union_state);
      }
      continue;
    }
    if (current_codepoint == '{') {
      if (stack.empty()) {
        return ResultErr("Invalid regex: no state before repeat!");
      }
      auto state = stack.top();
      if (std::holds_alternative<char>(state)) {
        return ResultErr("Invalid regex: no state before repeat!");
      }
      stack.pop();
      auto bounds_result = RegexIR::CheckRepeat(regex_codepoints, i);
      if (bounds_result.IsErr()) {
        return ResultErr(std::move(bounds_result).UnwrapErr());
      }
      auto bounds = std::move(bounds_result).Unwrap();
      auto child = std::get<RegexIR::State>(state);
      RegexIR::Repeat repeat;
      repeat.lower_bound = bounds.first;
      repeat.upper_bound = bounds.second;
      repeat.states.push_back(child);
      stack.push(repeat);
      if (i + 1 < static_cast<int>(regex_codepoints.size()) && regex_codepoints[i + 1] == '?') {
        i++;
      }
      if (i + 1 < static_cast<int>(regex_codepoints.size()) &&
          (regex_codepoints[i + 1] == '{' || regex_codepoints[i + 1] == '*' ||
           regex_codepoints[i + 1] == '+' || regex_codepoints[i + 1] == '?')) {
        return ResultErr("Two consecutive repetition modifiers are not allowed.");
      }
      continue;
    }
    RegexIR::Leaf leaf;
    if (current_codepoint == '\\') {
      auto leaf_result = RegexIR::ParseEscape(regex_codepoints, &i, false);
      if (leaf_result.IsErr()) {
        return ResultErr(std::move(leaf_result).UnwrapErr());
      }
      leaf = std::move(leaf_result).Unwrap();
    } else if (current_codepoint == '.') {
      leaf.ranges = {{0, kMaxUnicodeCodepoint}};
    } else {
      leaf.ranges = {{current_codepoint, current_codepoint}};
    }
    stack.push(std::move(leaf));
    continue;
  }
  std::vector<RegexIR::State> res_states;
  std::vector<decltype(res_states)> union_state_list;
  bool unioned = false;
  while (!stack.empty()) {
    if (std::holds_alternative<char>(stack.top())) {
      char c = std::get<char>(stack.top());
      if (c == '|') {
        union_state_list.push_back(res_states);
        res_states.clear();
        unioned = true;
        stack.pop();
        continue;
      }
      return ResultErr("Invalid regex: no paired!");
    }
    auto state = stack.top();
    stack.pop();
    auto child = std::get<RegexIR::State>(state);
    res_states.push_back(std::move(child));
  }
  if (!unioned) {
    for (auto it = res_states.rbegin(); it != res_states.rend(); ++it) {
      ir.states.push_back(std::move(*it));
    }
  } else {
    union_state_list.push_back(res_states);
    RegexIR::Union union_state;
    for (auto it = union_state_list.begin(); it != union_state_list.end(); ++it) {
      RegexIR::Bracket bracket;
      for (auto state = it->rbegin(); state != it->rend(); ++state) {
        bracket.states.push_back(std::move(*state));
      }
      union_state.states.push_back(std::move(bracket));
    }
    ir.states.push_back(std::move(union_state));
  }
  return ir.Build();
}

Result<FSMWithStartEnd> RegexFSMBuilder::BuildWithForbiddenChars(
    const std::string& regex, const std::bitset<256>& forbidden_chars
) {
  auto build_result = Build(regex);
  if (build_result.IsErr() || forbidden_chars.none()) {
    return build_result;
  }
  auto fsm_wse = std::move(build_result).Unwrap();
  const auto& fsm = fsm_wse.GetFsm();
  FSM new_fsm(fsm_wse.NumStates());
  for (int state = 0; state < fsm_wse.NumStates(); ++state) {
    for (const auto& edge : fsm.GetEdges(state)) {
      if (!edge.IsCharRange()) {
        new_fsm.AddEdge(state, edge.target, edge.min, edge.max);
        continue;
      }
      // Split the character range into the maximal sub-ranges of allowed characters.
      int range_start = -1;
      for (int c = edge.min; c <= edge.max + 1; ++c) {
        if (c <= edge.max && !forbidden_chars[c]) {
          if (range_start == -1) {
            range_start = c;
          }
        } else if (range_start != -1) {
          new_fsm.AddEdge(state, edge.target, range_start, c - 1);
          range_start = -1;
        }
      }
    }
  }
  return ResultOk(FSMWithStartEnd(new_fsm, fsm_wse.GetStart(), fsm_wse.GetEnds()));
}

class TrieFSMBuilderImpl {
 public:
  TrieFSMBuilderImpl() = default;
  std::optional<FSMWithStartEnd> Build(
      const std::vector<std::string>& patterns,
      const std::vector<std::string>& excluded_patterns,
      std::vector<int32_t>* end_states,
      bool allow_overlap,
      bool add_back_edges
  );
  void AddBackEdges(FSM* fsm, int start, const std::unordered_set<int>& ends);
};

std::optional<FSMWithStartEnd> TrieFSMBuilderImpl::Build(
    const std::vector<std::string>& patterns,
    const std::vector<std::string>& excluded_patterns,
    std::vector<int32_t>* end_states,
    bool allow_overlap,
    bool add_back_edges
) {
  FSM fsm(1);
  int start = 0;
  std::unordered_set<int> ends;

  if (end_states) {
    end_states->clear();
  }

  for (const auto& pattern : patterns) {
    // Check for empty patterns
    if (!allow_overlap && pattern.empty()) {
      return std::nullopt;
    }

    int current_state = 0;
    for (const auto& ch : pattern) {
      int32_t ch_int32 = static_cast<int32_t>(static_cast<uint8_t>(ch));
      int next_state = fsm.GetNextState(current_state, ch_int32);
      if (next_state == FSM::kNoNextState) {
        next_state = fsm.AddState();
        fsm.AddEdge(current_state, next_state, ch_int32, ch_int32);
      }
      current_state = next_state;
      if (!allow_overlap && ends.count(current_state) > 0) {
        return std::nullopt;
      }
    }
    if (!allow_overlap && fsm.GetEdges(current_state).size() > 0) {
      return std::nullopt;
    }
    ends.insert(current_state);
    if (end_states) {
      end_states->push_back(current_state);
    }
  }

  std::unordered_set<int32_t> dead_state_set;

  if (add_back_edges) {
    // Build trie for excluded patterns.
    for (const auto& excluded_pattern : excluded_patterns) {
      if (!allow_overlap && excluded_pattern.empty()) {
        return std::nullopt;
      }

      int current_state = 0;
      for (const auto& ch : excluded_pattern) {
        int32_t ch_int32 = static_cast<int32_t>(static_cast<uint8_t>(ch));
        int next_state = fsm.GetNextState(current_state, ch_int32);
        if (next_state == FSM::kNoNextState) {
          next_state = fsm.AddState();
          fsm.AddEdge(current_state, next_state, ch_int32, ch_int32);
        }
        current_state = next_state;
        if (!allow_overlap && ends.count(current_state) > 0) {
          return std::nullopt;
        }
      }
      if (!allow_overlap && fsm.GetEdges(current_state).size() > 0) {
        return std::nullopt;
      }

      ends.insert(current_state);
      dead_state_set.insert(current_state);
    }

    // Add back edges.
    AddBackEdges(&fsm, start, ends);

    // Remove the edges to excluded end states.
    if (dead_state_set.size() != 0) {
      for (int state = 0; state < fsm.NumStates(); state++) {
        std::vector<FSMEdge>& edges = fsm.GetEdges(state);
        std::vector<FSMEdge> new_edges;
        for (const auto& edge : edges) {
          if (dead_state_set.count(edge.target) == 0) {
            new_edges.push_back(edge);
          }
        }
        edges = std::move(new_edges);
      }
    }
  } else if (excluded_patterns.size() > 0) {
    XGRAMMAR_LOG(WARNING) << "Excluded patterns are ignored when back edges are not added.";
  }

  return FSMWithStartEnd(fsm, start, std::vector<int32_t>(ends.begin(), ends.end()));
}

void TrieFSMBuilderImpl::AddBackEdges(FSM* fsm, int start, const std::unordered_set<int>& ends) {
  // Build an Aho-Corasick automaton by adding back edges.
  // When matching on the trie fails at state u on byte b, the matcher must resume from
  // the longest proper suffix of u's prefix that is still a path in the trie (the
  // failure state), and retry b from there. Falling back only to the start state (or to
  // the start state's direct children) loses matches whose start lies inside an
  // already-followed branch of another pattern. Example: patterns {"bcd", "abce"} on
  // input "abcd" -- after following "abc" of the "abce" branch, 'd' must transition to
  // the "bcd" end state via the failure state "bc", not back to the start state.

  int num_states = fsm->NumStates();

  // Step 1. Record the BFS order of the trie (a tree at this point), so that shallower
  // states are always processed first.
  std::vector<int> bfs_order;
  bfs_order.reserve(num_states);
  bfs_order.push_back(start);
  for (size_t head = 0; head < bfs_order.size(); head++) {
    for (const auto& edge : fsm->GetEdges(bfs_order[head])) {
      XGRAMMAR_DCHECK(edge.min == edge.max && edge.min >= 0 && edge.min <= 255);
      bfs_order.push_back(edge.target);
    }
  }
  XGRAMMAR_DCHECK(static_cast<int>(bfs_order.size()) == num_states);

  // Step 2. Compute the failure link and the fully resolved transition table with the
  // textbook O(num_states * 256) dynamic program: delta[u][b] is the trie child when it
  // exists, and delta[fail[u]][b] otherwise -- fail[u] is strictly shallower than u, so
  // its row is already final when u is processed in BFS order.
  std::vector<int> fail(num_states, start);
  std::vector<std::array<int, 256>> delta(num_states);
  for (auto& row : delta) {
    row.fill(FSM::kNoNextState);
  }
  for (int u = 0; u < num_states; u++) {
    for (const auto& edge : fsm->GetEdges(u)) {
      delta[u][edge.min] = edge.target;
    }
  }
  for (int u : bfs_order) {
    for (int byte = 0; byte < 256; byte++) {
      // Entries of deeper states are untouched so far, so a non-empty entry here is
      // exactly a trie child of u.
      int child = delta[u][byte];
      int fallback = (u == start) ? start : delta[fail[u]][byte];
      if (child == FSM::kNoNextState) {
        delta[u][byte] = fallback;
      } else {
        fail[child] = fallback;
      }
    }
  }

  // Step 3. Overwrite the edges of every non-end state with its resolved row,
  // compressing consecutive bytes with the same target into range edges.
  for (int u = 0; u < num_states; u++) {
    if (u != start && ends.count(u) > 0) {
      continue;
    }
    const auto& row = delta[u];
    std::vector<FSMEdge> new_edges;
    for (int byte = 0; byte < 256;) {
      int target = row[byte];
      int range_end = byte;
      while (range_end + 1 < 256 && row[range_end + 1] == target) {
        range_end++;
      }
      new_edges.push_back(FSMEdge(byte, range_end, target));
      byte = range_end + 1;
    }
    fsm->GetEdges(u) = std::move(new_edges);
  }
}

std::optional<FSMWithStartEnd> TrieFSMBuilder::Build(
    const std::vector<std::string>& patterns,
    const std::vector<std::string>& exclude_patterns,
    std::vector<int32_t>* end_states,
    bool allow_overlap,
    bool add_back_edges
) {
  return TrieFSMBuilderImpl().Build(
      patterns, exclude_patterns, end_states, allow_overlap, add_back_edges
  );
}

}  // namespace xgrammar
