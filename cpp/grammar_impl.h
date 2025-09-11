/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_GRAMMAR_IMPL_H_
#define XGRAMMAR_GRAMMAR_IMPL_H_

#include <xgrammar/xgrammar.h>

#include <cstddef>
#include <string>
#include <vector>

#include "fsm.h"
#include "support/logging.h"
#include "support/reflection.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

/*!
 * \brief This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar.
 * The BNF definition here is standard BNF, and the characters are represented using regex-style
 * character classes (e.g. [a-z], [^a-z]).
 *
 * \details
 * ### Rules
 * The BNF grammar AST consists of a set of rules. Each rule contains a name and a definition, and
 * corresponds to a production in the grammar. The definition of a rule is a GrammarExpr. Each rule
 * has a rule_id for reference.
 *
 * ### GrammarExprs
 * GrammarExpr is the definition of a rule or part of the definition of a rule. It can contain
 * elements, empty string, reference to other GrammarExprs, or reference to other rules. Each
 * GrammarExpr corresponds to a grammar_expr_id for reference.
 *
 * For example, in the following rule: rule ::= ("a" "b") | "c"
 * ("a" "b"), "c", ("a" "b") | "c" are all GrammarExprs.
 *
 * #### Types of GrammarExprs
 * Every GrammarExpr is represented by a type as well as a variable-length array containing its
 * data. GrammarExpr has several types:
 * - Byte string: a string of bytes (0~255). Supports UTF-8 strings.
 * - Character class: a range of characters (each character is a unicode codepoint), e.g. [a-z],
 *   [ac-z]. Can be negated: [^a-z], [^ac-z]. Now only ascii chars is allowed in [], but this
 *   expression can accept/reject unicode chars.
 * - Character class star: a star quantifier of a character class. e.g. [a-z]*, [^a-z]*.
 * - EmptyStr: an empty string, i.e. ""
 * - Rule reference: a reference to another rule
 * - Sequence: a sequence of grammar_exprs, e.g. ("a" "b"). These grammar_exprs are concatenated
 * together.
 * - Choices: a choice of grammar_exprs, e.g. ("a" "b") | "c". Each grammar_expr can be matched.
 *
 * #### Storage of GrammarExprs
 * Each type of GrammarExpr has a different data format. For the format of each type of GrammarExpr,
 * see docs in Grammar::Impl::GrammarExprType.
 *
 * We store all GrammarExprs in csr_matrix style. That is, they are stored consecutively in one
 * vector (data vector) and the starting position of each GrammarExpr is recorded in the indptr
 * vector.
 *
 * \remark The character class star GrammarExpr is for the special support for elements like [a-z]*
 * in the grammar. We add it to make the matching more efficient, as we can avoid recursion into
 * rules when matching a sequence of characters. It should be used like:
 * rule1 ::= ((element1 element2 rule2 ...) | ...)
 * rule2 ::= character_class_star_grammar_expr(id_of_a_character_class_grammar_expr)
 */
class Grammar::Impl {
 public:
  /*! \brief A rule with name. */
  struct Rule {
    /*! \brief The name of the rule. */
    std::string name;
    /*! \brief The GrammarExpr id of the body of the rule. */
    int32_t body_expr_id;
    /*! \brief The id of the associated lookahead assertion expr. For now it must be a id of a
     * sequence GrammarExpr. -1 if not exists. */
    int32_t lookahead_assertion_id = -1;
    /*! \brief Whether the lookahead assertion is exact. */
    bool is_exact_lookahead = false;
  };

  /*! \brief Get the number of rules. */
  int32_t NumRules() const { return rules_.size(); }
  /*! \brief Get the rule with the given id. */
  const Rule& GetRule(int32_t rule_id) const {
    XGRAMMAR_DCHECK(rule_id >= 0 && rule_id < static_cast<int32_t>(rules_.size()))
        << "rule_id " << rule_id << " is out of bound";
    return rules_[rule_id];
  }
  Rule& GetRule(int32_t rule_id) {
    XGRAMMAR_DCHECK(rule_id >= 0 && rule_id < static_cast<int32_t>(rules_.size()))
        << "rule_id " << rule_id << " is out of bound";
    return rules_[rule_id];
  }
  /*! \brief Get the root rule id of the grammar. */
  int32_t GetRootRuleId() const { return root_rule_id_; }
  /*! \brief Get the root rule of the grammar. */
  const Rule& GetRootRule() const {
    XGRAMMAR_DCHECK(root_rule_id_ >= 0 && root_rule_id_ < static_cast<int32_t>(rules_.size()))
        << "root_rule_id " << root_rule_id_ << " is out of bound";
    return rules_[root_rule_id_];
  }

  /*! \brief The type of the grammar expr. */
  enum class GrammarExprType : int32_t {
    // data format: [byte0, byte1, ...]
    kByteString,
    // data format: [is_negative, lower0, upper0, lower1, upper1, ...]
    kCharacterClass,
    kCharacterClassStar,
    // data format: []
    kEmptyStr,
    // data format: [rule_id]
    kRuleRef,
    // data format: [grammar_expr_id0, grammar_expr_id1, ...]
    kSequence,
    // data format: [grammar_expr_id0, grammar_expr_id1, ...]
    kChoices,
    // data format: [tag_expr0, rule_id0, tag_expr1, rule_id1, ..., stop_eos, stop_str_expr_id,
    // loop_after_dispatch]
    // where stop_eos is a bool, stop_str_expr_id is a choices GrammarExpr id.
    // tag_expr should be a byte string, and rule_id should be a rule id.
    // loop_after_dispatch is a bool.
    kTagDispatch,
    // data format: [grammar_expr_id, min_repeat_count, max_repeat_count]
    kRepeat,
  };

  /*! \brief The object representing a grammar expr. */
  struct GrammarExpr {
    /*! \brief The type of the grammar expr. */
    GrammarExprType type;
    /*! \brief The data of the GrammarExpr. A variable-length array. */
    const int32_t* data;
    /*! \brief The length of the data array. */
    int32_t data_len;

    int32_t size() const { return data_len; }
    /*! \brief Get the i-th element of the data array. */
    const int32_t& operator[](int i) const {
      XGRAMMAR_DCHECK(i >= 0 && i < static_cast<int32_t>(data_len))
          << "Index " << i << " is out of bound";
      return data[i];
    }
    const int32_t* begin() const { return data; }
    const int32_t* end() const { return data + data_len; }
    void SetData(int index, int value) { const_cast<int32_t*>(data)[index] = value; }
  };

  /*! \brief Get the number of grammar_exprs. */
  int32_t NumGrammarExprs() const { return grammar_expr_indptr_.size(); }

  /*! \brief Get the grammar_expr with the given id. */
  GrammarExpr GetGrammarExpr(int32_t grammar_expr_id) const {
    XGRAMMAR_DCHECK(
        grammar_expr_id >= 0 && grammar_expr_id < static_cast<int32_t>(grammar_expr_indptr_.size())
    ) << "grammar_expr_id "
      << grammar_expr_id << " is out of bound";
    int start_index = grammar_expr_indptr_[grammar_expr_id];
    auto start_ptr = grammar_expr_data_.data() + start_index;
    auto type = static_cast<GrammarExprType>(start_ptr[0]);
    auto data_ptr = start_ptr + 2;
    auto data_len = start_ptr[1];
    return {type, data_ptr, data_len};
  }

  /******************* GrammarExpr Getters *******************/

  /*! \brief Get the string of the byte string grammar expr. */
  std::string GetByteString(const GrammarExpr& grammar_expr) const {
    std::string str;
    str.reserve(grammar_expr.size());
    for (int i = 0; i < grammar_expr.size(); ++i) {
      str.push_back(static_cast<char>(static_cast<uint8_t>(grammar_expr[i])));
    }
    return str;
  }

  /*! \brief Get the string of the byte string grammar expr. */
  std::string GetByteString(int32_t grammar_expr_id) const {
    return GetByteString(GetGrammarExpr(grammar_expr_id));
  }

  /*! \brief The object representing a tag dispatch. */
  struct TagDispatch {
    /*! \brief The tag and rule id pairs. */
    std::vector<std::pair<std::string, int32_t>> tag_rule_pairs;
    /*! \brief If true, EOS is allowed to generate and will stop the tag dispatch. */
    bool stop_eos;
    /*! \brief The strings that will stop the tag dispatch. Only work if stop_eos is false. */
    std::vector<std::string> stop_str;
    /*! \brief If true, the tag dispatch will loop after dispatching. */
    bool loop_after_dispatch;
  };

  /*! \brief Get the tag dispatch from the grammar expr. */
  TagDispatch GetTagDispatch(const GrammarExpr& grammar_expr) {
    XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kTagDispatch)
        << "GrammarExpr is not a tag dispatch";

    TagDispatch result;
    XGRAMMAR_DCHECK(grammar_expr.size() >= 3);
    result.tag_rule_pairs.reserve((grammar_expr.size() - 3) / 2);

    for (int i = 0; i < grammar_expr.size() - 3; i += 2) {
      auto tag_expr_id = grammar_expr[i];
      auto rule_id = grammar_expr[i + 1];
      result.tag_rule_pairs.push_back({GetByteString(tag_expr_id), rule_id});
    }

    result.stop_eos = static_cast<bool>(grammar_expr[grammar_expr.size() - 3]);

    auto stop_str_expr = GetGrammarExpr(grammar_expr[grammar_expr.size() - 2]);
    XGRAMMAR_DCHECK(stop_str_expr.type == GrammarExprType::kChoices);
    result.stop_str.reserve(stop_str_expr.size());
    for (int j = 0; j < stop_str_expr.size(); j++) {
      result.stop_str.push_back(GetByteString(stop_str_expr[j]));
    }

    result.loop_after_dispatch = static_cast<bool>(grammar_expr[grammar_expr.size() - 1]);

    return result;
  }

  /*! \brief Get the tag dispatch from the grammar expr with the given id. */
  TagDispatch GetTagDispatch(int32_t grammar_expr_id) {
    return GetTagDispatch(GetGrammarExpr(grammar_expr_id));
  }

 private:
  /*! \brief The rules of the grammar. rule_id corresponds the index of this vector. */
  std::vector<Rule> rules_;
  /*! \brief The data of all grammar_exprs. */
  std::vector<int32_t> grammar_expr_data_;
  /*! \brief The start index of every grammar_expr in grammar_expr_data_. grammar_expr_id is the
   * index to the elements in this vector. */
  std::vector<int32_t> grammar_expr_indptr_;
  /*! \brief The id of the root rule. */
  int32_t root_rule_id_ = -1;

 public:
  /******************* Aux information for matching *******************/

  /*! \brief The complete FSM for the grammar. It contains the FSMs for all rules. */
  CompactFSM complete_fsm{NullObj{}};

  /*!
   * \brief The FSM for each rule.
   * \details The FSM will be used in matching if it exists. If it does not exist (std::nullopt),
   * the rule will be used in matching, and the rule's body must be a kChoices expr.
   */
  std::vector<std::optional<CompactFSMWithStartEnd>> per_rule_fsms;

  /*! \brief The ids of the rules that are allowed to be empty. */
  std::vector<int32_t> allow_empty_rule_ids;

  /*! \brief Whether the grammar is optimized. */
  bool optimized = false;

  friend class GrammarBuilder;
  friend class GrammarCompiler;

  friend std::size_t MemorySize(const Impl& impl);
  friend struct member_trait<Impl>;
};

XGRAMMAR_MEMBER_ARRAY(
    Grammar::Impl::Rule,
    &Grammar::Impl::Rule::name,
    &Grammar::Impl::Rule::body_expr_id,
    &Grammar::Impl::Rule::lookahead_assertion_id,
    &Grammar::Impl::Rule::is_exact_lookahead
);

XGRAMMAR_MEMBER_TABLE(
    Grammar::Impl,
    "rules",
    &Grammar::Impl::rules_,
    "grammar_expr_data",
    &Grammar::Impl::grammar_expr_indptr_,
    "grammar_expr_indptr",
    &Grammar::Impl::grammar_expr_data_,
    "root_rule_id",
    &Grammar::Impl::root_rule_id_,
    "complete_fsm",
    &Grammar::Impl::complete_fsm,
    "per_rule_fsms",
    &Grammar::Impl::per_rule_fsms,
    "allow_empty_rule_ids",
    &Grammar::Impl::allow_empty_rule_ids,
    "optimized",
    &Grammar::Impl::optimized
);

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_IMPL_H_
