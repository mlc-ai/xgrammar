/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/compiler.h
 * \brief The header for the compiler.
 */

#ifndef XGRAMMAR_COMPILER_H_
#define XGRAMMAR_COMPILER_H_

#include <xgrammar/grammar.h>
#include <xgrammar/tokenizer_info.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace xgrammar {

/*!
 * \brief The compiled grammar of a GrammarMatcher. It contains the preprocessing results of the
 * grammar and tokenizer.
 */
class CompiledGrammar {
 public:
  Grammar GetGrammar() const;
  TokenizerInfo GetTokenizerInfo() const;
  /*! \brief Return the approximate memory usage of the grammar in bytes. */
  std::size_t MemorySize() const;

  XGRAMMAR_DEFINE_PIMPL_METHODS(CompiledGrammar);
};

/*!
 * \brief A cache to get the compiled grammar for grammar or schema. This class avoids
 * redundant preprocessing of the grammar or schema when constructing a CompiledGrammar.
 * \note This class is associated with a vocabulary when constructed. The vocabulary is used to
 * create every compiled grammar. If multiple toke tables are used to create init
 * contexts, an instance of this class for each vocabulary should be created.
 */
class GrammarCompiler {
 public:
  /*!
   * \brief Construct a GrammarCompiler with a vocabulary. This class will always
   * create compiled grammars with this vocabulary.
   * \param tokenizer_info The tokenizer info.
   * \param max_threads The maximum number of threads to use for compiling grammars.
   * \param cache_enabled Whether to enable the cache.
   * \param max_memory_bytes The maximum memory usage in bytes.
   */
  GrammarCompiler(
      const TokenizerInfo& tokenizer_info,
      int max_threads = 8,
      bool cache_enabled = true,
      long long max_memory_bytes = -1  // unlimited
  );

  /*! \brief Get the compiled grammar for a JSON schema string. */
  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace = true,
      std::optional<int> indent = std::nullopt,
      std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
      bool strict_mode = true
  );

  /*! \brief Get the compiled grammar for pure JSON. */
  CompiledGrammar CompileBuiltinJSONGrammar();

  /*! \brief Get the compiled grammar for a grammar. */
  CompiledGrammar CompileGrammar(const Grammar& grammar);

  /*! \brief Get the compiled grammar for a structural tag. */
  CompiledGrammar CompileStructuralTag(
      const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
  );

  /*! \brief Get the compiled grammar for a regex. */
  CompiledGrammar CompileRegex(const std::string& regex);

  /*! \brief Clear the internal cache of compiled grammars. */
  void ClearCache();

  XGRAMMAR_DEFINE_PIMPL_METHODS(GrammarCompiler);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_COMPILER_H_
