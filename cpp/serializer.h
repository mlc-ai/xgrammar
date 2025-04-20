#ifndef XGRAMMAR_SERIALIZER_H_
#define XGRAMMAR_SERIALIZER_H_
#include <picojson.h>

#include <string>

#include "xgrammar/compiler.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

struct JSONSerializer {
  static std::string SerializeGrammar(const Grammar&, bool prettify = false);
  static Grammar DeserializeGrammar(const std::string&);
  static std::string SerializeTokenizerInfo(const TokenizerInfo&, bool prettify = false);
  static TokenizerInfo DeserializeTokenizerInfo(
      const std::string&, const std::vector<std::string>& encoded_vocab = {}
  );
  static std::string SerializeCompiledGrammar(const CompiledGrammar&, bool prettify = false);
  static CompiledGrammar DeserializeCompiledGrammar(
      const std::string&, const std::vector<std::string>& encoded_vocab = {}
  );
  static CompiledGrammar DeserializeCompiledGrammarWithTokenizer(
      const std::string&, const TokenizerInfo& tokenizer_info
  );
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SERIALIZER_H_
