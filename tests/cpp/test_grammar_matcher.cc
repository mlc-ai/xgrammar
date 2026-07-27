#include <dlpack/dlpack.h>
#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

DLTensor MakeTensor(
    void* data, int32_t ndim, DLDataType dtype, int64_t* shape, uint64_t byte_offset = 0
) {
  return DLTensor{data, DLDevice{kDLCPU, 0}, ndim, dtype, shape, nullptr, byte_offset};
}

}  // namespace

TEST(XGrammarMatcherTest, TraverseDraftTreeHonorsBitmaskByteOffset) {
  using namespace xgrammar;

  const TokenizerInfo tokenizer_info({"a", "b", "c"}, VocabType::RAW, 3, std::vector<int32_t>{});
  GrammarCompiler compiler(tokenizer_info);
  const CompiledGrammar grammar = compiler.CompileGrammar(R"(root ::= "ab")");

  std::array<int64_t, 3> retrieve_next_token = {1, 2, -1};
  std::array<int64_t, 3> retrieve_next_sibling = {-1, -1, -1};
  std::array<int64_t, 3> draft_tokens = {2, 0, 1};
  std::array<int64_t, 1> tree_shape = {3};
  std::array<int64_t, 2> bitmask_shape = {3, GetBitmaskSize(3)};

  DLTensor next_token_tensor =
      MakeTensor(retrieve_next_token.data(), 1, DLDataType{kDLInt, 64, 1}, tree_shape.data());
  DLTensor next_sibling_tensor =
      MakeTensor(retrieve_next_sibling.data(), 1, DLDataType{kDLInt, 64, 1}, tree_shape.data());
  DLTensor draft_token_tensor =
      MakeTensor(draft_tokens.data(), 1, DLDataType{kDLInt, 64, 1}, tree_shape.data());

  std::array<int32_t, 3> reference_storage = {};
  DLTensor reference_bitmask =
      MakeTensor(reference_storage.data(), 2, GetBitmaskDLType(), bitmask_shape.data());

  constexpr int32_t kGuardValue = 0x12345678;
  std::array<int32_t, 4> offset_storage = {kGuardValue, 0, 0, 0};
  DLTensor offset_bitmask = MakeTensor(
      offset_storage.data(), 2, GetBitmaskDLType(), bitmask_shape.data(), sizeof(offset_storage[0])
  );

  GrammarMatcher reference_matcher(grammar);
  GrammarMatcher offset_matcher(grammar);
  ASSERT_TRUE(reference_matcher.TraverseDraftTree(
      &next_token_tensor, &next_sibling_tensor, &draft_token_tensor, &reference_bitmask
  ));
  ASSERT_TRUE(offset_matcher.TraverseDraftTree(
      &next_token_tensor, &next_sibling_tensor, &draft_token_tensor, &offset_bitmask
  ));

  EXPECT_EQ(offset_storage[0], kGuardValue);
  EXPECT_EQ(
      std::vector<int32_t>(offset_storage.begin() + 1, offset_storage.end()),
      std::vector<int32_t>(reference_storage.begin(), reference_storage.end())
  );
}
