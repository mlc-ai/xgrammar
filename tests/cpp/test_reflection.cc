#include "test_reflection.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "reflection/reflection.h"

namespace xgrammar {

struct A {
  char a;
  int b;
  double c;
  A(char a, int b, double c) : a(a), b(b), c(c) {}
};

// support array of member pointers
XGRAMMAR_MEMBER_ARRAY(A, &A::a, &A::b, &A::c);

struct B {
  A a;
  std::string str;
  B(A a, std::string str) : a(a), str(str) {}
};

// support table of (name -> member pointer)
XGRAMMAR_MEMBER_TABLE(B, "a", &B::a, "str", &B::str);

template <typename T>
struct C {
  T t;
  C(T t) : t(t) {}
};

// support template types
template <typename T>
XGRAMMAR_MEMBER_ARRAY_TEMPLATE(C<T>, &C<T>::t);

struct D {
  uint16_t data[4];
  // we only care about the lower 32 bits of the uint64_t value
  D(uint64_t x) {
    data[0] = static_cast<uint16_t>(x & 0xFFFF);
    data[1] = static_cast<uint16_t>((x >> 16) & 0xFFFF);
  }
  operator uint64_t() const {
    return static_cast<uint64_t>(data[0]) | (static_cast<uint64_t>(data[1]) << 16);
  }
};

// in delegate mode, we will convert to delegate_type before comparing
XGRAMMAR_MEMBER_DELEGATE(D, uint64_t);

// automatically generate equality and not equal operators for the types

static XGRAMMAR_GENERATE_EQUALITY_DEMO(A);
static XGRAMMAR_GENERATE_EQUALITY_DEMO(B);
template <typename T>
static XGRAMMAR_GENERATE_EQUALITY_DEMO(C<T>);
static XGRAMMAR_GENERATE_EQUALITY_DEMO(D);

// For gtest, we need to define the not-equal operators...
template <typename T>
inline bool operator!=(const T& lhs, const T& rhs) {
  return !(lhs == rhs);
}

}  // namespace xgrammar

TEST(XGrammarReflectionTest, BasicEq) {
  using namespace xgrammar;
  A a1{'x', 42, 3.14};
  A a2{'x', 42, 3.14};
  A a3{'y', 42, 3.14};
  ASSERT_EQ(a1, a2);
  ASSERT_NE(a1, a3);
  B b1{a1, "hello"};
  B b2{a2, "hello"};
  B b3{a3, "hello"};
  B b4{a1, "world"};
  ASSERT_EQ(b1, b2);
  ASSERT_NE(b1, b3);
  ASSERT_NE(b1, b4);

  C<A> c1{a1};
  C<A> c2{a2};
  C<A> c3{a3};

  ASSERT_EQ(c1, c2);
  ASSERT_NE(c1, c3);

  constexpr uint64_t x = 0x123456789ABCDEF0;

  D d1{x};
  D d2{x ^ (1ull << 32)};  // different in the upper 32 bits
  D d3{x ^ 1};             // different in the lower 32 bits

  ASSERT_EQ(d1, d2);
  ASSERT_NE(d1, d3);
}
