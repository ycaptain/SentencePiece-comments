#include "freelist.h"
#include "testharness.h"

namespace sentencepiece {
namespace model {

// DOC:
// 进行freelist相关性能的测试
TEST(FreeListTest, BasicTest) {
    // 测试chunk_size相关
  FreeList<int> l(5);
  EXPECT_EQ(0, l.size());

  constexpr size_t kSize = 32;

  // 测试空间分配相关内容
  for (size_t i = 0; i < kSize; ++i) {
    int *n = l.Allocate();
    EXPECT_EQ(0, *n);
    *n = i;
  }

  // 测试freelist内部元素
  EXPECT_EQ(kSize, l.size());
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(i, *l[i]);
  }

  // 测试释放相关内容
  l.Free();
  EXPECT_EQ(0, l.size());

  // 在释放空间后初始化为0
  // Zero-initialized after `Free`.
  for (size_t i = 0; i < kSize; ++i) {
    int *n = l.Allocate();
    EXPECT_EQ(0, *n);
  }
}
}  // namespace model
}  // namespace sentencepiece
