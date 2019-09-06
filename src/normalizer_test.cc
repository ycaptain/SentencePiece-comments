// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include "normalizer.h"

#include <vector>
#include "builder.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "util.h"

namespace sentencepiece {
namespace normalizer {
namespace {
// DOC:
// 将要替换空格的 ▁ 字符，Lower One Eighth Block，U+2581
// Space symbol
#define WS "\xe2\x96\x81"

// DOC:
// Unicode Specials 区的 Replacement Char 替换字符 �，用来替换无法辨别的字符，U+FFFD
// Replacement char
#define RC "\xEF\xBF\xBD"

// DOC:
// 创建规范化器规格实例，使用 NFKC 形式的 Unicode 规范化
NormalizerSpec MakeDefaultSpec() {
  return SentencePieceTrainer::GetNormalizerSpec("nmt_nfkc");
}
}  // namespace

// DOC:
// 进行规范化器测试
TEST(NormalizerTest, NormalizeTest) {
  auto spec = MakeDefaultSpec();
  const Normalizer normalizer(spec);

  // DOC:
  // 测试由空格构成的字符串与空字符串的等价性
  // Empty strings.
  EXPECT_EQ("", normalizer.Normalize(""));
  EXPECT_EQ("", normalizer.Normalize("      "));
  EXPECT_EQ("", normalizer.Normalize("　"));

  // DOC:
  // 测试字符串行尾存在冗余空格的等价性
  // Sentence with heading/tailing/redundant spaces.
  EXPECT_EQ(WS "ABC", normalizer.Normalize("ABC"));
  EXPECT_EQ(WS "ABC", normalizer.Normalize(" ABC "));
  EXPECT_EQ(WS "A" WS "B" WS "C", normalizer.Normalize(" A  B  C "));
  EXPECT_EQ(WS "ABC", normalizer.Normalize("   ABC   "));
  EXPECT_EQ(WS "ABC", normalizer.Normalize("   ＡＢＣ   "));
  EXPECT_EQ(WS "ABC", normalizer.Normalize("　　ABC"));
  EXPECT_EQ(WS "ABC", normalizer.Normalize("　　ABC　　"));

  // DOC:
  // 测试 NFKC 形式字符串的逐字规范化的 Unicode 标准等价性，①②③ 规范化为 123
  // NFKC char to char normalization.
  EXPECT_EQ(WS "123", normalizer.Normalize("①②③"));

  // DOC:
  // 测试 NFKC 形式字符串的多字节规范化的 Unicode 兼容等价性，Unicode 合字 ㍿ 规范化为 株式会社
  // NFKC char to multi-char normalization.
  EXPECT_EQ(WS "株式会社", normalizer.Normalize("㍿"));

  // DOC:
  // 测试半角片假名的规范化，半角浊化符号规范化时发生字符合成
  // Half width katakana, character composition happens.
  EXPECT_EQ(WS "グーグル", normalizer.Normalize(" ｸﾞｰｸﾞﾙ "));

  // DOC:
  // 测试英文文本的规范化
  EXPECT_EQ(WS "I" WS "saw" WS "a" WS "girl",
            normalizer.Normalize(" I  saw a　 　girl　　"));

  // DOC:
  // 将 Unicode 中的控制字符规范化为空字符串
  // Remove control chars.
  EXPECT_EQ("", normalizer.Normalize(string_util::UnicodeCharToUTF8(0x7F)));
  EXPECT_EQ("", normalizer.Normalize(string_util::UnicodeCharToUTF8(0x8F)));
  EXPECT_EQ("", normalizer.Normalize(string_util::UnicodeCharToUTF8(0x9F)));
  EXPECT_EQ("", normalizer.Normalize(string_util::UnicodeCharToUTF8(0x0B)));
  for (char32 c = 0x10; c <= 0x1F; ++c) {
    EXPECT_EQ("", normalizer.Normalize(string_util::UnicodeCharToUTF8(c)));
  }
}

// DOC:
// 同上；启用 set_add_dummy_prefix，前缀添加特殊 WS 符
TEST(NormalizerTest, NormalizeWithoutDummyPrefixTest) {
  auto spec = MakeDefaultSpec();
  spec.set_add_dummy_prefix(false);
  const Normalizer normalizer(spec);

  // Empty strings.
  EXPECT_EQ("", normalizer.Normalize(""));
  EXPECT_EQ("", normalizer.Normalize("      "));
  EXPECT_EQ("", normalizer.Normalize("　"));

  // Sentence with heading/tailing/redundant spaces.
  EXPECT_EQ("ABC", normalizer.Normalize("ABC"));
  EXPECT_EQ("ABC", normalizer.Normalize(" ABC "));
  EXPECT_EQ("A" WS "B" WS "C", normalizer.Normalize(" A  B  C "));
  EXPECT_EQ("ABC", normalizer.Normalize("   ABC   "));
  EXPECT_EQ("ABC", normalizer.Normalize("   ＡＢＣ   "));
  EXPECT_EQ("ABC", normalizer.Normalize("　　ABC"));
  EXPECT_EQ("ABC", normalizer.Normalize("　　ABC　　"));
}

// DOC:
// 同上；启用 set_treat_whitespace_as_suffix，分词时将单词空格视为后缀
TEST(NormalizerTest, NormalizeTreatWSAsSuffixTest) {
  auto spec = MakeDefaultSpec();
  TrainerSpec trainer_spec;
  trainer_spec.set_treat_whitespace_as_suffix(true);
  const Normalizer normalizer(spec, trainer_spec);

  EXPECT_EQ("", normalizer.Normalize(""));
  EXPECT_EQ("", normalizer.Normalize("      "));
  EXPECT_EQ("", normalizer.Normalize("　"));

  EXPECT_EQ("ABC" WS, normalizer.Normalize("ABC"));
  EXPECT_EQ("ABC" WS, normalizer.Normalize(" ABC "));
  EXPECT_EQ("A" WS "B" WS "C" WS, normalizer.Normalize(" A  B  C "));
  EXPECT_EQ("ABC" WS, normalizer.Normalize("   ABC   "));
}

// DOC:
// 同上；规范化时不移除额外的字符串中空格
TEST(NormalizerTest, NormalizeWithoutRemoveExtraWhitespacesTest) {
  auto spec = MakeDefaultSpec();
  spec.set_remove_extra_whitespaces(false);
  const Normalizer normalizer(spec);

  // Empty strings.
  EXPECT_EQ("", normalizer.Normalize(""));
  EXPECT_EQ(WS WS WS WS WS WS WS, normalizer.Normalize("      "));
  EXPECT_EQ(WS WS, normalizer.Normalize("　"));

  // Sentence with heading/tailing/redundant spaces.
  EXPECT_EQ(WS "ABC", normalizer.Normalize("ABC"));
  EXPECT_EQ(WS WS "ABC" WS, normalizer.Normalize(" ABC "));
  EXPECT_EQ(WS WS WS "A" WS WS "B" WS WS "C" WS WS,
            normalizer.Normalize("  A  B  C  "));
}

// DOC:
// 同上；规范化时规格不设置转义字符
TEST(NormalizerTest, NormalizeWithoutEscapeWhitespacesTest) {
  auto spec = MakeDefaultSpec();
  spec.set_add_dummy_prefix(false);
  spec.set_remove_extra_whitespaces(true);
  spec.set_escape_whitespaces(false);
  const Normalizer normalizer(spec);

  // Empty strings.
  EXPECT_EQ("", normalizer.Normalize(""));
  EXPECT_EQ("", normalizer.Normalize("      "));
  EXPECT_EQ("", normalizer.Normalize("　"));

  // Sentence with heading/tailing/redundant spaces.
  EXPECT_EQ("ABC", normalizer.Normalize("ABC"));
  EXPECT_EQ("ABC", normalizer.Normalize(" ABC "));
  EXPECT_EQ("A B C", normalizer.Normalize("  A  B  C  "));
  EXPECT_EQ("A B C", normalizer.Normalize("A　 B　 C"));
}

// DOC:
// 规范化规格包含空格字符
TEST(NormalizeTest, NomalizeWithSpaceContainedRules) {
  Builder::CharsMap charsmap;

  // DOC:
  // 建立 lambda 函数添加规范化规格规则
  // 参数:
  //        src -- 源字符串
  //        trg -- 目标字符串
  auto AddRule = [&](const std::string &src, const std::string &trg) {
    Builder::Chars src_chars, trg_chars;
    for (const char32 c : string_util::UTF8ToUnicodeText(src)) {
      src_chars.push_back(c);
    }
    for (const char32 c : string_util::UTF8ToUnicodeText(trg)) {
      trg_chars.push_back(c);
    }
    charsmap[src_chars] = trg_chars;
  };

  // Adds rules containing whitespaes.
  AddRule("a", " A");
  AddRule("b", "B");
  AddRule("c", "D E");
  AddRule("d", " F G ");

  NormalizerSpec spec;
  EXPECT_OK(
      Builder::CompileCharsMap(charsmap, spec.mutable_precompiled_charsmap()));

  // DOC:
  // 添加规格规则后测试默认行为
  // Test default behavior
  {
    const Normalizer normalizer(spec);
    EXPECT_EQ(WS "A", normalizer.Normalize("a"));
    EXPECT_EQ(WS "B" WS "A", normalizer.Normalize("ba"));
    EXPECT_EQ(WS "D" WS "E", normalizer.Normalize("c"));
    EXPECT_EQ(WS "F" WS "G" WS "A", normalizer.Normalize("da"));
    EXPECT_EQ(WS "A" WS "F" WS "G", normalizer.Normalize("ad"));
    EXPECT_EQ(WS "A" WS "F" WS "G" WS "B", normalizer.Normalize("adb"));
  }

  spec.set_escape_whitespaces(false);
  {
    spec.set_add_dummy_prefix(false);
    spec.set_remove_extra_whitespaces(true);

    const Normalizer normalizer(spec);
    EXPECT_EQ("A", normalizer.Normalize("a"));
    EXPECT_EQ("B A", normalizer.Normalize("ba"));
    EXPECT_EQ("D E", normalizer.Normalize("c"));
    EXPECT_EQ("F G A", normalizer.Normalize("da"));
    EXPECT_EQ("A F G", normalizer.Normalize("ad"));
    EXPECT_EQ("A F G B", normalizer.Normalize("adb"));
  }

  {
    spec.set_add_dummy_prefix(false);
    spec.set_remove_extra_whitespaces(false);

    const Normalizer normalizer(spec);
    EXPECT_EQ(" A", normalizer.Normalize("a"));
    EXPECT_EQ("B A", normalizer.Normalize("ba"));
    EXPECT_EQ("D E", normalizer.Normalize("c"));
    EXPECT_EQ(" F G  A", normalizer.Normalize("da"));
    EXPECT_EQ(" A F G ", normalizer.Normalize("ad"));
    EXPECT_EQ(" A F G B", normalizer.Normalize("adb"));
  }

  {
    spec.set_add_dummy_prefix(true);
    spec.set_remove_extra_whitespaces(true);

    const Normalizer normalizer(spec);
    EXPECT_EQ(" A", normalizer.Normalize("a"));
    EXPECT_EQ(" B A", normalizer.Normalize("ba"));
    EXPECT_EQ(" D E", normalizer.Normalize("c"));
    EXPECT_EQ(" F G A", normalizer.Normalize("da"));
    EXPECT_EQ(" A F G", normalizer.Normalize("ad"));
    EXPECT_EQ(" A F G B", normalizer.Normalize("adb"));
  }

  {
    spec.set_add_dummy_prefix(true);
    spec.set_remove_extra_whitespaces(false);

    const Normalizer normalizer(spec);
    EXPECT_EQ("  A", normalizer.Normalize("a"));
    EXPECT_EQ(" B A", normalizer.Normalize("ba"));
    EXPECT_EQ(" D E", normalizer.Normalize("c"));
    EXPECT_EQ("  F G  A", normalizer.Normalize("da"));
    EXPECT_EQ("  A F G ", normalizer.Normalize("ad"));
    EXPECT_EQ("  A F G B", normalizer.Normalize("adb"));
  }

  // DOC:
  // 添加部分与空格匹配有关的边界案例
  // Added several corner cases around spaces.
  struct SpacePattern {
    bool add_dummy_prefix;
    bool remove_extra_whitespaces;
    bool escape_whitespaces;
    const char *input;
    const char *expected;
  };

  constexpr SpacePattern kSpacePatternData[] = {
      {false, false, false, WS, WS},    {false, false, true, WS, WS},
      {false, true, false, WS, WS},     {false, true, true, WS, ""},
      {true, false, false, WS, " " WS}, {true, false, true, WS, WS WS},
      {true, true, false, WS, " " WS},  {true, true, true, WS, ""},
      {false, false, false, " ", " "},  {false, false, true, " ", WS},
      {false, true, false, " ", ""},    {false, true, true, " ", ""},
      {true, false, false, " ", "  "},  {true, false, true, " ", WS WS},
      {true, true, false, " ", ""},     {true, true, true, " ", ""}};

  for (const auto &c : kSpacePatternData) {
    spec.set_add_dummy_prefix(c.add_dummy_prefix);
    spec.set_remove_extra_whitespaces(c.remove_extra_whitespaces);
    spec.set_escape_whitespaces(c.escape_whitespaces);
    const Normalizer normalizer(spec);
    EXPECT_EQ(c.expected, normalizer.Normalize(c.input));
  }
}

// DOC:
// 同上；规范化测试包括与此前替换字符有关的案例
TEST(NormalizerTest, NormalizeReplacementChar) {
  auto spec = MakeDefaultSpec();
  spec.set_add_dummy_prefix(false);
  const Normalizer normalizer(spec);
  EXPECT_EQ("abc" RC "xy", normalizer.Normalize("abc\x80xy"));
  EXPECT_EQ("abc" RC, normalizer.Normalize("abc\xc3"));
  EXPECT_EQ("ab" RC RC "xy", normalizer.Normalize("ab\xe3\x81xy"));
  EXPECT_EQ("a" RC RC RC "xy", normalizer.Normalize("a\xf3\x81\x81xy"));
  EXPECT_EQ("ab" RC RC "xy", normalizer.Normalize("ab\xc0\x82xy"));
}

// DOC:
// 完整的规范化测试
TEST(NormalizerTest, NormalizeFullTest) {
  std::vector<size_t> n2i;
  std::string output;

  auto spec = MakeDefaultSpec();
  const Normalizer normalizer(spec);

  {
    const std::string input = "I saw a girl";
    normalizer.Normalize(input, &output, &n2i);
    EXPECT_EQ(WS "I" WS "saw" WS "a" WS "girl", output);
    const std::vector<size_t> expected = {0, 0, 0,       // WS (3byte)
                                          0,             // I
                                          1, 1, 1,       // WS
                                          2, 3, 4,       // saw
                                          5, 5, 5,       // WS
                                          6,             // a
                                          7, 7, 7,       // WS
                                          8, 9, 10, 11,  // girl
                                          12};
    EXPECT_EQ(expected, n2i);
  }

  {
    const std::string input = " I   saw a　 　girl　　";
    EXPECT_OK(normalizer.Normalize(input, &output, &n2i));
    LOG(INFO) << output;
    EXPECT_EQ(WS "I" WS "saw" WS "a" WS "girl", output);
    const std::vector<size_t> expected = {1,  1,  1,       // WS (3byte)
                                          1,               // I
                                          2,  2,  2,       // WS
                                          5,  6,  7,       // saw
                                          8,  8,  8,       // WS
                                          9,               // a
                                          10, 10, 10,      // WS
                                          17, 18, 19, 20,  // girl
                                          21};
    EXPECT_EQ(expected, n2i);
  }

  {
    const std::string input = " ｸﾞｰｸﾞﾙ ";  // halfwidth katakana
    normalizer.Normalize(input, &output, &n2i);
    EXPECT_EQ(WS "グーグル", output);
    const std::vector<size_t> expected = {1,  1,  1,   // WS (3byte)
                                          1,  1,  1,   // グ
                                          7,  7,  7,   // ー
                                          10, 10, 10,  // グ
                                          16, 16, 16,  // ル
                                          19};
    EXPECT_EQ(expected, n2i);
  }

  {
    const std::string input = "①②③";
    normalizer.Normalize(input, &output, &n2i);
    EXPECT_EQ(WS "123", output);
    const std::vector<size_t> expected = {0, 0, 0,  // WS (3byte)
                                          0,        // 1
                                          3,        // 2
                                          6,        // 3
                                          9};
    EXPECT_EQ(expected, n2i);
  }

  {
    const std::string input = "㍿";
    normalizer.Normalize(input, &output, &n2i);
    EXPECT_EQ(WS "株式会社", output);
    const std::vector<size_t> expected = {0, 0, 0,  // WS (3byte)
                                          0, 0, 0,  // 株
                                          0, 0, 0,  // 式
                                          0, 0, 0,  // 会
                                          0, 0, 0,  // 社
                                          3};
    // When "株式" is one piece, this has no alignment to the input.
    // Sentencepieces which includes the last character ("会社" or "社")
    // have the alignment to the input.
    EXPECT_EQ(expected, n2i);
  }
}

// DOC:
// 同上；使用指定规则的预编译字符对应表进行规范化测试
TEST(NormalizerTest, EncodeDecodePrecompiledCharsMapTest) {
  const std::string blob = Normalizer::EncodePrecompiledCharsMap("foo", "bar");
  absl::string_view trie_blob, normalized_blob;
  EXPECT_OK(Normalizer::DecodePrecompiledCharsMap(blob, &trie_blob,
                                                  &normalized_blob));
  EXPECT_EQ("foo", trie_blob);
  EXPECT_EQ("bar", normalized_blob);

  EXPECT_NOT_OK(
      Normalizer::DecodePrecompiledCharsMap("", &trie_blob, &normalized_blob));
}

// DOC:
// 进行规范化状态测试
TEST(NormalizerTest, StatusTest) {
  NormalizerSpec spec;
  {
    const Normalizer normalizer(spec);
    EXPECT_OK(normalizer.status());  // fallback to identity.
  }

  {
    spec.set_precompiled_charsmap("x");
    const Normalizer normalizer(spec);
    EXPECT_FALSE(normalizer.status().ok());
  }

  spec = MakeDefaultSpec();
  {
    const Normalizer normalizer(spec);
    EXPECT_TRUE(normalizer.status().ok());
  }
}

// DOC:
// 规范化的前缀匹配测试
TEST(NormalizerTest, PrefixMatcherTest) {
  const PrefixMatcher matcher({"abc", "ab", "xy", "京都"});
  bool found;
  EXPECT_EQ(1, matcher.PrefixMatch("test", &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(3, matcher.PrefixMatch("abcd", &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(2, matcher.PrefixMatch("abxy", &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(1, matcher.PrefixMatch("x", &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(2, matcher.PrefixMatch("xyz", &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(6, matcher.PrefixMatch("京都大学", &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(3, matcher.PrefixMatch("東京大学", &found));
  EXPECT_FALSE(found);

  EXPECT_EQ("", matcher.GlobalReplace("", ""));
  EXPECT_EQ("", matcher.GlobalReplace("abc", ""));
  EXPECT_EQ("--de-pqr", matcher.GlobalReplace("xyabcdeabpqr", "-"));
}

// DOC:
// 规范化的前缀匹配测试，匹配规则为空
TEST(NormalizerTest, PrefixMatcherWithEmptyTest) {
  const PrefixMatcher matcher({});
  bool found;
  EXPECT_EQ(1, matcher.PrefixMatch("test", &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(1, matcher.PrefixMatch("abcd", &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(1, matcher.PrefixMatch("abxy", &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(1, matcher.PrefixMatch("x", &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(1, matcher.PrefixMatch("xyz", &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(3, matcher.PrefixMatch("京都大学", &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(3, matcher.PrefixMatch("東京大学", &found));
  EXPECT_FALSE(found);

  EXPECT_EQ("", matcher.GlobalReplace("", ""));
  EXPECT_EQ("abc", matcher.GlobalReplace("abc", ""));
}

}  // namespace normalizer
}  // namespace sentencepiece
