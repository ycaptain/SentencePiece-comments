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

#ifndef BUILDER_H_
#define BUILDER_H_

#include <map>
#include <string>
#include <vector>
#include "common.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/strings/string_view.h"

namespace sentencepiece {
namespace normalizer {

// DOC:
// Builder通过用户自定义的映射创建文本正规化规则。
// 正规化规则映射被保存到model proto中，并被编译为一个二进制文件。
// 该类提供了基于Unicode NFKC的预定义正规化规则。
// Builder creates a text normalization rule from user-defined string
// to string mappings. The normalization mapping is compiled into
// a single and compact blob index which is stored into the model proto.
// This class also provides pre-defined rules based on Unicode NFKC.
// https://en.wikipedia.org/wiki/Unicode_equivalence#Normalization
class Builder {
 public:
  Builder() = delete;
  ~Builder() = delete;

  // 基础Unicode字符序列
  // Basic Unicode character sequence.
  using Chars = std::vector<char32>;

  // 字符映射键值表
  // String-to-string mapping.
  using CharsMap = std::map<Chars, Chars>;

  // DOC:
  // 编译字符映射键值表为二进制字符序列。
  //
  // 参数:
  //      chars_map -- 字符映射键值表
  //      output -- 输出目标
  //
  // 返回:
  //      编译状态。
  static util::Status CompileCharsMap(const CharsMap &chars_map,
                                      std::string *output);

  // DOC:
  // 将二进制`blob`反编译到`chars_map`。
  //
  // 参数:
  //        blob -- 二进制字符序列
  //        chars_map -- 字符映射键值表
  //
  // 返回:
  //        反编译状态。
  // Decompiles `blob` into `chars_map`.
  static util::Status DecompileCharsMap(absl::string_view blob,
                                        CharsMap *chars_map);

  // DOC:
  // 返回指令对应的二进制表示。
  //
  // 参数:
  //        name -- 指令
  //        output -- 输出目标
  //
  // 返回:
  //        指令对应的二进制表示。
  // Returns a pre-compiled binary index with `name`.
  static util::Status GetPrecompiledCharsMap(const std::string &name,
                                             std::string *output);

  // Makes a normalization mapping based on NFKC.
  //
  // Note that Normalizer/Builder classes do not support
  // full NFKC normalization, since full NFKC normalization cannot
  // be implemented with a simple longest matching string-to-string
  // replacement. One unsupported normalization is multiple combining
  // marks.
  //
  // Strings with multiple combining marks cannot correctly
  // be normalized, because it needs to sort the combining marks
  // with Canonical_Combining_Class (CCC).
  // http://unicode.org/reports/tr15/#Multiple_Mark_Figure
  //
  // Example:
  //  Original:    U+1E0B U+0323
  //  Decomposed:  U+0064 U+0307 U+0323
  //  NFKD:        U+0064 U+0323 U+0307 (Combining characters are sorted by CCC)
  //  NFKC:        U+1E0D U+0307 (U+0064 U+0323 => U+1E0D)
  //
  // To support the normalization above with a longest matching, we need to
  // enumerate all possible permutations of combining marks in advance,
  // which is not feasible. For example, suppose the case there are three
  // combining marks X, Y and Z, which are sorted into one canonical order
  // Z, Y, X with NFK(D|C).  In this case, all permutations (XYZ, XZY, YXZ...)
  // are normalized into ZYX. When we implement this normalization with
  // a longest matching, we need to have 3! rules. XYZ=>ZYX, XZY=>ZYX..
  // Since Unicode has more than 100 combining characters, it is not possible
  // to expand all permutations.
  //
  // We will not implement the full NFKC in SentencePiece because
  //  1) It is unusual to see decomposed Unicode characters in real text.
  //  2) Providing a flexible, user-customizable, and self-contained
  //     normalizer is the goal of SentencePiece.
  //
  // TODO(taku): Make NFC, NFD, and NFKD mapping if necessary.
  // DOC:
  // 链接NFKC字符映射键值表。
  //
  // 参数:
  //      chars_map -- 字符映射键值表
  //
  // 返回:
  //      链接状态。
  static util::Status BuildNFKCMap(CharsMap *chars_map);

  // DOC:
  // 链接Nmt NFKC字符映射键值表。
  //
  // 参数:
  //      chars_map -- 字符映射键值表
  //
  // 返回:
  //      链接状态。
  // Makes an NFKC-based mapping with NMT specific modifications around
  // whitespaces.
  static util::Status BuildNmtNFKCMap(CharsMap *chars_map);

  // DOC:
  // 合并Unicode大小写折叠到`chars_map`。
  //
  // 参数:
  //        chars_map -- 字符映射键值表
  //
  // 返回:
  //        合并状态。
  // Merge Unicode case folding mapping into `chars_map`.
  static util::Status MergeUnicodeCaseFoldMap(CharsMap *chars_map);

  // DOC:
  // 根据Unicode大小写折叠创建NFKC字符映射键值表。
  //
  // 参数:
  //        chars_map -- 字符映射键值表
  //
  // 返回:
  //        创建状态。
  // Makes NFKC with Unicode case folding.
  static util::Status BuildNFKC_CFMap(CharsMap *chars_map);

  // DOC:
  // 根据Unicode大小写折叠创建Nmt NFKC字符映射键值表
  //
  // 参数:
  //        chars_map -- 字符映射键值表
  //
  // 返回:
  //        创建状态。
  // Makes NMT NFKC with Unicode case folding.
  static util::Status BuildNmtNFKC_CFMap(CharsMap *chars_map);

  // DOC:
  // 从文件中载入字符映射键值表。
  //
  // 参数:
  //        filename -- 源文件路径
  //        chars_map -- 用于保存字符映射键值表
  //
  // 返回:
  //        载入状态。
  // Builds Chars map save in `filename`.
  // Format:
  // src_uchar1 src_uchar2 ... <tab> trg_uchar1 trg_uchar2...
  // (src|trg)_ucharX must be a LoadCharsMaphex of Unicode code point.
  static util::Status LoadCharsMap(absl::string_view filename,
                                   CharsMap *chars_map);

  // DOC:
  // 保存字符映射键值表到tsv文件。
  //
  // 参数:
  //        filename -- 保存文件路径
  //        chars_map -- 字符映射键值表
  //
  // 返回:
  //        保存状态。
  // Saves Chars map to `filename` as TSV.
  static util::Status SaveCharsMap(absl::string_view filename,
                                   const CharsMap &chars_map);

 private:
  // DOC:
  // 进行全局友类测试
  FRIEND_TEST(BuilderTest, RemoveRedundantMapTest);

  // DOC:
  // 移除字符映射键值表中的冗余规则。
  //
  // 参数:
  //        chars_map -- 字符映射键值表
  //
  // 返回:
  //        移除状态。
  // Removes redundant rules from `chars_map`.
  // When char_maps have "aa" => "bb" and "a" => "b", the first
  // rule is not necessary since the second rule can cover the first rule.
  static util::Status RemoveRedundantMap(CharsMap *chars_map);
};
}  // namespace normalizer
}  // namespace sentencepiece
#endif  // BUILDER_H_
