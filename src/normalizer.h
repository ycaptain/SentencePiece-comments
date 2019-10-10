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

#ifndef NORMALIZER_NORMALIZER_H_
#define NORMALIZER_NORMALIZER_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/darts_clone/darts.h"

namespace sentencepiece {
namespace normalizer {

// Given a list of strings, finds the longest string which is a
// prefix of a query.
// DOC:
// 在字符串组成的列表中，寻找查询中带有前缀的最长字符串匹配
class PrefixMatcher {
 public:
  // Initializes the PrefixMatcher with `dic`.
  explicit PrefixMatcher(const std::set<absl::string_view> &dic);

  // Finds the longest string in dic, which is a prefix of `w`.
  // Returns the UTF8 byte length of matched string.
  // `found` is set if a prefix match exists.
  // If no entry is found, consumes one Unicode character.
  // DOC:
  // 在 dic 中寻找带有'w'前缀的最长字符串，返回匹配字符串的 UTF8 字节长度
  // 如果前缀匹配成立设置 found 信标，否则消耗一个 Unicode 字符
  int PrefixMatch(absl::string_view w, bool *found = nullptr) const;

  // Replaces entries in `w` with `out`.
  std::string GlobalReplace(absl::string_view w, absl::string_view out) const;

 private:
  std::unique_ptr<Darts::DoubleArray> trie_;
};

// Normalizer implements a simple text normalizer with
// user-defined string-to-string rules and leftmost longest
// matching. The rules of Normalizer are built with
// Builder::CompileCharsMap() method. Pre-compiled rules are
// also available via Builder::GetPrecompiledCharsMap(<name>) method.
//
// The motivation of Normalizer is to make flexible, user-customizable
// and self-contained normalizer.  All the logic of normalization is
// encoded in the model proto which allows us to define language/task
// dependent normalization rules without breaking the default rule.

// DOC:
// 规范化器实现了一个包含用户定义规则与左起最长字符串匹配的简单文本规范化
// 规范化器的规范规则使用工厂方法 Builder::CompileCharsMap() 构建
// 预编码规则也可以通过 Builder::GetPrecompiledCharsMap(<name>) 方法取得
// 规范化器的建立动机是制造一个更加灵活 用户高度定制化 完全独立 的规范化装置
// 规范化的全部逻辑都通过模型原型编码 方便用户在缺省规则外自行定义独立的规范化规则
class Normalizer {
 public:
  // Instantiates Normalizer with |spec|.
  // |spec| should not be deleted until Normalizer is destroyed.

  // DOC:
  // 使用特性初始化规范化器
  // 特性在初始化器消失前不应当被删除
  explicit Normalizer(const NormalizerSpec &spec);
  Normalizer(const NormalizerSpec &spec, const TrainerSpec &trainer_Spec);
  virtual ~Normalizer();

  virtual void SetPrefixMatcher(const PrefixMatcher *matcher) {
    matcher_ = matcher;
  }

  // Returns Status.
  // Normalizes function is valid only when status is OK.
  // DOC:
  // 该函数返回当前状态（仅当状态为 OK 时该函数可用）
  virtual util::Status status() const { return status_; }

  // Normalizes a plain utf8 string into an internal representation for
  // Sentencepiece model. |norm_to_orig| stores the byte-alignment from
  // normalized string to the original input.
  // This function can do the following normalizations:
  // - Character normalization.
  //   (NFKC / full-width to half-width conversion etc).
  // - Adds a prefix space.
  // - Replaces a space with a meta symbol.
  // - Removing heading, tailing and other redundant spaces.

  // DOC:
  // 将一个普通 UTF8 格式字符串规范化为 SentencePiece 模型的内部表示形式。
  // 其中 norm_to_orig 向量存储从规范化字符串到原始输入的字节对齐情况。
  // 此函数可以执行以下规范化操作：
  // - 字符规范化。（NFKC/全宽到半宽转换等）。
  // - 添加前缀空格。
  // - 用元符号（‘_’）替换空格。
  // - 清除头、尾和其他多余空格。
  virtual util::Status Normalize(absl::string_view input,
                                 std::string *normalized,
                                 std::vector<size_t> *norm_to_orig) const;

  // Returns a normalized string without alignments.
  // This function is used in sentencepiece training.

  // DOC:
  // 返回未经对齐的初始化字符串，适用于 sentencepiece 训练
  virtual std::string Normalize(absl::string_view input) const;

  friend class Builder;

 private:
  FRIEND_TEST(NormalizerTest, EncodeDecodePrecompiledCharsMapTest);

  void Init();

  // Normalizes the prefix of |input| and returns the pair of
  // normalized prefix and length we must consume after
  // normalization.

  // DOC:
  // 规范化输入字符串的前缀并且返回规范化后消耗的字符长度与规范化后前缀的 pair

  // Here's the sample code for the full text normalization.
  //
  // string output;
  // absl::string_view input = "...";
  // while (!input.empty()) {
  //   const auto p = normalizer.NormalizePrefix(input);
  //   output.append(p.first.data(), p.first.size());
  //   input.remove_prefix(p.second);
  // }
  std::pair<absl::string_view, int> NormalizePrefix(
      absl::string_view input) const;

  // Encodes trie_blob and normalized string and return compiled blob.
  // DOC:
  // 将 trie_blob 与规范化字符串进行编码，返回编码后的字符串
  static std::string EncodePrecompiledCharsMap(absl::string_view trie_blob,
                                               absl::string_view normalized);

  // Decodes blob into trie_blob and normalized string.
  // DOC:
  // 将编码后的字符串进行解码，返回 trie_blob 与规范化字符串
  static util::Status DecodePrecompiledCharsMap(absl::string_view blob,
                                                absl::string_view *trie_blob,
                                                absl::string_view *normalized);

  // Maximum size of the return value of Trie, which corresponds
  // to the maximum size of shared common prefix in the chars map.
  // DOC:
  // Trie 的最大长度，与字符映射中的共享前缀最大长度保持一致
  static constexpr int kMaxTrieResultsSize = 32;

  // Internal trie for efficient longest matching.
  // DOC:
  // 提升最长匹配效率的内部 Trie
  std::unique_ptr<Darts::DoubleArray> trie_;

  // "\0" delimitered output string.
  // the value of |trie_| stores pointers to this string.
  // DOC:
  // "\0" 作为输出字符串的定界符
  // trie_ 指针指向的值存储该字符串指针
  const char *normalized_ = nullptr;

  // Spec for normalization.
  // DOC:
  // 规范化特性
  const NormalizerSpec *spec_;

  // Prefix matcher;
  // DOC:
  // 对字符串前缀进行匹配
  const PrefixMatcher *matcher_ = nullptr;

  // Split hello world into "hello_" and "world_" instead of
  // "_hello" and "_world".
  // DOC:
  // 将字符串文本中的空格当做后缀
  const bool treat_whitespace_as_suffix_ = false;

  // DOC:
  // 规范化器的状态
  // Normalizer's status.
  util::Status status_;
};
}  // namespace normalizer
}  // namespace sentencepiece
#endif  // NORMALIZER_NORMALIZER_H_
