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

#ifndef BPE_MODEL_TRAINER_H_
#define BPE_MODEL_TRAINER_H_

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "sentencepiece_model.pb.h"
#include "trainer_interface.h"

namespace sentencepiece {
namespace bpe {

// BPE模型的训练器类。
// Trainer class for BPE model.
class Trainer : public TrainerInterface {
 public:
  Trainer(const TrainerSpec &trainer_spec,
          const NormalizerSpec &normalizer_spec)
      : TrainerInterface::TrainerInterface(trainer_spec, normalizer_spec) {}

  util::Status Train() override;

 private:
  // 符号代表着一(一元符号)或两个单字(二元符号)。
  // Symbol represents a character or symbol bigram.
  struct Symbol {
	// 此二元模型中左侧的符号。
    const Symbol *left;              // left symbol in bigram
	// 此二元模型中右侧的符号。
    const Symbol *right;             // right symbol in bigram
	// 所有压平的字序列。
    string_util::UnicodeText chars;  // all flattend chracter sequence
	// 这个符号是否是未知字。
    bool is_unk;                     // true if this symbol is unknown.
	// 该符号的指纹。
    uint64 fp;                       // fingerprint of this symbol.
	// 该符号出现的频数。
    uint64 freq;                     // frequency of this symbol.

	// 该符号出现过的位置列表。存储的是编码后的位置形式。(见下)
    // Position list. Use set so that we can keep the order of occurrence.
    // See EncodePos/DecodePos.
    std::set<uint64> positions;

	// DOC：
	// 返回该符号是否是二元符号(左符号不为空且右符号不为空)。
    bool IsBigram() const { return left != nullptr && right != nullptr; }
    std::string ToString() const;
    Symbol() : left(nullptr), right(nullptr), is_unk(false), fp(0), freq(0) {}
  };

  // [prev, left], [left, right], [right, next]各组成一个二元符号
  struct Position {
	// 句子id
    int sid;    // sentence id
	// 左符号索引
    int left;   // left symbol index
	// 右符号索引
    int right;  // right symbol index
  };

  // 将Position编码成64位无符号整数。
  // 编码后的值的结构：前32位表示sentence id，中间16位表示left symbol index，后16位表示right symbol index。
  // Encodes sid, left and right bigram index into uint64.
  // Encoded value keeps the order of sid, left and right.
  static uint64 EncodePos(int sid, int l, int r) {
    CHECK_GE(l, 0);
    CHECK_GE(r, 0);
    CHECK_LE(l, kuint16max);
    CHECK_LE(r, kuint16max);
    const uint64 n = (static_cast<uint64>(sid) << 32 | (l << 16 | r));
    return n;
  }

  // 从64位无符号整数解码并返回Position。
  // Decodes sid, left and right bigram index from uint64.
  static Position DecodePos(uint64 n) {
    Position p;
    p.sid = n >> 32;
    p.left = (n >> 16) & 0xffff;
    p.right = n & 0xffff;
    return p;
  }

  // 从指纹获得对应的一元符号。
  // 返回值如果不在symbols_cache_中则会被尝试插入symbols_cache_，如果已存在则直接返回。
  // Gets unary (character) symbol from the char code |c|.
  // The return value is cached.
  Symbol *GetCharSymbol(char32 c);

  // 从左符号、右符号查找二元符号。
  // 返回值如果不在symbols_cache_中则会被尝试插入symbols_cache_，如果已存在则直接返回。
  // Gets symbol pair from left/right symbols. The return value is cached.
  Symbol *GetPairSymbol(const Symbol *left, const Symbol *right);

  // 计算符号的出现频数并更新符号的频数字段。
  // Computes the frequency of |symbol| and update symbol->freq field.
  void ComputeFreq(Symbol *symbol) const;

  //  返回symbols_[sid][index]代表的符号的下一个有效符号的索引。不存在则返回-1。
  // Returns the valid index before symbols_[sid][index].
  int GetNextIndex(int sid, int index) const;

  //  返回symbols_[sid][index]代表的符号的前一个有效符号的索引。不存在则返回-1。
  // Returns the valid index after symbols_[sid][index].
  int GetPrevIndex(int sid, int index) const;

  // 从指定的句子和相邻索引中创建一个新的二元符号并将它添加至symbols_cache_和active_symbols_。
  // Makes a new bigram from [symbols_[sid][left], symbols_[sid][right]] and
  // Adds it to symbols_cache_ and active_symbols_.
  void AddNewPair(int sid, int left, int right);

  // 若指定的句子和相邻索引所代表的符号与指定的符号best不同，则重设句子中的那个符号的出现频数为0。
  // 参数：
  //       sid, left, right --- 用来指定一个想要重设的二元符号。
  // Resets the fequency of bigram [symbols_[sid][left] symbols_[sid][right]],
  // if this bigram is not |best|.
  void ResetFreq(int sid, int left, int right, const Symbol *best);

  // 复制出现频数在前5%的symbols_cache_中的符号来更新active_symbols_。
  // Updates |active_symbols_| by copying the top 5% frequent symbols in
  // symbols_cache_.
  void UpdateActiveSymbols();

  // 所有不重复的符号集。以符号的指纹为键。
  // All unique symbols. Key is a fingerprint of Symbol.
  std::unordered_map<uint64, Symbol *> symbols_cache_;

  // 出现频数最高的符号，意味着它们最有可能是词汇。
  // Set of symbols from which we find the best symbol in each iteration.
  std::set<Symbol *> active_symbols_;

  // 存储堆中分配过的符号指针一遍一次性清理。
  // Stores symbols allocated in heap so that we can delete them at onece.
  std::vector<Symbol *> allocated_;

  // 句子集。symbols[sid]代表一个句子，symbols[sid][index]存储着一个句id为sid的句子的index位置上的符号。
  // Sentences. symbols_[sid][index] stores a symbol in sentence_[sid][index].
  std::vector<std::vector<Symbol *>> symbols_;
};
}  // namespace bpe
}  // namespace sentencepiece
#endif  // BPE_MODEL_TRAINER_H_
