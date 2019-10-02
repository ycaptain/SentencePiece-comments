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

#include "unigram_model.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/strings/string_view.h"
#include "util.h"

// DOC:命名空间 sentencepiece::unigram

namespace sentencepiece {
namespace unigram {
namespace {

// DOC:
// 将Lattice中的提前分配结点数预设为1024
// Size of nodes pre-allocated in Lattice.
constexpr size_t kPreallocateLatticeNodeSize = 1024;

// DOC:
//      根据输入模式返回log(exp(x) + exp(y))
//
// 参数:
//      x,y -- 计算所需浮点数
//      init_mode -- 输入模式(0:返回y;1:返回log(exp(x) + exp(y)))
// Returns log(exp(x) + exp(y)).
// if init_mode is true, returns log(exp(y)) == y.
// log(\sum_i exp(a[i])) can be computed as
// for (int i = 0; i < a.size(); ++i)
//   x = LogSumExp(x, a[i], i == 0);
inline float LogSumExp(float x, float y, bool init_mode) {
  if (init_mode) {
    return y;
  }
  const float vmin = std::min(x, y);
  const float vmax = std::max(x, y);
  constexpr float kMinusLogEpsilon = 50;
  if (vmax > vmin + kMinusLogEpsilon) {
    return vmax;
  } else {
    return vmax + log(exp(vmin - vmax) + 1.0);
  }
}
}  // namespace

// DOC:
//      对Lattice的初始化
Lattice::Lattice() : node_allocator_(kPreallocateLatticeNodeSize) {}
Lattice::~Lattice() {}

// DOC:
// 返回begin_nodes_中在pos处开始的结点组
//
// 参数:
//    pos -- 表示开始位置的参数
const std::vector<Lattice::Node *> &Lattice::begin_nodes(int pos) const {
  return begin_nodes_[pos];
}

// DOC:
// 返回在pos处结束的结点组
//
// 参数:
//    pos -- 表示结束位置的参数
const std::vector<Lattice::Node *> &Lattice::end_nodes(int pos) const {
  return end_nodes_[pos];
}

// DOC:
// 返回Unicode编码下的长度
int Lattice::size() const {
  // DOC:
  // 对 size - 1 是由于 surface_ 末尾存在终止字符
  // -1 because surface_ may include the EOS.
  return std::max<int>(0, surface_.size() - 1);
}

// DOC:
// 返回多字节字符编码下(utf8)的长度
int Lattice::utf8_size() const { return sentence_.size(); }

// DOC:
// 返回整个当前句子的头指针
// 相当于surface(0)
const char *Lattice::sentence() const { return sentence_.data(); }

// DOC:
// 返回当前句子从pos位至结尾的子串
// 相当于对句子取从pos为至结尾的切片(sentence[pos:])
// 参数:
//      pos -- 表示sentence切片的起始位置
// 返回:
//      从pos为至结尾的切片 -- sentence[pos:]
const char *Lattice::surface(int pos) const { return surface_[pos]; }

// DOC:
// 返回起始符指针
Lattice::Node *Lattice::bos_node() const { return end_nodes_[0][0]; }

// DOC:
// 返回终止符指针
Lattice::Node *Lattice::eos_node() const { return begin_nodes_[size()][0]; }

// DOC:
// 分配新结点空间并赋予其ID
//
// 返回:
//      新结点的指针
Lattice::Node *Lattice::NewNode() {
  Node *node = node_allocator_.Allocate();
  node->node_id = node_allocator_.size() - 1;
  return node;
}

// DOC:
// 清除lattice
void Lattice::Clear() {
  begin_nodes_.clear();
  end_nodes_.clear();
  sentence_ = absl::string_view("");
  surface_.clear();
  node_allocator_.Free();
}

// DOC:
// 根据一个string_view类型的sentence 创建一个Lattice对象
void Lattice::SetSentence(absl::string_view sentence) {
  Clear();

  sentence_ = sentence;
  surface_.reserve(sentence.size() + 1);

  // DOC:
  // 根据 sentence 处理 surface_
  // 将 sentence 插入 surface_ 尾部 并且每次操作后移除 sentence 句首的一个字符
  while (!sentence.empty()) {
    const int mblen = std::min<int>(string_util::OneCharLen(sentence.data()),
                                    sentence.size());
    surface_.push_back(sentence.data());
    sentence.remove_prefix(mblen);
  }
  surface_.push_back(sentence.data());

  // DOC:
  // 初始化begin_nodes_ end_nodes_的空间
  const int len = size();
  begin_nodes_.resize(len + 1);
  end_nodes_.resize(len + 1);

  constexpr size_t kReservedNodeSize = 16;
  for (int i = 0; i <= len; ++i) {
    begin_nodes_[i].reserve(kReservedNodeSize);
    end_nodes_[i].reserve(kReservedNodeSize);
  }

  // DOC:
  // 加入起始符与终止符
  Node *bos = NewNode();
  bos->id = -1;
  bos->pos = 0;
  end_nodes_[0].push_back(bos);

  Node *eos = NewNode();
  eos->id = -1;
  eos->pos = len;
  begin_nodes_[len].push_back(eos);
}

// DOC:
// 将sentence[pos, pos + length - 1]子串 作为一个新的结点插入到lattice
//
// 参数:
//    pos -- 表示插入位置的参数
//    length -- 表示插入长度的参数
//
// 返回:
//    新插入结点的指针
//
// 注意:
// 在调用此方法之后 必须设置该结点的score与id参数
Lattice::Node *Lattice::Insert(int pos, int length) {
  Node *node = NewNode();
  node->pos = pos;
  node->length = length;
  const int utf8_length =
      static_cast<int>(surface(pos + length) - surface(pos));
  node->piece = absl::string_view(surface(pos), utf8_length);
  begin_nodes_[pos].push_back(node);
  end_nodes_[pos + node->length].push_back(node);

  return node;
}

// DOC:
// 预处理到达各位置的最佳路径及其概率对数值
//
// 注意:
// 所有结点都必须提前输入
//
// 返回:
// 到达各位置的最佳路径的遍历结点序
std::vector<Lattice::Node *> Lattice::Viterbi() {
  const int len = size();

  // DOC:
  // 自左向右递推各位置结点的backtrace_score 并记录前向结点
  // 方法:枚举比较法
  for (int pos = 0; pos <= len; ++pos) {
    for (Node *rnode : begin_nodes_[pos]) {
      rnode->prev = nullptr;
      float best_score = 0.0;
      Node *best_node = nullptr;
      for (Node *lnode : end_nodes_[pos]) {
        const float score = lnode->backtrace_score + rnode->score;
        if (best_node == nullptr || score > best_score) {
          best_node = lnode;
          best_score = score;
        }
      }
      if (best_node == nullptr) {
        LOG(ERROR) << "Failed to find the best path in Viterbi.";
        return {};
      }
      rnode->prev = best_node;
      rnode->backtrace_score = best_score;
    }
  }

  // DOC:
  // 回溯以确定最佳路径
  // backtrace
  std::vector<Node *> results;
  for (Node *node = begin_nodes_[len][0]->prev; node->prev != nullptr;
       node = node->prev) {
    results.push_back(node);
  }

  std::reverse(results.begin(), results.end());

  return results;
}

    // DOC:
    // 返回当前句子出现概率的对数值
    //
    // 参数:
    //    freq -- 表示句子出现的频率
    //    excepted -- 保存各单词出现概率对数值vector的指针 其下表为单词ID
    //
    // 返回:
    //    当前句子出现概率的对数值
float Lattice::PopulateMarginal(float freq,
                                std::vector<float> *expected) const {
  if (expected == nullptr) return 0.0;

  const int len = size();

  // DOC:
  // alpha 保存前向概率累计的对数值
  // beta 保存后向概率累计d对数值
  // 注意:alpha/beta的下标为Node::node_id
  // alpha and beta (accumulative log prob) in Forward Backward.
  // the index of alpha/beta is Node::node_id.
  std::vector<float> alpha(node_allocator_.size(), 0.0);
  std::vector<float> beta(node_allocator_.size(), 0.0);

  // DOC:
  //    初始化alpha
  //    自前向后遍历叠加概率
  for (int pos = 0; pos <= len; ++pos) {
    for (Node *rnode : begin_nodes_[pos]) {
      for (Node *lnode : end_nodes_[pos]) {
        alpha[rnode->node_id] = LogSumExp(alpha[rnode->node_id],
                                          lnode->score + alpha[lnode->node_id],
                                          lnode == end_nodes_[pos][0]);
      }
    }
  }

  // DOC:
  //    初始化beta
  //    自后向前遍历叠加概率
  for (int pos = len; pos >= 0; --pos) {
    for (Node *lnode : end_nodes_[pos]) {
      for (Node *rnode : begin_nodes_[pos]) {
        beta[lnode->node_id] =
            LogSumExp(beta[lnode->node_id], rnode->score + beta[rnode->node_id],
                      rnode == begin_nodes_[pos][0]);
      }
    }
  }

  // DOC:
  //    更新expected数组
  const float Z = alpha[begin_nodes_[len][0]->node_id];
  for (int pos = 0; pos < len; ++pos) {
    for (Node *node : begin_nodes_[pos]) {
      if (node->id >= 0) {
        // 注意:|expected|的下标为该单词在词库中的ID 而不是在lattice中的
        // the index of |expected| is a Node::id, which is a vocabulary id.
        (*expected)[node->id] += freq * exp(alpha[node->node_id] + node->score +
                                            beta[node->node_id] - Z);
      }
    }
  }

  return freq * Z;
}

    // DOC:
    // 返回n个结点条件下的最佳路径的遍历结点序
    //
    // 参数:
    //    size_t -- 预分配hypothesis_allocator的大小
    //    nbest_size -- 表示nbest_size个结点的条件下
    //
    // 返回:
    //    n个结点条件下的最佳路径的遍历结点序
std::vector<std::vector<Lattice::Node *>> Lattice::NBest(size_t nbest_size) {
  // DOC:
  // 针对不同的 nbest_size 分类
  if (nbest_size < 1) {
    LOG(WARNING) << "nbest_size >= 1. Returns empty result.";
    return {};
  }

  if (nbest_size == 1) {
    return {Viterbi()};
  }

  // DOC:
  //    采用枚举法解决 N-bests 问题
  //    数据储存:优先队列
  //    对于每一个lattice 自句末(终止符)枚举每一条路径
  //    对于每条到达结点x的路径 计算 f(x) = g(x) + h(x)
  //    其中 g(x): 表示自终止符到该路径最前段的score和
  //        h(x): 表示自起始符到路径x的最高score
  //        f(x): 表示该假设sentence自优先队列被取出的优先级 即该假设sentence的可能性
  // 注意:
  //    自前向后使用Viterbi算法恰好可能得到h(x)
  //    自优先队列中取得 N-bests 问题的最终答案
  // Uses A* search to enumerate N-bests.
  // Given a lattice, enumerates hypotheses (paths) from EOS.
  // At each partial path x, compute f(x) as follows
  //   f(x) = g(x) + h(x).
  // g(x): the sum of scores from  EOS to the left-most node in x.
  // h(x): a heuristic that estimates the largest score from x to BOS.
  // f(x): the priority to pop a new hypothesis from the priority queue.
  //
  // As left-to-right Viterbi search can tell the *exact* value of h(x),
  // we can obtain the exact n-best results with A*.

  // DOC:
  // 成员变量:
  //    node --
  //    next --
  //    fx -- 当前假设路径的f(x)值
  //    gx -- 当前假设路径的g(x)值
  struct Hypothesis {
    Node *node;
    Hypothesis *next;
    float fx;
    float gx;
  };

  // DOC:
  //    假设路径比较器
  //
  // 调用:
  //    优先队列Agenda 优先级操作符重载
  class HypothesisComparator {
   public:
    const bool operator()(Hypothesis *h1, Hypothesis *h2) {
      return (h1->fx < h2->fx);
    }
  };

  // DOC:
  //    初始化:创建优先队列 预设分配空间大小
  using Agenda = std::priority_queue<Hypothesis *, std::vector<Hypothesis *>,
                                     HypothesisComparator>;
  constexpr size_t kPreallocatedHypothesisSize = 512;
  model::FreeList<Hypothesis> hypothesis_allocator(kPreallocatedHypothesisSize);

  Agenda agenda;
  std::vector<std::vector<Node *>> results;

  // DOC:
  //    初始化:将终字符加入队列中 自后向前搜素
  auto *eos = hypothesis_allocator.Allocate();
  eos->node = eos_node();
  eos->next = nullptr;
  eos->fx = eos->node->score;
  eos->gx = eos->node->score;
  agenda.push(eos);

  // DOC:
  //    预处理:运行 Viterbi 算法得到前向score
  // Run Viterbi first to fill backtrace score.
  Viterbi();

  while (!agenda.empty()) {
    auto *top = agenda.top();
    agenda.pop();
    auto *node = top->node;

    // DOC:
    //      当到达起始符后:回溯 将单词存入result 退出循环
    // Reaches to BOS
    if (node == bos_node()) {
      results.resize(results.size() + 1);
      for (auto *n = top->next; n->next != nullptr; n = n->next) {
        results.back().push_back(n->node);
      }
      if (results.size() == nbest_size) {
        break;
      }
      continue;
    }

    // DOC:
    //      扩展在pos处结尾的结点
    // Expands new node ending at node->pos
    for (Node *lnode : end_nodes(node->pos)) {
      auto *hyp = hypothesis_allocator.Allocate();
      hyp->node = lnode;
      hyp->gx = lnode->score + top->gx;  // just adds node->score
      // 当前路径的g(x) = 当前结点的score + 之前路径的g(x)
      hyp->fx =
          lnode->backtrace_score + top->gx;  // backtrace_score is h(node).
      // 结点x的backtrace_score即为h(x)
      hyp->next = top;
      agenda.push(hyp);
    }

    // DOC:
    //      当输入字符串太长或包含重复部分时
    //      保留前kMinAgendaSize的假设路径 以缩小agenda 从而避免agenda空间过大占用过多内存的问题
    // When the input is too long or contains duplicated phrases,
    // `agenda` will get extremely big. Here we avoid this case by
    // dynamically shrinking the agenda.
    constexpr int kMaxAgendaSize = 100000;
    constexpr int kMinAgendaSize = 512;
    if (agenda.size() >= kMaxAgendaSize) {
      LOG(WARNING) << "Too big agenda. shrinking";
      // Keeps the top `kMinAgendaSize` hypothesis.
      Agenda new_agenda;
      const int size = std::min<int>(kMinAgendaSize, nbest_size * 10);
      for (int i = 0; i < size; ++i) {
        new_agenda.push(agenda.top());
        agenda.pop();
      }
      agenda = std::move(new_agenda);
    }
  }

  return results;
}

    // DOC:
    // 返回根据分词块的产生可能性 在lattice中选择的一条组成路径
    //
    // 参数:
    //    theta -- 平滑参数
    //
    // 返回:
    //    lattice中的一条组成路径
std::vector<Lattice::Node *> Lattice::Sample(float theta) {
  const int len = size();
  if (len == 0) return {};

  std::vector<float> alpha(node_allocator_.size(), 0.0);

  for (int pos = 0; pos <= len; ++pos) {
    for (Node *rnode : begin_nodes_[pos]) {
      for (Node *lnode : end_nodes_[pos]) {
        alpha[rnode->node_id] = LogSumExp(
            alpha[rnode->node_id], theta * lnode->score + alpha[lnode->node_id],
            lnode == end_nodes_[pos][0]);
      }
    }
  }

  auto *mt = random::GetRandomGenerator();

  std::vector<Node *> results;
  std::vector<float> probs;

  float Z = alpha[eos_node()->node_id];
  Node *node = eos_node();
  while (true) {
    probs.clear();
    for (const Node *lnode : end_nodes_[node->pos]) {
      probs.push_back(exp(alpha[lnode->node_id] + theta * lnode->score - Z));
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    node = end_nodes_[node->pos][dist(*mt)];
    if (node == bos_node()) break;

    Z = alpha[node->node_id];
    results.push_back(node);
  }

  std::reverse(results.begin(), results.end());
  return results;
}

// Model::Model() {}
// Model::~Model() {}

// DOC:
//      预处理lattice
// 参数:
//      lattice -- 待处理的Lattice对象的指针
void Model::PopulateNodes(Lattice *lattice) const {
  // DOC:
  //    返回自begin_pos位切片到结尾的字符长度
  // 参数:
  //    begin_pos -- 起始单词的序号
  //    end -- 终止处结点指针
  auto get_chars_length = [&lattice](int begin_pos, const char *end) {
    int pos = begin_pos;
    while (lattice->surface(pos) < end) ++pos;
    return pos - begin_pos;
  };

  // DOC:
  // 设置词典未知词的惩罚参数
  // 未知词的score = 最低值 - 惩罚参数
  constexpr float kUnkPenalty = 10.0;
  const float unk_score = min_score() - kUnkPenalty;

  // DOC:
  // len -- lattice的大小
  // end -- latice结尾字符指针
  const int len = lattice->size();
  const char *end = lattice->sentence() + lattice->utf8_size();

  // DOC:
  //    创建字典树存放最佳匹配
  //    size +1 以防越界
  // +1 just in case.
  std::vector<Darts::DoubleArray::result_pair_type> trie_results(
      trie_results_size_ + 1);

  for (int begin_pos = 0; begin_pos < len; ++begin_pos) {
    const char *begin = lattice->surface(begin_pos);

    // DOC:
    //      返回begin_pos号切片的前缀结点数
    // Finds all pieces which are prefix of surface(begin_pos).
    const size_t num_nodes = trie_->commonPrefixSearch(
        begin, trie_results.data(), trie_results.size(),
        static_cast<int>(end - begin));
    CHECK_LT(num_nodes, trie_results.size());

    bool has_single_node = false;

    // DOC:
    //      插入分词
    // Inserts pieces to the lattice.
    for (size_t k = 0; k < num_nodes; ++k) {
      const int length =
          get_chars_length(begin_pos, begin + trie_results[k].length);
      const int id = trie_results[k].value; // Trie的value储存单词id
      if (IsUnusedInlined(id)) continue;
      Lattice::Node *node = lattice->Insert(begin_pos, length);
      node->id = id;  // the value of Trie stores vocab_id.
      // User defined symbol receives extra bonus to always be selected.
      node->score = IsUserDefinedInlined(id) ? (length * max_score_ + 1.0)
                                             : GetScoreInlined(id);
      if (!has_single_node && node->length == 1) {
        has_single_node = true;
      }
    }

    // 处理未登录词
    if (!has_single_node) {
      Lattice::Node *node = lattice->Insert(begin_pos, 1);
      node->id = unk_id_;  // add UNK node.
      node->score = unk_score;
    }
  }
}

// DOC:
//      将分词转换为ID
// 参数:
//      piece -- 分词的字符串形式
// 返回:
//      所查询分词在字典中的ID
int Model::PieceToId(absl::string_view piece) const {
  auto it = reserved_id_map_.find(piece);
  if (it != reserved_id_map_.end()) {
    return it->second;
  }
  int id = 0;
  trie_->exactMatchSearch(piece.data(), id);
  return id == -1 ? unk_id_ : id;
}

// DOC:
//      建立字典树
void Model::BuildTrie(std::vector<std::pair<absl::string_view, int>> *pieces) {
  if (!status().ok()) return;

  if (pieces->empty()) {
    status_ = util::InternalError("no pieces are loaded.");
    return;
  }

  // DOC:
  //    在创建DoubleArray后对分词排序
  //    并且只接受排序后的字符串
  // sort by sentencepiece since DoubleArray::build()
  // only accepts sorted strings.
  sort(pieces->begin(), pieces->end());

  // DOC:
  //    在字典树中创建键值对
  // Makes key/value set for DoubleArrayTrie.
  std::vector<const char *> key(pieces->size());
  std::vector<int> value(pieces->size());
  for (size_t i = 0; i < pieces->size(); ++i) {
    key[i] = (*pieces)[i].first.data();  // sorted piece.
    value[i] = (*pieces)[i].second;      // vocab_id
  }

  trie_ = port::MakeUnique<Darts::DoubleArray>();
  if (trie_->build(key.size(), const_cast<char **>(&key[0]), nullptr,
                   &value[0]) != 0) {
    status_ = util::InternalError("cannot build double-array.");
    return;
  }

  // DOC:
  //    计算字典树中最大的共享前缀数
  // Computes the maximum number of shared prefixes in the trie.
  const int kMaxTrieResultsSize = 1024;
  std::vector<Darts::DoubleArray::result_pair_type> results(
      kMaxTrieResultsSize);
  trie_results_size_ = 0;
  for (const auto &p : *pieces) {
    const int num_nodes = trie_->commonPrefixSearch(
        p.first.data(), results.data(), results.size(), p.first.size());
    trie_results_size_ = std::max(trie_results_size_, num_nodes);
  }

  pieces_.clear();

  if (trie_results_size_ == 0)
    status_ = util::InternalError("no entry is found in the trie.");
}

// DOC:
//      根据原型创建模型
Model::Model(const ModelProto &model_proto) {
  model_proto_ = &model_proto;

  InitializePieces();

  min_score_ = FLT_MAX;
  max_score_ = FLT_MIN;
  for (const auto &sp : model_proto_->pieces()) {
    if (sp.type() == ModelProto::SentencePiece::NORMAL) {
      min_score_ = std::min(min_score_, sp.score());
      max_score_ = std::max(max_score_, sp.score());
    }
  }

  std::vector<std::pair<absl::string_view, int>> pieces;
  for (const auto &it : pieces_) pieces.emplace_back(it.first, it.second);

  BuildTrie(&pieces);
}

Model::~Model() {}

// DOC:
//      对字符串进行处理计算
// 参数:
//      normalized -- 已规范化的字符串
// 返回:
//      经过Viterbi算法处理过的最佳分词序列
EncodeResult Model::Encode(absl::string_view normalized) const {
  if (!status().ok() || normalized.empty()) {
    return {};
  }

  Lattice lattice;
  lattice.SetSentence(normalized);
  PopulateNodes(&lattice);

  EncodeResult results;
  for (const auto *node : lattice.Viterbi()) {
    results.emplace_back(node->piece, node->id);
  }

  return results;
}

// DOC:
//      处理计算字符串nbest_size条件下的最佳匹配
// 参数:
//      normalized -- 已规范化的字符串
//      nbest_size -- 分词的个数限制
// 返回:
//      经过NBest算法处理过的最佳分词序列
NBestEncodeResult Model::NBestEncode(absl::string_view normalized,
                                     int nbest_size) const {
  if (!status().ok() || normalized.empty()) {
    return {{{}, 0.0}};
  }

  nbest_size = std::max<int>(1, std::min<int>(nbest_size, 1024));

  Lattice lattice;
  lattice.SetSentence(normalized);
  PopulateNodes(&lattice);

  NBestEncodeResult nbest_results;
  for (const auto &nbest : lattice.NBest(nbest_size)) {
    EncodeResult results;
    float score = 0.0;
    for (const auto *node : nbest) {
      score += node->score;
      results.emplace_back(node->piece, node->id);
    }
    nbest_results.emplace_back(results, score);
  }

  return nbest_results;
}

// DOC:
//      根据平滑参数计算样本字符串的最佳匹配
// 参数:
//      normalized -- 已规范化的字符串
//      theta -- 平滑参数
// 返回:
//      样本字符串的最佳分词序列
EncodeResult Model::SampleEncode(absl::string_view normalized,
                                 float theta) const {
  if (!status().ok() || normalized.empty()) {
    return {};
  }

  Lattice lattice;
  lattice.SetSentence(normalized);
  PopulateNodes(&lattice);

  EncodeResult results;
  for (const auto *node : lattice.Sample(theta)) {
    results.emplace_back(node->piece, node->id);
  }

  return results;
}

}  // namespace unigram
}  // namespace sentencepiece
