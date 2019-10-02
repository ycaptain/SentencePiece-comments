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

#ifndef UNIGRAM_MODEL_H_
#define UNIGRAM_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "freelist.h"
#include "model_interface.h"
#include "sentencepiece_model.pb.h"
#include "third_party/darts_clone/darts.h"

// DOC:命名空间 sentencepiece::unigram

namespace sentencepiece {
namespace unigram {
// DOC:
// Lattice类
// 保存unigram处理后的所有分词片段
//
// Lattice represents a search space of sentence piece segmentation.
class Lattice {
// DOC:
// 公共成员变量
 public:

// DOC:
// Lattice对象的构造和删除
  Lattice();
  virtual ~Lattice();

// DOC:
// 用于保存单个piece
// 成员变量:
//      piece -- 当前结点单词保存为string_view格式
//      pos -- 当前结点保存单词的在句子中的位置
//      length -- 当前结点保存以Unicode编码单词的长度 注意:不是utf8
//      node_id -- 当前结点在当前lattice中的位置
//      id -- 当前结点保存单词的ID 注意:以-1表示unkrown单词
//      score -- 当前结点保存单词的词频对数 即该词出现概率的对数值
//      backtrace_score -- 在Viterbi算法中 自当前结点向前回溯的最大概率叠加值
//      prev -- 在Viterbi算法中 当前结点的最佳路径中 该结点的上层结点指针
//      DebugString -- 未使用
//
  struct Node {
    absl::string_view piece;  // Sentence piece representation.
    uint32 pos;               // Unicode position in the sentence.
    uint32 length;            // Unicode length, not UT8 byte.
    uint32 node_id;           // unique id in the current lattice.
    int id;                   // vocab id. (maybe -1 for UNK)
    float score;              // logprob of this sentencepiece.
    float backtrace_score;    // backtrace info used in Viterbi.
    Node *prev;               // best previous node on Viterbi path.

    std::string DebugString() const;
  };

  // DOC:
  // 返回起始符指针
  // Returns bos node.
  Node *bos_node() const;

  // DOC:
  // 返回终止符指针
  // Returns eos node.
  Node *eos_node() const;

  // DOC:
  // 返回begin_nodes_中在pos处开始的结点组
  //
  // 参数:
  //    pos -- 表示开始位置的参数
  // Returns nodes starting at |pos|.
  const std::vector<Node *> &begin_nodes(int pos) const;

  // DOC:
  // 返回在pos处结束的结点组
  //
  // 参数:
  //    pos -- 表示结束位置的参数
  // Returns nodes ending at |pos|.
  const std::vector<Node *> &end_nodes(int pos) const;

  // DOC:
  // 返回Unicode编码下的长度
  // Returns Unicode character length.
  int size() const;

  // DOC:
  // 返回多字节字符编码下(utf8)的长度
  // Returns multi-byte (utf8) length.
  int utf8_size() const;

  // DOC:
  // 返回当前句子从pos位至结尾的子串
  // 相当于句子从pos为至结尾的切片
  // Returns the substring of sentence. sentence[pos:]
  const char *surface(int pos) const;

  // DOC:
  // 返回整个当前句子的头指针
  // 相当于surface(0)
  // Returns immutable sentence. The same as surface(0)
  const char *sentence() const;

  // DOC:
  // 清除lattice
  // Clears the lattice.
  void Clear();

  // DOC:
  // 根据一个string_view类型的sentence 创建一个Lattice对象
  // Sets new sentence.
  void SetSentence(absl::string_view sentence);

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
  // Inserts a new node at [pos, pos + length - 1].
  // After calling this method, The caller must set Node::score and Node::id.
  Node *Insert(int pos, int length);

  // DOC:
  // 预处理到达各位置的最佳路径及其概率对数值
  //
  // 注意:
  // 所有结点都必须提前输入
  //
  // 返回:
  // 到达各位置的最佳路径的遍历结点序
  // Returns Viterbi path. All nodes must be populated in advance.
  std::vector<Node *> Viterbi();

  // DOC:
  // 返回n个结点条件下的最佳路径的遍历结点序
  //
  // 参数:
  //    size_t -- 预分配hypothesis_allocator的大小
  //    nbest_size -- 表示nbest_size个结点的条件下
  //
  // 返回:
  //    n个结点条件下的最佳路径的遍历结点序
  // Returns n-best results.
  std::vector<std::vector<Node *>> NBest(size_t nbest_size);

  // DOC:
  // 返回根据分词块的产生可能性 在lattice中选择的一条组成路径
  //
  // 参数:
  //    theta -- 平滑参数
  //
  // 返回:
  //    lattice中的一条组成路径
  //
  // Samples one path from the lattice according to the
  // generation probability (Product of piece probabilities).
  // `theta` is a smoothing parameter.
  std::vector<Node *> Sample(float theta);

  // DOC:
  // 返回当前句子出现概率的对数值
  //
  // 参数:
  //    freq -- 表示句子出现的频率
  //    excepted -- 保存各单词出现概率对数值vector的指针 其下表为单词ID
  //
  // 返回:
  //    当前句子出现概率的对数值
  // Populates marginal probability of every node in this lattice.
  // |freq| is the frequency of the sentence.
  //  for (auto *node : all_nodes_) {
  //    (*expected)[node->id] += marginal_prob_of_node * freq;
  //  }
  // Returns the log-likelihood of this sentence.
  float PopulateMarginal(float freq, std::vector<float> *expected) const;

  // DOC:
  // 私有成员变量
 private:
  // DOC:
  // 返回新结点的指针
  // Returns new node.
  // Lattice class has the ownership of the returned value.
  Node *NewNode();

  // DOC:
  // 各私有成员变量：
  // sentence_ -- 保存sentence
  // surface_ -- 保存sentenced的全部切片
  // begin_nodes_ -- 保存各位置开始的结点组
  // end_nodes_ -- 保存各位置结束的结点组
  // node_allocator_ -- 结点空间分配器 保存lattice中的全部结点
  absl::string_view sentence_;
  std::vector<const char *> surface_;
  std::vector<std::vector<Node *>> begin_nodes_;
  std::vector<std::vector<Node *>> end_nodes_;
  model::FreeList<Node> node_allocator_;
};

// DOC:
// Model类
// 实现ModelInterface接口
//
class Model : public ModelInterface {
 public:
  explicit Model(const ModelProto &model_proto);
  Model() {}
  ~Model() override;

    // DOC:
    //      对字符串进行处理计算
    // 参数:
    //      normalized -- 已规范化的字符串
    // 返回:
    //      经过Viterbi算法处理过的最佳分词序列
  EncodeResult Encode(absl::string_view normalized) const override;

    // DOC:
    //      处理计算字符串nbest_size条件下的最佳匹配
    // 参数:
    //      normalized -- 已规范化的字符串
    //      nbest_size -- 分词的个数限制
    // 返回:
    //      经过NBest算法处理过的最佳分词序列
  NBestEncodeResult NBestEncode(absl::string_view normalized,
                                int nbest_size) const override;

    // DOC:
    //      根据平滑参数计算样本字符串的最佳匹配
    // 参数:
    //      normalized -- 已规范化的字符串
    //      theta -- 平滑参数
    // 返回:
    //      样本字符串的最佳分词序列
  EncodeResult SampleEncode(absl::string_view normalized,
                            float theta) const override;

  // DOC：
  //    返回句子所有分词中的最小概率值
  //    设定未登录词的score为min_score() - 10
  // Returns the minimum score in sentence pieces.
  // min_score() - 10 is used for the cost of unknown sentence.
  float min_score() const { return min_score_; }

  // DOC:
  //    返回句子所有分词中的最大概率值
  //    用户自定义词的score为max_score()
  // Returns the maximum score in sentence pieces.
  // max_score() is used for the cost of user defined symbols.
  float max_score() const { return max_score_; }

  // DOC:
  //      预处理lattice
  // 参数:
  //      lattice -- 待处理的Lattice对象的指针
  // 注意:
  //      在调用该函数后 Viterbi()返回最佳分词片段
  // 调用关系:
  //      调用Viterbi()函数
  // Populates all sentence pieces to the |lattice|.
  // After calling this function, lattice.Viterbi() returns the
  // best segmentation.
  void PopulateNodes(Lattice *lattice) const;

    // DOC:
    //      将分词转换为ID
    // 参数:
    //      piece -- 分词的字符串形式
    // 返回:
    //      所查询分词在字典中的ID
  // Returns a vocab id of |piece|.
  int PieceToId(absl::string_view piece) const override;

 protected:
  // DOC:
  //    创建字典树
  // Builds a Trie index.
  void BuildTrie(std::vector<std::pair<absl::string_view, int>> *pieces);

  float min_score_ = 0.0;
  float max_score_ = 0.0;
  std::unique_ptr<Darts::DoubleArray> trie_;

  // DOC:
  //    字典树results返回的最大值
  //    取决于字典树中最大的共享前缀数
  // Maximum size of the return value of Trie, which corresponds
  // to the maximum size of shared common prefix in the sentence pieces.
  int trie_results_size_;
};

}  // namespace unigram
}  // namespace sentencepiece
#endif  // UNIGRAM_MODEL_H_
