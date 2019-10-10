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

#include "sentencepiece_processor.h"

#include <map>
#include <set>
#include <utility>

#include "common.h"
#include "filesystem.h"
#include "model_factory.h"
#include "model_interface.h"
#include "normalizer.h"
#include "sentencepiece.pb.h"
#include "unigram_model.h"
#include "util.h"

namespace sentencepiece {
namespace {

// Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK).
const char kSpaceSymbol[] = "\xe2\x96\x81";

// Encodes <unk> into U+2047 (DOUBLE QUESTION MARK),
// since this character can be useful both for user and
// developer. We can easily figure out that <unk> is emitted.
const char kDefaultUnknownSymbol[] = " \xE2\x81\x87 ";
}  // namespace

SentencePieceProcessor::SentencePieceProcessor() {}
SentencePieceProcessor::~SentencePieceProcessor() {}

// DOC:
// 加载文件函数，根据文件路径读取加载文件
//
// 参数：
//      filename -- min_string_view 类型的文件
//
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::Load(util::min_string_view filename) {
  auto input = filesystem::NewReadableFile(string_util::ToSV(filename), true);
  RETURN_IF_ERROR(input->status());
  std::string proto;
  CHECK_OR_RETURN(input->ReadAll(&proto));
  return LoadFromSerializedProto(proto);
}

// DOC:
// 对加载文件是否成功调用宏函数进行检测，若加载文件失败则进入Die环节
// 参数：
//      filename -- min_string_view 类型的文件
void SentencePieceProcessor::LoadOrDie(util::min_string_view filename) {
  CHECK_OK(Load(filename));
}

// DOC:
// 加载文件函数，根据文件路径读取加载文件
//
// 参数：
//      is -- istream 类型的文件指针
//
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::Load(std::istream *is) {
  return util::StatusBuilder(util::error::UNIMPLEMENTED)
         << "std::stream API is deprecated. Use LoadFromSerializedProto() "
         << "to load model from any serialized blob object.";
}

// DOC:
// 加载文件函数，根据文件路径读取加载文件
//
// 参数：
//      model_proto -- ModelProto 类型的文件引用
//
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::Load(const ModelProto &model_proto) {
  auto model_proto_copy = port::MakeUnique<ModelProto>();
  *model_proto_copy = model_proto;
  return Load(std::move(model_proto_copy));
}

// DOC:
// 加载文件函数，根据文件路径读取加载文件
// 参数：
//      serialized -- min_string_view 类型的序列化文件
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::LoadFromSerializedProto(
    util::min_string_view serialized) {
  auto model_proto = port::MakeUnique<ModelProto>();
  CHECK_OR_RETURN(
      model_proto->ParseFromArray(serialized.data(), serialized.size()));
  return Load(std::move(model_proto));
}

// DOC:
// 加载文件函数，根据文件路径读取加载文件
//
// 参数：
//      model_proto -- unique_ptr 指针引用文件
//
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::Load(
    std::unique_ptr<ModelProto> &&model_proto) {
  model_proto_ = std::move(model_proto);
  model_ = ModelFactory::Create(*model_proto_);
  normalizer_ = port::MakeUnique<normalizer::Normalizer>(
      model_proto_->normalizer_spec(), model_proto_->trainer_spec());

  // Escapes user-defined-symbols in normalizer.
  // DOC:
  // 调用 normalizer 模块的前缀匹配设置
  normalizer_->SetPrefixMatcher(model_->prefix_matcher());

  RETURN_IF_ERROR(status());

  // Running self-testing.
  std::vector<std::string> errors, sps;
  // DOC:
  // 测试模块，在 for 循环体内进行测试，如输出符合预期无失误则继续
  for (const auto &s : model_proto_->self_test_data().samples()) {
    RETURN_IF_ERROR(Encode(s.input(), &sps));
    const std::string result = string_util::Join(sps, " ");
    if (s.expected() != result) {
      errors.emplace_back(
          string_util::StrCat(s.input(), "\t", s.expected(), "\t", result));
    }
  }

  if (!errors.empty()) {
    LOG(INFO) << errors.size() << "/"
              << model_proto_->self_test_data().samples_size()
              << " samples did not pass the test.";
    for (const auto &e : errors) {
      LOG(INFO) << e;
    }
    return util::InternalError("Self-test failures. See LOG(INFO).");
  }

  return util::OkStatus();
}

// DOC:
// 加载编码额外选项
//
// 参数：
//      extra_option -- min_string_view 类型的设置
//
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::SetEncodeExtraOptions(
    util::min_string_view extra_options) {
  return ParseExtraOptions(extra_options, &encode_extra_options_);
}

// DOC:
// 加载解码额外选项
//
// 参数：
//      extra_option -- min_string_view 类型的设置
//
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::SetDecodeExtraOptions(
    util::min_string_view extra_options) {
  return ParseExtraOptions(extra_options, &decode_extra_options_);
}

// DOC:
// 对model和规范化器的配置进行检测
util::Status SentencePieceProcessor::status() const {
  CHECK_OR_RETURN(model_) << "Model is not initialized.";
  CHECK_OR_RETURN(normalizer_) << "Normalizer is not initialized.";
  RETURN_IF_ERROR(model_->status());
  RETURN_IF_ERROR(normalizer_->status());
  return util::OkStatus();
}

// DOC:
// 设置语料库内容
//
// 参数：
//      valid_vocab -- string 向量类型的语料
//
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::SetVocabulary(
    const std::vector<std::string> &valid_vocab) {
  RETURN_IF_ERROR(status());

  // TODO(taku): supports vocabulary constraint in BPE model.
  const auto type = model_proto_->trainer_spec().model_type();
  CHECK_OR_RETURN(type == TrainerSpec::UNIGRAM || type == TrainerSpec::BPE)
      << "Vocabulary constraint is only enabled in subword units.";

  const std::set<std::string> vocab(valid_vocab.begin(), valid_vocab.end());

  for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    auto *piece = model_proto_->mutable_pieces(i);
    if (piece->type() == ModelProto::SentencePiece::CONTROL ||
        piece->type() == ModelProto::SentencePiece::UNKNOWN ||
        piece->type() == ModelProto::SentencePiece::USER_DEFINED) {
      continue;
    }
    if (vocab.find(piece->piece()) != vocab.end() ||
        string_util::OneCharLen(piece->piece().c_str()) ==
            piece->piece().size()) {
      piece->set_type(ModelProto::SentencePiece::NORMAL);
    } else {
      piece->set_type(ModelProto::SentencePiece::UNUSED);
    }
  }

  return util::OkStatus();
}

// DOC:
// 重置语料库为初始值
// 返回：
//      util::Status 类型的状态信息
util::Status SentencePieceProcessor::ResetVocabulary() {
  RETURN_IF_ERROR(status());
  for (auto &piece : *(model_proto_->mutable_pieces())) {
    if (piece.type() == ModelProto::SentencePiece::UNUSED)
      piece.set_type(ModelProto::SentencePiece::NORMAL);
  }

  return util::OkStatus();
}

// DOC:
// 通过文件名加载语料库
//
// 参数：
//              filename -- min_string_view 类型的语料库文件名
//             threshold -- 整数类型的阈值设置
//
// 返回：
//          util::Status 类型的状态信息
util::Status SentencePieceProcessor::LoadVocabulary(
    util::min_string_view filename, int threshold) {
  auto input = filesystem::NewReadableFile(string_util::ToSV(filename));
  RETURN_IF_ERROR(input->status());

  std::string line;
  std::vector<std::string> vocab;

  while (input->ReadLine(&line)) {
    const std::vector<std::string> v = string_util::Split(line, "\t");
    CHECK_GE_OR_RETURN(v.size(), 1);
    CHECK_OR_RETURN(!v[0].empty());
    int32 freq = 1;
    if (v.size() >= 2) freq = atoi(v[1].c_str());
    if (freq >= threshold) vocab.emplace_back(v[0]);
  }

  return SetVocabulary(vocab);
}

#define CHECK_OR_RETURN_STATUS_STL(container)               \
  RETURN_IF_ERROR(status());                                \
  CHECK_OR_RETURN(container) << "output container is null"; \
  container->clear();

#define CHECK_OR_RETURN_STATUS_PROTO(proto)         \
  RETURN_IF_ERROR(status());                        \
  CHECK_OR_RETURN(proto) << "output proto is null"; \
  proto->Clear();

//////////////////////////////////////////////////////////////
// Simple API.
// DOC:
// 对内容进行编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             pieces -- 字符串类型的向量指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::Encode(
    util::min_string_view input, std::vector<std::string> *pieces) const {
  CHECK_OR_RETURN_STATUS_STL(pieces);

  SentencePieceText spt;
  RETURN_IF_ERROR(Encode(input, &spt));
  for (const auto &sp : spt.pieces()) {
    pieces->emplace_back(sp.piece());
  }

  return util::OkStatus();
}

// DOC:
// 对内容进行编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             ids -- 整型类型的向量指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::Encode(util::min_string_view input,
                                            std::vector<int> *ids) const {
  CHECK_OR_RETURN_STATUS_STL(ids);

  SentencePieceText spt;
  RETURN_IF_ERROR(Encode(input, &spt));
  for (const auto &sp : spt.pieces()) {
    ids->emplace_back(sp.id());
  }

  return util::OkStatus();
}

// DOC:
// 对内容进行解码的函数
//
// 参数：
//              detokenized -- 字符串类型的逆词法分析指针
//             pieces -- 字符串类型的向量引用
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::Decode(
    const std::vector<std::string> &pieces, std::string *detokenized) const {
  CHECK_OR_RETURN_STATUS_STL(detokenized);

  SentencePieceText spt;
  RETURN_IF_ERROR(Decode(pieces, &spt));
  *detokenized = std::move(spt.text());

  return util::OkStatus();
}

// DOC:
// 对内容进行解码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             ids -- 整型类型的向量引用
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                            std::string *detokenized) const {
  CHECK_OR_RETURN_STATUS_STL(detokenized);

  SentencePieceText spt;
  RETURN_IF_ERROR(Decode(ids, &spt));
  *detokenized = std::move(spt.text());

  return util::OkStatus();
}

// DOC:
// 对内容进行N-Best的N个最好结果编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             nbest_size -- 整型类型的结果数目
//             pieces -- 字符串类型的向量指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::NBestEncode(
    util::min_string_view input, int nbest_size,
    std::vector<std::vector<std::string>> *pieces) const {
  CHECK_OR_RETURN_STATUS_STL(pieces);

  NBestSentencePieceText spt;
  RETURN_IF_ERROR(NBestEncode(input, nbest_size, &spt));
  for (const auto &nbest : spt.nbests()) {
    std::vector<std::string> result;
    for (const auto &sp : nbest.pieces()) {
      result.emplace_back(sp.piece());
    }
    pieces->emplace_back(result);
  }

  return util::OkStatus();
}

// DOC:
// 对内容进行N-Best的N个最好结果编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             nbest_size -- 整型类型的结果数目
//             ids -- 整型类型的向量指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::NBestEncode(
    util::min_string_view input, int nbest_size,
    std::vector<std::vector<int>> *ids) const {
  CHECK_OR_RETURN_STATUS_STL(ids);

  NBestSentencePieceText spt;
  RETURN_IF_ERROR(NBestEncode(input, nbest_size, &spt));
  for (const auto &nbest : spt.nbests()) {
    std::vector<int> result;
    for (const auto &sp : nbest.pieces()) {
      result.emplace_back(sp.id());
    }
    ids->emplace_back(result);
  }

  return util::OkStatus();
}

// DOC:
// 对内容进行样本编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             nbest_size -- 整型类型的结果数目
//             pieces -- 字符串类型的向量指针
//             alpha -- 浮点类型的配置数值
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::SampleEncode(
    util::min_string_view input, int nbest_size, float alpha,
    std::vector<std::string> *pieces) const {
  CHECK_OR_RETURN_STATUS_STL(pieces);

  SentencePieceText spt;
  RETURN_IF_ERROR(SampleEncode(input, nbest_size, alpha, &spt));
  for (const auto &sp : spt.pieces()) {
    pieces->emplace_back(sp.piece());
  }

  return util::OkStatus();
}

// DOC:
// 对内容进行样本编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             nbest_size -- 整型类型的结果数目
//             ids -- 整型类型的向量指针
//             alpha -- 浮点类型的配置数值
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::SampleEncode(util::min_string_view input,
                                                  int nbest_size, float alpha,
                                                  std::vector<int> *ids) const {
  CHECK_OR_RETURN_STATUS_STL(ids);

  SentencePieceText spt;
  RETURN_IF_ERROR(SampleEncode(input, nbest_size, alpha, &spt));
  for (const auto &sp : spt.pieces()) {
    ids->emplace_back(sp.id());
  }

  return util::OkStatus();
}

// 对SentencePiece文本进行储存
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             normalized -- min_string_view 类型的规范化字符串
//             norm_to_orig -- size_t 类型的转换字节数占用
//             spt -- SentencePieceText 类型的文本指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::PopulateSentencePieceText(
    util::min_string_view input, util::min_string_view normalized,
    const std::vector<size_t> &norm_to_orig, const EncodeResult &result,
    SentencePieceText *spt) const {
  size_t consumed = 0;
  bool is_prev_unk = false;
  for (const auto &p : result) {
    const absl::string_view w = p.first;  // piece
    const int id = p.second;              // id

    CHECK_OR_RETURN(!w.empty()) << "Empty piece is not allowed.";

    const bool is_unk = IsUnknown(id);

    if (IsControl(id)) {
      // Control symbol has no corresponding source surface, so begin == end.
      auto *sp = spt->add_pieces();
      sp->set_piece(w.data(), w.size());
      sp->set_id(id);
      sp->set_begin(norm_to_orig[consumed]);
      sp->set_end(norm_to_orig[consumed]);
    } else {
      const size_t begin = consumed;
      const size_t end = consumed + w.size();
      CHECK_LT_OR_RETURN(begin, norm_to_orig.size());
      CHECK_LT_OR_RETURN(end, norm_to_orig.size());
      const size_t orig_begin = norm_to_orig[begin];
      const size_t orig_end = norm_to_orig[end];
      CHECK_LE_OR_RETURN(orig_begin, input.size());
      CHECK_LE_OR_RETURN(orig_end, input.size());
      CHECK_LE_OR_RETURN(orig_begin, orig_end);
      const auto surface =
          absl::ClippedSubstr(input.data(), orig_begin, orig_end - orig_begin);
      // Merges continuous run of unknown pieces so that decoder
      // can copy or generate unknown tokens easily.
      // Note that merged tokens are still unknown,
      // since known pieces never consist of unknown characters.
      if (is_prev_unk && is_unk) {
        auto *sp = spt->mutable_pieces(spt->pieces_size() - 1);
        sp->set_piece(sp->piece() + std::string(w));
        sp->set_surface(sp->surface() + std::string(surface));
        sp->set_end(orig_end);
      } else {
        auto *sp = spt->add_pieces();
        sp->set_piece(w.data(), w.size());
        sp->set_id(id);
        sp->set_surface(surface.data(), surface.size());
        sp->set_begin(orig_begin);
        sp->set_end(orig_end);
      }
      consumed += w.size();
    }
    is_prev_unk = is_unk;
  }

  CHECK_EQ_OR_RETURN(consumed, normalized.size())
      << "all normalized characters are not consumed.";

  RETURN_IF_ERROR(ApplyExtraOptions(encode_extra_options_, spt));

  spt->set_text(input.data(), input.size());

  return util::OkStatus();
}  // namespace sentencepiece

// DOC:
// 对内容进行编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             spt -- SentencePieceText 类型的文本指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::Encode(util::min_string_view input,
                                            SentencePieceText *spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(spt);

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  RETURN_IF_ERROR(normalizer_->Normalize(string_util::ToSV(input), &normalized,
                                         &norm_to_orig));

  const auto result = model_->Encode(normalized);
  RETURN_IF_ERROR(
      PopulateSentencePieceText(input, normalized, norm_to_orig, result, spt));

  return util::OkStatus();
}

// DOC:
// 对内容进行N-Best编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             nbest_size -- 整型类型的最好结果数目
//             nbest_spt -- SentencePieceText 类型的文本指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::NBestEncode(
    util::min_string_view input, int nbest_size,
    NBestSentencePieceText *nbest_spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(nbest_spt);

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  RETURN_IF_ERROR(normalizer_->Normalize(string_util::ToSV(input), &normalized,
                                         &norm_to_orig));

  const auto nbests = model_->NBestEncode(normalized, nbest_size);
  CHECK_OR_RETURN(!nbests.empty()) << "NBestEncode returns empty result.";

  for (const auto &result : nbests) {
    auto *spt = nbest_spt->add_nbests();
    spt->set_score(result.second);
    RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig,
                                              result.first, spt));
  }

  return util::OkStatus();
}

// DOC:
// 对内容进行样本t编码的函数
//
// 参数：
//              input -- min_string_view 类型的原字符串内容
//             nbest_size -- 整型类型的最好结果数目
//             alpha -- 浮点类型的设置数值
//             spt -- SentencePieceText 类型的文本指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::SampleEncode(
    util::min_string_view input, int nbest_size, float alpha,
    SentencePieceText *spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(spt);

  CHECK_LE_OR_RETURN(nbest_size, 512) << "nbest_size must be nbest_size <= 512";

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  RETURN_IF_ERROR(normalizer_->Normalize(string_util::ToSV(input), &normalized,
                                         &norm_to_orig));

  if (nbest_size == 1 || nbest_size == 0) {
    const auto result = model_->Encode(normalized);
    RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig,
                                              result, spt));
  } else if (nbest_size > 1) {
    const auto nbests = model_->NBestEncode(normalized, nbest_size);
    CHECK_OR_RETURN(!nbests.empty()) << "NBestEncode returns empty result.";

    std::vector<float> probs(nbests.size(), 0.0);
    for (size_t i = 0; i < nbests.size(); ++i) {
      probs[i] = std::exp(alpha * nbests[i].second);
    }

    auto *mt = random::GetRandomGenerator();
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig,
                                              nbests[dist(*mt)].first, spt));

  } else if (nbest_size < 0) {
    const auto result = model_->SampleEncode(normalized, alpha);
    RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig,
                                              result, spt));
  }

  return util::OkStatus();
}

// DOC:
// 对内容进行解码的函数
//
// 参数：
//              pieces -- 字符串类型的向量引用
//             spt -- SentencePieceText 类型的文本指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::Decode(
    const std::vector<std::string> &pieces, SentencePieceText *spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(spt);

  const char *unk_surface = kDefaultUnknownSymbol;
  if (model_proto_ && model_proto_->trainer_spec().has_unk_surface())
    unk_surface = model_proto_->trainer_spec().unk_surface().c_str();

  auto DecodeSentencePiece = [&](absl::string_view piece, int id,
                                 bool is_bos_ws) -> std::string {
    if (IsControl(id)) {  // <s>, </s>
      return "";          // invisible symbol.
    } else if (IsUnknown(id)) {
      if (IdToPiece(id) == piece) {  // <unk>
        return unk_surface;
      } else {  // return piece when piece is not <unk>.
        return std::string(piece);
      }
    }

    if (is_bos_ws) {
      // Consume if the current position is bos and
      // piece starts with kSpaceSymbol.
      string_util::ConsumePrefix(&piece, kSpaceSymbol);
    }

    return string_util::StringReplace(piece, kSpaceSymbol, " ", true);
  };

  for (const std::string &w : pieces) {
    auto *sp = spt->add_pieces();
    sp->set_piece(w);
    sp->set_id(PieceToId(w));
  }

  RETURN_IF_ERROR(ApplyExtraOptions(decode_extra_options_, spt));

  std::string *text = spt->mutable_text();
  for (auto &sp : *(spt->mutable_pieces())) {
    sp.set_surface(DecodeSentencePiece(sp.piece(), sp.id(), text->empty()));
    sp.set_begin(text->size());
    sp.set_end(text->size() + sp.surface().size());
    *text += sp.surface();
  }

  return util::OkStatus();
}

// DOC:
// 对内容进行解码的函数
//
// 参数：
//              ids -- 整型类型的向量引用
//             spt -- SentencePieceText 类型的文本指针
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                            SentencePieceText *spt) const {
  std::vector<std::string> pieces;
  for (const int id : ids) {
    pieces.emplace_back(IdToPiece(id));
  }
  return Decode(pieces, spt);
}

// DOC:
// 以序列化原型形式对内容进行编码的函数
//
// 参数：
//          input -- min_string_view 类型的原文件内容
//
// 返回：
//          util::bytes 类型的输出字符串信息
util::bytes SentencePieceProcessor::EncodeAsSerializedProto(
    util::min_string_view input) const {
  SentencePieceText spt;
  if (!Encode(input, &spt).ok()) return "";
  return spt.SerializeAsString();
}

// DOC:
// 以序列化原型形式对内容进行样本编码的函数
//
// 参数：
//              input -- min_string_view 类型的原文件内容
//           alpha -- 浮点类型的配置数值
//           nbest_size -- 整型最好结果数目
//
// 返回：
//          util::bytes 类型的输出字符串信息
util::bytes SentencePieceProcessor::SampleEncodeAsSerializedProto(
    util::min_string_view input, int nbest_size, float alpha) const {
  SentencePieceText spt;
  if (!SampleEncode(input, nbest_size, alpha, &spt).ok()) return "";
  return spt.SerializeAsString();
}

// DOC:
// 以序列化原型形式对内容进行N-best编码的函数
//
// 参数：
//          input -- min_string_view 类型的原文件内容
//           nbest_size -- 整型最好结果数目
//
// 返回：
//          util::bytes 类型的输出字符串信息
util::bytes SentencePieceProcessor::NBestEncodeAsSerializedProto(
    util::min_string_view input, int nbest_size) const {
  NBestSentencePieceText spt;
  if (!NBestEncode(input, nbest_size, &spt).ok()) return "";
  return spt.SerializeAsString();
}

// DOC:
// 以序列化原型形式对内容进行解码的函数
//
// 参数：
//      pieces -- 字符串向量类型的字符串内容
//
// 返回：
//      util::bytes 类型的输出字符串信息
util::bytes SentencePieceProcessor::DecodePiecesAsSerializedProto(
    const std::vector<std::string> &pieces) const {
  SentencePieceText spt;
  if (!Decode(pieces, &spt).ok()) return "";
  return spt.SerializeAsString();
}

// DOC:
// 以序列化原型形式对内容进行解码的函数
//
// 参数：
//      ids -- 整型向量类型的引用
//
// 返回：
//      util::bytes 类型的输出字符串信息
util::bytes SentencePieceProcessor::DecodeIdsAsSerializedProto(
    const std::vector<int> &ids) const {
  SentencePieceText spt;
  if (!Decode(ids, &spt).ok()) return "";
  return spt.SerializeAsString();
}

// DOC:
// 使用宏函数检验状态是否返回默认值
#define CHECK_STATUS_OR_RETURN_DEFAULT(value)                            \
  if (!status().ok()) {                                                  \
    LOG(ERROR) << status().error_message() << "\nReturns default value " \
               << value;                                                 \
    return value;                                                        \
  }

// DOC:
// 获取 Piece 的大小数据
//
// 返回：
//      int 类型的 Piece 大小
int SentencePieceProcessor::GetPieceSize() const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->GetPieceSize();
}

// DOC:
// 获取指定 piece 的 PieceTold
//
// 参数：
//      piece -- min_string_view 类型的原文件内容
//
// 返回：
//      int 类型的 Piece 数据
int SentencePieceProcessor::PieceToId(util::min_string_view piece) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->PieceToId(string_util::ToSV(piece));
}

// DOC:
// 获取指定 id 的对应 piece 字符串内容
//
// 参数：
//      id -- 整型类型的待获取内容编码
//
// 返回：
//      字符串类型的 id 对应引用
const std::string &SentencePieceProcessor::IdToPiece(int id) const {
  static const std::string *kEmptyString = new std::string;
  CHECK_STATUS_OR_RETURN_DEFAULT(*kEmptyString);
  return model_->IdToPiece(id);
}

// DOC:
// 获取指定 id 的对应分数
//
// 参数：
//      id -- 整型类型的待获取内容编码
//
// 返回：
//      浮点类型的 id 对应分数
float SentencePieceProcessor::GetScore(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0.0);
  return model_->GetScore(id);
}

// DOC:
// 获取指定 id 是否为控制符
//
// 参数：
//      id -- 整型类型的待获取内容编码
//
// 返回：
//      布尔类型的控制符判断
bool SentencePieceProcessor::IsControl(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->IsControl(id);
}

// DOC:
// 获取指定 id 是否为未知
//
// 参数：
//      id -- 整型类型的待获取内容编码
//
// 返回：
//      布尔类型的是否未知判断
bool SentencePieceProcessor::IsUnknown(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->IsUnknown(id);
}

// DOC:
// 获取指定 id 是否未被使用
//
// 参数：
//      id -- 整型类型的待获取内容编码
//
// 返回：
//      布尔类型的id是否未被使用
bool SentencePieceProcessor::IsUnused(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(false);
  return model_->IsUnused(id);
}

int SentencePieceProcessor::unk_id() const {
  const int id = PieceToId(util::min_string_view(model_->unk_piece().data()));
  if (IsUnknown(id)) return id;
  return -1;
}

int SentencePieceProcessor::bos_id() const {
  const int id = PieceToId(util::min_string_view(model_->bos_piece().data()));
  if (IsControl(id)) return id;
  return -1;
}

int SentencePieceProcessor::eos_id() const {
  const int id = PieceToId(util::min_string_view(model_->eos_piece().data()));
  if (IsControl(id)) return id;
  return -1;
}

int SentencePieceProcessor::pad_id() const {
  const int id = PieceToId(util::min_string_view(model_->pad_piece().data()));
  if (IsControl(id)) return id;
  return -1;
}

// static
// DOC:
// 对额外设置进行应用的函数
//
// 参数：
//             extra_options -- 额外设置类型的向量信息
//             spt -- SentencePieceText 类型的文本指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::ApplyExtraOptions(
    const std::vector<ExtraOption> &extra_options,
    SentencePieceText *spt) const {
  for (const auto &extra_option : extra_options) {
    switch (extra_option) {
      case REVERSE:
        std::reverse(spt->mutable_pieces()->begin(),
                     spt->mutable_pieces()->end());
        break;
      case EOS: {
        auto *piece = spt->add_pieces();
        piece->set_id(
            PieceToId(util::min_string_view(model_->eos_piece().data())));
        piece->set_piece(model_->eos_piece().data(),
                         model_->eos_piece().size());
      } break;
      case BOS: {
        auto *array = spt->mutable_pieces();
        array->Add();
        for (int i = array->size() - 1; i > 0; --i) {
          array->SwapElements(i - 1, i);
        }
        auto *piece = array->Mutable(0);
        piece->set_id(
            PieceToId(util::min_string_view(model_->bos_piece().data())));
        piece->set_piece(model_->bos_piece().data(),
                         model_->bos_piece().size());
      } break;
      default:
        return util::InternalError("unknown extra_option type.");
    }
  }

  return util::OkStatus();
}

// static
// DOC:
// 对额外设置信息进行解包的函数
//
// 参数：
//              _extra_option -- min_string_view 类型的额外设置信息
//             extra_options -- 解包后待存放入的 ExtraOption 类型向量指针
//
// 返回：
//              util::Status 类型的状态信息
util::Status SentencePieceProcessor::ParseExtraOptions(
    util::min_string_view _extra_option,
    std::vector<SentencePieceProcessor::ExtraOption> *extra_options) const {
  absl::string_view extra_option(_extra_option.data(), _extra_option.size());

  extra_options->clear();
  if (extra_option.empty()) return util::OkStatus();

  RETURN_IF_ERROR(status());

  static std::map<absl::string_view, SentencePieceProcessor::ExtraOption>
      extra_option_map = {{"bos", SentencePieceProcessor::BOS},
                          {"eos", SentencePieceProcessor::EOS},
                          {"reverse", SentencePieceProcessor::REVERSE}};
  for (const auto &s : string_util::SplitPiece(extra_option, ":")) {
    const auto it = extra_option_map.find(s);
    CHECK_OR_RETURN(it != extra_option_map.end())
        << "option \"" << s << "\" is not available.";
    extra_options->push_back(it->second);

    if (it->second == SentencePieceProcessor::BOS) {
      CHECK_OR_RETURN(!IsUnknown(
          PieceToId(util::min_string_view(model_->bos_piece().data()))))
          << "id for `" << model_->bos_piece() << "` is not defined.";
    }
    if (it->second == SentencePieceProcessor::EOS) {
      CHECK_OR_RETURN(!IsUnknown(
          PieceToId(util::min_string_view(model_->eos_piece().data()))))
          << "id for `" << model_->eos_piece() << "` is not defined.";
    }
  }
  return util::OkStatus();
}

// DOC:
// 设置模型的函数
//
// 参数：
//      model -- 实现了Model接口的多态类型指针集合
void SentencePieceProcessor::SetModel(std::unique_ptr<ModelInterface> &&model) {
  model_ = std::move(model);
}

// DOC:
// 设置规范化器
//
// 参数：
//      normalizer -- Normalizer类型指针集合
void SentencePieceProcessor::SetNormalizer(
    std::unique_ptr<normalizer::Normalizer> &&normalizer) {
  normalizer_ = std::move(normalizer);
}

// DOC:
// 获取模型原型
//
// 返回：
//      模型原型类型的模型原型数据
const ModelProto &SentencePieceProcessor::model_proto() const {
  return *model_proto_;
}
}  // namespace sentencepiece
