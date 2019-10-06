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

#include "builder.h"
#include "common.h"
#include "filesystem.h"
#include "flags.h"
#include "normalizer.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"

// DOC:
// 初始化flags，定义模型标准化参数的数据类型，名称，默认值和帮助信息
DEFINE_string(model, "", "Model file name");
DEFINE_bool(use_internal_normalization, false,
            "Use NormalizerSpec \"as-is\" to run the normalizer "
            "for SentencePiece segmentation");
DEFINE_string(normalization_rule_name, "",
              "Normalization rule name. "
              "Choose from nfkc or identity");
DEFINE_string(normalization_rule_tsv, "", "Normalization rule TSV file. ");
DEFINE_bool(remove_extra_whitespaces, true, "Remove extra whitespaces");
DEFINE_bool(decompile, false,
            "Decompile compiled charamap and output it as TSV.");
DEFINE_string(output, "", "Output filename");

// DOC:
// 定义命名空间
using sentencepiece::ModelProto;
using sentencepiece::NormalizerSpec;
using sentencepiece::SentencePieceProcessor;
using sentencepiece::SentencePieceTrainer;
using sentencepiece::normalizer::Builder;
using sentencepiece::normalizer::Normalizer;

// DOC:
// normalizer主函数，载入命令行参数，正规化语料库
int main(int argc, char *argv[]) {
  // DOC: 
  // 读取额外的参数 
  std::vector<std::string> rest_args;
  // 解析命令行参数并修改对应Flag
  sentencepiece::flags::ParseCommandLineFlags(argc, argv, &rest_args);
  // 定义正规化特性
  NormalizerSpec spec;

  // DOC:
  // 如果flags中模型不为空
  if (!FLAGS_model.empty()) {
    // DOC:
    // 定义模型原型
    ModelProto model_proto;
    // sentencepiece处理器
    SentencePieceProcessor sp;
    // 检测Flag参数model是否为空，
    // 为空则打印错误和帮助信息到控制台并退出    
    CHECK_OK(sp.Load(FLAGS_model));
    // 获取正规化特征
    spec = sp.model_proto().normalizer_spec();
    // 如果flags正规化规则的tsv文件不为空
  } else if (!FLAGS_normalization_rule_tsv.empty()) {
    // 设置正规化规则
    spec.set_normalization_rule_tsv(FLAGS_normalization_rule_tsv);
    // 确定正规化特征是正确的
    CHECK_OK(SentencePieceTrainer::PopulateNormalizerSpec(&spec));
    // 如果flags正规化规则的名称不为空
  } else if (!FLAGS_normalization_rule_name.empty()) {
    // 设置正规化名称
    spec.set_name(FLAGS_normalization_rule_name);
    // 确定正规化名称是正确的   
    CHECK_OK(SentencePieceTrainer::PopulateNormalizerSpec(&spec));
  } else {
    // 输出错误日志
    LOG(FATAL) << "Sets --model, normalization_rule_tsv, or "
                  "normalization_rule_name flag.";
  }
  // DOC:
  // 使用规范器特征参数在模型原型中标记
  // Uses the normalizer spec encoded in the model_pb.
  if (!FLAGS_use_internal_normalization) {
    spec.set_add_dummy_prefix(false);    // do not add dummy prefix.
    spec.set_escape_whitespaces(false);  // do not output meta symbol.
    spec.set_remove_extra_whitespaces(FLAGS_remove_extra_whitespaces);
  }
  // DOC:
  // 如果flags允许被反编译
  if (FLAGS_decompile) {
    Builder::CharsMap chars_map;
    // DOC:
    // 确定反编译字符图及保存过程正确
    CHECK_OK(
        Builder::DecompileCharsMap(spec.precompiled_charsmap(), &chars_map));
    CHECK_OK(Builder::SaveCharsMap(FLAGS_output, chars_map));
  } else {
    // DOC:
    // 定义规范器
    const Normalizer normalizer(spec);

    // 获取输出
    auto output = sentencepiece::filesystem::NewWritableFile(FLAGS_output);
    // 确定输出状态正确
    CHECK_OK(output->status());

    // DOC:
    // 如果额外参数为空（即从标准输入读取）
    if (rest_args.empty()) {
      // 返回额外参数""
      rest_args.push_back("");  // empty means that read from stdin.
    }
    // DOC:
    // 定义行变量
    std::string line;
    for (const auto &filename : rest_args) {
      // DOC:
      // 获取输入
      auto input = sentencepiece::filesystem::NewReadableFile(filename);
      // 确定输入状态正常
      CHECK_OK(input->status());
      while (input->ReadLine(&line)) {
        // 输入规范器
        output->WriteLine(normalizer.Normalize(line));
      }
    }
  }

  return 0;
}
