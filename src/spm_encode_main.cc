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

#include <functional>
#include <unordered_map>
#include "common.h"
#include "filesystem.h"
#include "flags.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_processor.h"
#include "trainer_interface.h"

// DOC:
// 初始化flags，定义模型编码参数的数据类型，名称，默认值和帮助信息
DEFINE_string(model, "", "model file name");
DEFINE_string(
    output_format, "piece",
    "choose from piece, id, proto, nbest_piece, nbest_id, or nbest_proto");
DEFINE_string(output, "", "output filename");
DEFINE_string(extra_options, "",
              "':' separated encoder extra options, e.g., \"reverse:bos:eos\"");
DEFINE_int32(nbest_size, 10, "NBest size");
DEFINE_double(alpha, 0.5, "Smoothing parameter for sampling mode.");

// Piece restriction with vocabulary file.
// https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt
DEFINE_string(vocabulary, "",
              "Restrict the vocabulary. The encoder only emits the "
              "tokens in \"vocabulary\" file");
DEFINE_int32(vocabulary_threshold, 0,
             "Words with frequency < threshold will be treated as OOV");

DEFINE_bool(generate_vocabulary, false,
            "Generates vocabulary file instead of segmentation");

// DOC:
// encoder主函数，载入命令行参数，将输入的文本正规化, 并使用Trainer训练过的子词模型将其标记为一个子词序列
// 
// Example:
//      $ echo "I saw a girl with a telescope." | spm_encode --model=m.model --output_format=id
int main(int argc, char *argv[]) {
  // DOC: 
  // 读取额外的参数
  std::vector<std::string> rest_args;
  // 解析命令行参数并修改对应Flag
  sentencepiece::flags::ParseCommandLineFlags(argc, argv, &rest_args);

  // 检测Flag参数model是否为空，
  // 为空则打印错误和帮助信息到控制台并退出
  CHECK_OR_HELP(model);
  // sentencepiece处理器
  sentencepiece::SentencePieceProcessor sp;
  // 检测加载处理器和标记时的额外选择设置是否成功
  CHECK_OK(sp.Load(FLAGS_model));
  CHECK_OK(sp.SetEncodeExtraOptions(FLAGS_extra_options));

  // DOC:
  // 如果flags中词语为空，则检测加载词汇是否成功
  if (!FLAGS_vocabulary.empty()) {
    CHECK_OK(sp.LoadVocabulary(FLAGS_vocabulary, FLAGS_vocabulary_threshold));
  }
  // DOC:
  // 获取输出
  auto output = sentencepiece::filesystem::NewWritableFile(FLAGS_output);
  // DOC:
  // 检测output状态是否正常
  CHECK_OK(output->status());

  // DOC:
  // 如果额外的参数为空，输入""
  if (rest_args.empty()) {
    rest_args.push_back("");  // empty means that reading from stdin.
  }
  // DOC:
  // 创建标记过程所需变量
  std::string line;
  std::vector<std::string> sps;
  std::vector<int> ids;
  std::vector<std::vector<std::string>> nbest_sps;
  std::vector<std::vector<int>> nbest_ids;
  std::unordered_map<std::string, int> vocab;
  sentencepiece::SentencePieceText spt;
  sentencepiece::NBestSentencePieceText nbest_spt;
  std::function<void(const std::string &line)> process;
  // DOC:
  // 如果能够生成词汇文件但尚未分割
  if (FLAGS_generate_vocabulary) {
    process = [&](const std::string &line) {
      // DOC:
      //  检测标记是否正常
      CHECK_OK(sp.Encode(line, &spt));
      for (const auto &piece : spt.pieces()) {
        // DOC:
        // 如果词汇文件在词汇表中并且是可控的，则词汇数量加一
        if (!sp.IsUnknown(piece.id()) && !sp.IsControl(piece.id()))
          vocab[piece.piece()]++;
      }
    };
    // DOC:
    // 如果输出格式为piece，检测标记是否正常，并将输出传入WriteLine函数中 
  } else if (FLAGS_output_format == "piece") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.Encode(line, &sps));
      output->WriteLine(sentencepiece::string_util::Join(sps, " "));
    };
    // DOC:
    // 如果输出格式为id，检测标记是否正常，并将输出传入WriteLine函数中 
  } else if (FLAGS_output_format == "id") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.Encode(line, &ids));
      output->WriteLine(sentencepiece::string_util::Join(ids, " "));
    };
    // DOC:
    // 如果输出格式为proto，检测标记是否正常，并将输出传入WriteLine函数中 
  } else if (FLAGS_output_format == "proto") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.Encode(line, &spt));
      //      output->WriteLine(spt.Utf8DebugString());
    };
    // DOC:
    // 如果输出格式为sample_piece，检测样本标记是否正常，并将输出传入WriteLine函数中 
  } else if (FLAGS_output_format == "sample_piece") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.SampleEncode(line, FLAGS_nbest_size, FLAGS_alpha, &sps));
      output->WriteLine(sentencepiece::string_util::Join(sps, " "));
    };
    // DOC:
    // 如果输出格式为sample_id，检测样本标记是否正常，并将输出传入WriteLine函数中 
  } else if (FLAGS_output_format == "sample_id") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.SampleEncode(line, FLAGS_nbest_size, FLAGS_alpha, &ids));
      output->WriteLine(sentencepiece::string_util::Join(ids, " "));
    };
    // DOC:
    // 如果输出格式为sample_proto，检测样本标记是否正常，并将输出传入WriteLine函数中 
  } else if (FLAGS_output_format == "sample_proto") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.SampleEncode(line, FLAGS_nbest_size, FLAGS_alpha, &spt));
      //      output->WriteLine(spt.Utf8DebugString());
    };
    // DOC:
    // 如果输出格式为nbest_piece，检测标记是否正常，并将输出传入WriteLine函数中 
  } else if (FLAGS_output_format == "nbest_piece") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.NBestEncode(line, FLAGS_nbest_size, &nbest_sps));
      for (const auto &result : nbest_sps) {
        output->WriteLine(sentencepiece::string_util::Join(result, " "));
      }
    };
    // DOC:
    // 如果输出格式为nbest_id，检测标记是否正常，并将输出传入WriteLine函数中     
  } else if (FLAGS_output_format == "nbest_id") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.NBestEncode(line, FLAGS_nbest_size, &nbest_ids));
      for (const auto &result : nbest_ids) {
        output->WriteLine(sentencepiece::string_util::Join(result, " "));
      }
    };
    // DOC:
    // 如果输出格式为nbest_proto，检测标记是否正常，并将输出传入WriteLine函数中  
  } else if (FLAGS_output_format == "nbest_proto") {
    process = [&](const std::string &line) {
      CHECK_OK(sp.NBestEncode(line, FLAGS_nbest_size, &nbest_spt));
      //      output->WriteLine(nbest_spt.Utf8DebugString());
    };
    // DOC:
    // 如果未产生上述结果，则输出错误日志：Unknown output format: FLAGS_output_format
  } else {
    LOG(FATAL) << "Unknown output format: " << FLAGS_output_format;
  }
  
  for (const auto &filename : rest_args) {
    // DOC:
    // 获取输入
    auto input = sentencepiece::filesystem::NewReadableFile(filename);
    // 确定输入状态正常
    CHECK_OK(input->status());
    while (input->ReadLine(&line)) {
      // 开始处理
      process(line);
    }
  }
  // DOC:
  // 输出标记后的结果
  if (FLAGS_generate_vocabulary) {
    for (const auto &it : sentencepiece::Sorted(vocab)) {
      output->WriteLine(it.first + "\t" +
                        sentencepiece::string_util::SimpleItoa(it.second));
    }
  }

  return 0;
}
