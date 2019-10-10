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
#include "common.h"
#include "filesystem.h"
#include "flags.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_processor.h"
#include "util.h"

// DOC:
// 初始化flags，定义模型解码参数的数据类型，名称，默认值和帮助信息
DEFINE_string(model, "", "model file name");
DEFINE_string(output, "", "output filename");
DEFINE_string(input_format, "piece", "choose from piece or id");
DEFINE_string(output_format, "string", "choose from string or proto");
DEFINE_string(extra_options, "",
              "':' separated encoder extra options, e.g., \"reverse:bos:eos\"");

// DOC:
// decoder主函数，载入命令行参数，将encoder标记后得到的字词序列解码
// 
// Example:
//      $ echo "9 459 11 939 44 11 4 142 82 8 28 21 132 6" | spm_decode --model=m.model --input_format=id
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
  // 检测加载处理器和解码时的额外选择设置是否成功
  CHECK_OK(sp.Load(FLAGS_model));
  CHECK_OK(sp.SetDecodeExtraOptions(FLAGS_extra_options));

  // DOC:
  // 获取输出
  auto output = sentencepiece::filesystem::NewWritableFile(FLAGS_output);
  CHECK_OK(output->status());

  // DOC:
  // 如果额外的参数为空，输入""
  if (rest_args.empty()) {
    rest_args.push_back("");  // empty means that reading from stdin.
  }
  // DOC:
  // 创建解码过程所需变量
  std::string detok, line;
  sentencepiece::SentencePieceText spt;
  std::function<void(const std::vector<std::string> &pieces)> process;

  // DOC:
  // 获取需要解码的ID
  auto ToIds = [&](const std::vector<std::string> &pieces) {
    std::vector<int> ids;
    for (const auto &s : pieces) {
      ids.push_back(atoi(s.c_str()));
    }
    return ids;
  };
  // DOC:
  // 如果输出格式为piece
  if (FLAGS_input_format == "piece") {
    // DOC:
    // 如果输出格式为string，检测解码是否正常，并将输出传入WriteLine函数中 
    if (FLAGS_output_format == "string") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(pieces, &detok));
        output->WriteLine(detok);
      };
      // DOC:
      // 如果输出格式为proto，检测解码是否正常
    } else if (FLAGS_output_format == "proto") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(pieces, &spt));
        //        output->WriteLine(spt.Utf8DebugString());
      };
      // DOC:
      // 如果未产生上述结果，则输出错误日志：Unknown output format: FLAGS_output_format
    } else {
      LOG(FATAL) << "Unknown output format: " << FLAGS_output_format;
    }
    // DOC:
    // 如果输出格式为id
  } else if (FLAGS_input_format == "id") {
    // DOC:
    // 如果输出格式为string，检测解码是否正常，并将输出传入WriteLine函数中 
    if (FLAGS_output_format == "string") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(ToIds(pieces), &detok));
        output->WriteLine(detok);
      };
      // DOC:
      // 如果输出格式为proto，检测解码是否正常，并将输出传入WriteLine函数中 
    } else if (FLAGS_output_format == "proto") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(ToIds(pieces), &spt));
        //        output->WriteLine(spt.Utf8DebugString());
      };
      // DOC:
      // 如果未产生上述结果，则输出错误日志：Unknown output format: FLAGS_output_format
    } else {
      LOG(FATAL) << "Unknown output format: " << FLAGS_output_format;
    }
    // DOC:
    // 如果未产生上述结果，则输出错误日志：Unknown input format: FLAGS_input_format
  } else {
    LOG(FATAL) << "Unknown input format: " << FLAGS_input_format;
  }

  for (const auto &filename : rest_args) {
    // DOC:
    // 获取输入
    auto input = sentencepiece::filesystem::NewReadableFile(filename);
    // 确定输入状态正常
    CHECK_OK(input->status());
    while (input->ReadLine(&line)) {
      // 用空格隔开各个词
      const auto pieces = sentencepiece::string_util::Split(line, " ");
      // 开始处理
      process(pieces);
    }
  }

  return 0;
}
