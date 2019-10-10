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

#ifndef FLAGS_H_
#define FLAGS_H_

#include <memory>
#include <string>
#include <vector>

namespace sentencepiece {
namespace flags {

// DOC:
// 匿名枚举，用于表示 6 种数据类型，包括 int，bool，int64, uint64, double, std::string。
enum { I, B, I64, U64, D, S };

// DOC:
// 结构体 Flag，用于保存模型参数
struct Flag;

// DOC:
// 用于构造 Flag。
// Example:
//      std::string storage = "storage";
//      std::string default_storage = "default_storage";
//      std::string help = "help";
//      FlagRegister fl("input", &storage, &default_storage, sentencepiece::flags::S,
//          help);
class FlagRegister {
 public:
  FlagRegister(const char *name, void *storage, const void *default_storage,
               int shorttpe, const char *help);
  ~FlagRegister();

 private:
  std::unique_ptr<Flag> flag_;
};

// DOC:
// 返回一个字符串，描述了 sentencepiece 的命令行参数用法。
//
// 参数:
//      programname -- 程序名，用于表示程序的名称
// 返回:
//      一个描述了 sentencepiece 的命令行参数用法的字符串
std::string PrintHelp(const char *programname);

// DOC:
// 解析命令行参数并修改对应 Flag
//
// 参数:
//      argc -- 传入命令行参数个数
//      argv -- 命令行参数字符串数组
//      rest_args -- 额外的参数
void ParseCommandLineFlags(int argc, char **argv,
                           std::vector<std::string> *rest_args = nullptr);
}  // namespace flags
}  // namespace sentencepiece

// DOC:
// 定义Flag变量及对应命名空间，并使用该命名空间
//
// 参数:
//      type -- Flag 值的类型
//      shorttype -- Flag值的类型在 sentencepiece::flags 的匿名函数中对应的值
//      name -- Flag 的名称
//      value -- Flag 的值
//      help -- Flag 的说明
#define DEFINE_VARIABLE(type, shorttype, name, value, help)               \
  namespace sentencepiece_flags_fL##shorttype {                           \
    using namespace sentencepiece::flags;                                 \
    type FLAGS_##name = value;                                            \
    static const type FLAGS_DEFAULT_##name = value;                       \
    static const sentencepiece::flags::FlagRegister fL##name(             \
        #name, reinterpret_cast<void *>(&FLAGS_##name),                   \
        reinterpret_cast<const void *>(&FLAGS_DEFAULT_##name), shorttype, \
        help);                                                            \
  }                                                                       \
  using sentencepiece_flags_fL##shorttype::FLAGS_##name

// DOC: （未被使用！！！)
// 声明 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      type -- Flag 值的类型
//      shorttype -- Flag 值的类型在 sentencepiece::flags 的匿名函数中对应的值
//      name -- Flag 的名称
#define DECLARE_VARIABLE(type, shorttype, name) \
  namespace sentencepiece_flags_fL##shorttype { \
    extern type FLAGS_##name;                   \
  }                                             \
  using sentencepiece_flags_fL##shorttype::FLAGS_##name

// DOC:
// 定义 int32 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      type -- Flag 值的类型
//      shorttype -- Flag 值的类型在 sentencepiece::flags 的匿名函数中对应的值
//      name -- Flag 的名称
#define DEFINE_int32(name, value, help) \
  DEFINE_VARIABLE(int32, I, name, value, help)

// DOC: (未被使用！！！)
// 定义 int32 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      name -- Flag 的名称
#define DECLARE_int32(name) DECLARE_VARIABLE(int32, I, name)

// DOC: (未被使用！！！)
// 定义 int64 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      type -- Flag 值的类型
//      shorttype -- Flag 值的类型在 sentencepiece::flags 的匿名函数中对应的值
//      name -- Flag 的名称
#define DEFINE_int64(name, value, help) \
  DEFINE_VARIABLE(int64, I64, name, value, help)

// DOC: (未被使用！！！)
// 定义 int64 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      name -- Flag 的名称
#define DECLARE_int64(name) DECLARE_VARIABLE(int64, I64, name)

// DOC: (未被使用！！！)
// 定义 uint64 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      type -- Flag 值的类型
//      shorttype -- Flag 值的类型在 sentencepiece::flags 的匿名函数中对应的值
//      name -- Flag 的名称
#define DEFINE_uint64(name, value, help) \
  DEFINE_VARIABLE(uint64, U64, name, value, help)

// DOC: (未被使用！！！)
// 定义 uint64 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      name -- Flag 的名称
#define DECLARE_uint64(name) DECLARE_VARIABLE(uint64, U64, name)

// DOC:
// 定义 double 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      type -- Flag值的类型
//      shorttype -- Flag 值的类型在 sentencepiece::flags 的匿名函数中对应的值
//      name -- Flag 的名称
#define DEFINE_double(name, value, help) \
  DEFINE_VARIABLE(double, D, name, value, help)

// DOC: (未被使用！！！)
// 定义 double 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      name -- Flag 的名称
#define DECLARE_double(name) DECLARE_VARIABLE(double, D, name)

// DOC:
// 定义bool类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      type -- Flag值的类型
//      shorttype -- Flag值的类型在sentencepiece::flags的匿名函数中对应的值
//      name -- Flag 的名称
#define DEFINE_bool(name, value, help) \
  DEFINE_VARIABLE(bool, B, name, value, help)

// DOC: (未被使用！！！)
// 定义bool类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      name -- Flag 的名称
#define DECLARE_bool(name) DECLARE_VARIABLE(bool, B, name)

// DOC:
// 定义std::string类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      type -- Flag 值的类型
//      shorttype -- Flag 值的类型在 sentencepiece::flags 的匿名函数中对应的值
//      name -- Flag 的名称
#define DEFINE_string(name, value, help) \
  DEFINE_VARIABLE(std::string, S, name, value, help)

// DOC: (未被使用！！！)
// 定义 std::string 类型的 Flag 变量及对应命名空间，并使用该命名空间
//
// 参数:
//      name -- Flag 的名称
#define DECLARE_string(name) DECLARE_VARIABLE(std::string, S, name)

// DOC:
// 检测 flag 是否为空，为空则打印错误和帮助信息到控制台并退出
//
// 参数:
//      flag -- flag 变量
#define CHECK_OR_HELP(flag)                                        \
  if (FLAGS_##flag.empty()) {                                      \
    std::cout << "ERROR: --" << #flag << " must not be empty\n\n"; \
    std::cout << sentencepiece::flags::PrintHelp(PACKAGE_STRING);  \
    sentencepiece::error::Exit(0);                                 \
  }

#endif  // FLAGS_H_
