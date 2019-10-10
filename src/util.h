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

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "common.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/strings/string_view.h"

#ifdef SPM_NO_THREADLOCAL
#include <pthread.h>
#endif

// DOC: 命名空间 sentencepiece
namespace sentencepiece {

// DOC:
// 重写 << 操作符实现空格分隔的 vector 输出
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  for (const auto n : v) {
    out << " " << n;
  }
  return out;
}

// DOC: 命名空间 sentencepiece::string_util
// String utilities
namespace string_util {

// DOC:
// 将 min_string_view 类型字符串转换为 string_view 类型
// 参数:
//      data -- 待转换的 min_string_view 字符串
// 返回:
//      转换后的 absl::string_view 类型字符串
inline absl::string_view ToSV(util::min_string_view data) {
  return absl::string_view(data.data(), data.size());
}

// DOC:
// 用于计算 string_view 的哈希值
// 采用 DJB hash 算法实现
struct string_view_hash {
  // DJB hash function.
  inline size_t operator()(const absl::string_view &sp) const {
    size_t hash = 5381;
    for (size_t i = 0; i < sp.size(); ++i) {
      hash = ((hash << 5) + hash) + sp[i];
    }
    return hash;
  }
};

// DOC:
// 借助 std::transform 方法将 string_view 转换为小写
// 参数:
//      arg -- 待转换的 absl:string_view 字符串
// 返回:
//      转换后的 std::string 字符串
inline std::string ToLower(absl::string_view arg) {
  std::string lower_value = std::string(arg);
  std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                 ::tolower);
  return lower_value;
}

// DOC:
// 借助 std::transform 方法将 string_view 转换为大写
// 参数:
//      arg -- 待转换的 absl:string_view 字符串
// 返回:
//      转换后的 std::string 字符串
inline std::string ToUpper(absl::string_view arg) {
  std::string upper_value = std::string(arg);
  std::transform(upper_value.begin(), upper_value.end(), upper_value.begin(),
                 ::toupper);
  return upper_value;
}

// DOC:
// 将 absl::string_view 词汇通过 std::stringstream 转换为目标类型
// 参数:
//      arg -- 待转换的 absl::string_view 字符串
//      result -- 存储转换结果的目标类型变量的指针
// 返回:
//      一个 bool 类型值，表示转换是否成功
template <typename Target>
inline bool lexical_cast(absl::string_view arg, Target *result) {
  std::stringstream ss;
  return (ss << arg.data() && ss >> *result);
}

// DOC:
// 将 absl::string_view 表示的词汇转换为 bool 布尔型
// 比对给定文本是否属于常见的 5 组表示布尔值的文本，若属于则转换成功
// 即 0/1、t/f、true/false、y/n 和 yes/no
// 参数:
//      arg -- 待转换的 absl::string_view 字符串
//      result -- 存储转换结果的 bool 类型变量的指针
// 返回:
//      一个 bool 类型值，表示转换是否成功
template <>
inline bool lexical_cast(absl::string_view arg, bool *result) {
  const char *kTrue[] = {"1", "t", "true", "y", "yes"};
  const char *kFalse[] = {"0", "f", "false", "n", "no"};
  std::string lower_value = std::string(arg);
  std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                 ::tolower);
  for (size_t i = 0; i < 5; ++i) {
    if (lower_value == kTrue[i]) {
      *result = true;
      return true;
    } else if (lower_value == kFalse[i]) {
      *result = false;
      return true;
    }
  }

  return false;
}

// DOC:
// 将 absl::string_view 表示的词汇转换为 std::string 普通文本类型
// 参数:
//      arg -- 待转换的 absl::string_view 字符串
//      result -- 存储转换结果的 std::string 类型变量的指针
// 返回:
//      一个 bool 类型值，表示转换是否成功
template <>
inline bool lexical_cast(absl::string_view arg, std::string *result) {
  *result = std::string(arg);
  return true;
}

// DOC:
// 针对 std::string 类型的文本分割函数
// 标准库字符串文本分割函数接口，调用 SplitInternal 实现
// 
// 参数:
//      str -- 原始文本
//      delim -- 分隔符
//      allow_empty -- 切分后元素是否可空
// 返回:
//      切分后的 vector 数组
std::vector<std::string> Split(const std::string &str, const std::string &delim,
                               bool allow_empty = false);

// DOC:
// 针对 absl::string_view 类型的文本分割函数
// 语料文本分割函数接口，调用 SplitInternal 实现
// 
// 参数:
//      str -- 原始文本
//      delim -- 分隔符
//      allow_empty -- 切分后元素是否可空
// 返回:
//      切分后的 vector 数组
std::vector<absl::string_view> SplitPiece(absl::string_view str,
                                          absl::string_view delim,
                                          bool allow_empty = false);

// DOC:
// 针对 std::string 实现的文本合并函数
// 将传入的子文本数组以 delim 作为分隔符合并为一个字符串
// 
// 参数:
//      tokens -- 子文本数组
//      delim -- 分隔符
// 返回:
//      合并后的 std::string 类型字符串
std::string Join(const std::vector<std::string> &tokens,
                 absl::string_view delim);

// DOC:
// 针对 int 实现的文本合并函数
// 将传入的 int 整形数组以 delim 作为分隔符合并为一个字符串
// 
// 参数:
//      tokens -- 整形数组
//      delim -- 分隔符
// 返回:
//      合并后的 std::string 类型字符串
std::string Join(const std::vector<int> &tokens, absl::string_view delim);

// DOC:
// 将 absl::string_view 类型文本转换为 std::string 类型
// 参数:
//      str -- 待转换的 absl::string_view 字符串
// 返回:
//      转换后的 std::string 类型文本
inline std::string StrCat(absl::string_view str) {
  return std::string(str.data(), str.size());
}

// DOC:
// 将 absl::string_view 类型的多个文本拼接
// 参数:
//      first -- 待转换的首个 absl::string_view 字符串
//      ... rest -- 待转换的其余字符串
// 返回:
//      转换后的 std::string 类型文本
template <typename... T>
inline std::string StrCat(absl::string_view first, const T &... rest) {
  return std::string(first) + StrCat(rest...);
}

std::string StringReplace(absl::string_view s, absl::string_view oldsub,
                          absl::string_view newsub, bool replace_all);

void StringReplace(absl::string_view s, absl::string_view oldsub,
                   absl::string_view newsub, bool replace_all,
                   std::string *res);

// DOC:
// 将 absl::string_view 类型的文本转换为 POD (平凡数据类型)
// 参数:
//      str -- 待转换的 absl::string_view 字符串
//      result -- 存储转换结果的特定类型变量的指针
// 返回:
//      一个 bool 类型值，表示转换是否成功
template <typename T>
inline bool DecodePOD(absl::string_view str, T *result) {
  CHECK_NOTNULL(result);
  if (sizeof(*result) != str.size()) {
    return false;
  }
  memcpy(result, str.data(), sizeof(T));
  return true;
}

// DOC:
// 将传入的 POD (平凡数据类型) 值转换为 std::string 字符串
// 参数:
//      value -- 待转换的 POD (平凡数据类型) 值
// 返回:
//      转换后的 std::string 类型字符串
template <typename T>
inline std::string EncodePOD(const T &value) {
  std::string s;
  s.resize(sizeof(T));
  memcpy(const_cast<char *>(s.data()), &value, sizeof(T));
  return s;
}

// DOC:
// 检测传入的 absl::string_view 字符串是否以 prefix 作为前缀
// 参数:
//      text -- 待检测的 absl::string_view 字符串
//      prefix -- 待检测的前缀字符串
// 返回:
//      一个 bool 类型值，表示 text 是否以 prefix 作为前缀
inline bool StartsWith(absl::string_view text, absl::string_view prefix) {
  return prefix.empty() ||
         (text.size() >= prefix.size() &&
          memcmp(text.data(), prefix.data(), prefix.size()) == 0);
}

// DOC:
// 检测传入的 absl::string_view 字符串是否以 suffix 作为后缀
// 参数:
//      text -- 待检测的 absl::string_view 字符串
//      suffix -- 待检测的后缀字符串
// 返回:
//      一个 bool 类型值，表示 text 是否以 suffix 作为后缀
inline bool EndsWith(absl::string_view text, absl::string_view suffix) {
  return suffix.empty() || (text.size() >= suffix.size() &&
                            memcmp(text.data() + (text.size() - suffix.size()),
                                   suffix.data(), suffix.size()) == 0);
}

// DOC:
// 若传入的 absl::string_view 字符串包含 expected 前缀，则去除
// 参数:
//      str -- 待处理的 absl::string_view 字符串
//      expected -- 预期的前缀字符串
// 返回:
//      一个 bool 类型值，表示 str 是否被处理
inline bool ConsumePrefix(absl::string_view *str, absl::string_view expected) {
  if (!StartsWith(*str, expected)) return false;
  str->remove_prefix(expected.size());
  return true;
}

// DOC:
// 将传入的数值 value 转换为十六进制文本
// 参数:
//      value -- 待转换的数值
// 返回:
//      转换后的 std::string 类型文本
template <typename T>
inline std::string IntToHex(T value) {
  std::ostringstream os;
  os << std::hex << std::uppercase << value;
  return os.str();
}

// DOC:
// 将传入的十六进制文本 value 转换为数值
// 参数:
//      value -- 待转换的十六进制文本
// 返回:
//      转换后的 T (模板) 类型数值数据
template <typename T>
inline T HexToInt(absl::string_view value) {
  T n;
  std::istringstream is(value.data());
  is >> std::hex >> n;
  return n;
}

// DOC:
// 将传入的数值 val 转换为字符串
// 参数:
//      val -- 待转换的数值数据
//      s -- 存储转换结果的字符串指针
// 返回:
//      一个 ULL 类型数据，表示转换后字符串的长度
template <typename T>
inline size_t Itoa(T val, char *s) {
  char *org = s;

  if (val < 0) {
    *s++ = '-';
    val = -val;
  }
  char *t = s;

  T mod = 0;
  while (val) {
    mod = val % 10;
    *t++ = static_cast<char>(mod) + '0';
    val /= 10;
  }

  if (s == t) {
    *t++ = '0';
  }

  *t = '\0';
  std::reverse(s, t);
  return static_cast<size_t>(t - org);
}

// DOC:
// 将传入的数值 val 转换为 std::string 类型字符串
// 参数:
//      val -- 待转换的数值数据
// 返回:
//      转换后的 std::string 类型字符串
template <typename T>
std::string SimpleItoa(T val) {
  char buf[32];
  Itoa<T>(val, buf);
  return std::string(buf);
}

// DOC:
// 返回一个 UTF-8 字符的长度
// 参数:
//      src -- 待查询的 ASCII 字符
// 返回:
//      ASCII 字符转换为 UTF-8 字符后的长度
// Return length of a single UTF-8 source character
inline size_t OneCharLen(const char *src) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

// DOC:
// 判断传入的字符 x 是否为双字节字符的尾字节 (优化方法)
// 参数:
//      x -- 待检测的字符
// 返回:
//      一个 bool 类型值，表示字符 x 是否为双字节字符的尾字节
// Return (x & 0xC0) == 0x80;
// Since trail bytes are always in [0x80, 0xBF], we can optimize:
inline bool IsTrailByte(char x) { return static_cast<signed char>(x) < -0x40; }

// DOC:
// 判断传入的 char32 型字符 c 是否是有效的 codepoint
// 注: codepoint 常用于表示 unicode 值 >0x10000 的部分，如 emoji 等
// 参数:
//      c -- 待检测的 char32 型字符
// 返回:
//      一个 bool 类型值，表示字符 x 是否为有效的 codepoint
inline bool IsValidCodepoint(char32 c) {
  return (static_cast<uint32>(c) < 0xD800) || (c >= 0xE000 && c <= 0x10FFFF);
}

// DOC:
// 逐个字符地检测给定文本是否符合 UTF-8 规范
// 
// 参数:
//      str -- 待检测的文本
// 返回:
//      一个 bool 类型值，表示给定的文本是否符合 UTF-8 规范
bool IsStructurallyValid(absl::string_view str);


// DOC:
// 定义 UnicodeText 类型
using UnicodeText = std::vector<char32>;

// DOC:
// UTF-8 字符解码函数
// 
// 参数:
//      begin -- UTF-8 字符起始位置指针
//      end -- UTF-8 字符终止位置指针
//      mblen -- 保存解码后所使用的字节数变量的指针
// 返回:
//      一个 char32 字符，即 UTF-8 解码的结果
char32 DecodeUTF8(const char *begin, const char *end, size_t *mblen);

// DOC:
// 针对 absl::string_view 实现的 UTF-8 字符解码函数
// 
// 参数:
//      input -- 待解码的字符
//      mblen -- 保存解码后所使用的字节数变量的指针
// 返回:
//      一个 char32 字符，即 UTF-8 解码的结果
inline char32 DecodeUTF8(absl::string_view input, size_t *mblen) {
  return DecodeUTF8(input.data(), input.data() + input.size(), mblen);
}

// DOC:
// 判断是否为有效的 UTF-8 字符
// 
// 参数:
//      input -- 待解码的字符
//      mblen -- 保存解码后所使用的字节数变量的指针
// 返回:
//      一个 bool 类型值，表示传入的 input 是否为有效的 UTF-8 字符
inline bool IsValidDecodeUTF8(absl::string_view input, size_t *mblen) {
  const char32 c = DecodeUTF8(input, mblen);
  return c != kUnicodeError || *mblen == 3;
}

// DOC:
// 将字符编码为 UTF-8
// 
// 参数:
//      c -- 待编码字符
//      output -- 保存编码结果的字符串指针
// 返回:
//      一个 ULL 类型值，表示编码 UTF-8 字符后所占的字节数
size_t EncodeUTF8(char32 c, char *output);

// DOC:
// 将 Unicode 字符编码为 UTF-8 (调用 UnicodeTextToUTF8 实现)
// 
// 参数:
//      c -- 待编码 Unicode 字符
// 返回:
//      一个 std::string 类型，包含编码后的 UTF-8 数据
std::string UnicodeCharToUTF8(const char32 c);

// DOC:
// 将 UTF-8 文本编码为 Unicode 文本
// 
// 参数:
//      utf8 -- 待编码的 UTF-8 文本
// 返回:
//      一个 UnicodeText 类型，包含编码后的 Unicode 数据
UnicodeText UTF8ToUnicodeText(absl::string_view utf8);

// DOC:
// 将 Unicode 文本编码为 UTF-8 文本
// 
// 参数:
//      utext -- 待编码的 Unicode 文本
// 返回:
//      一个 std::string 类型，包含编码后的 UTF-8 数据
std::string UnicodeTextToUTF8(const UnicodeText &utext);

}  // namespace string_util

// DOC: 命名空间 sentencepiece::port
// 其它关于 map 和指针相关的通用操作
// other map/ptr utilties
namespace port {

// DOC:
// 判断一个集合类型中是否含有某一个键值的模板函数
// 
// 参数:
//      collection -- 待检测的集合类型
//      key -- 待检测的键值
// 返回:
//      一个 bool 类型值，表示传入的集合类型中是否包含给定键值
template <class Collection, class Key>
bool ContainsKey(const Collection &collection, const Key &key) {
  return collection.find(key) != collection.end();
}

// DOC:
// 获取一个集合类型中给定键值对应的数值，如果未找到则输出异常并结束运行
// 
// 参数:
//      collection -- 待检测的集合类型
//      key -- 待检测的键值
// 返回:
//      给定集合中键值对应的数值
template <class Collection>
const typename Collection::value_type::second_type &FindOrDie(
    const Collection &collection,
    const typename Collection::value_type::first_type &key) {
  typename Collection::const_iterator it = collection.find(key);
  CHECK(it != collection.end()) << "Map key not found: " << key;
  return it->second;
}

// DOC:
// 获取一个集合类型中给定键值对应的数值，如果未找到则返回一个默认值
// 
// 参数:
//      collection -- 待检测的集合类型
//      key -- 待检测的键值
//      value - 未找到时返回的默认值
// 返回:
//      给定集合中键值对应的数值或默认值
template <class Collection>
const typename Collection::value_type::second_type &FindWithDefault(
    const Collection &collection,
    const typename Collection::value_type::first_type &key,
    const typename Collection::value_type::second_type &value) {
  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    return value;
  }
  return it->second;
}

// DOC:
// 将一组未出现过的数据插入到给定的集合类型中
// 
// 参数:
//      collection -- 待插入的集合类型
//      vt -- 待插入的数据 (集合类型)
// 返回:
//      一个 bool 类型值，表示是否插入成功
template <class Collection>
bool InsertIfNotPresent(Collection *const collection,
                        const typename Collection::value_type &vt) {
  return collection->insert(vt).second;
}

// DOC:
// 将一组未出现过的数据插入到给定的集合类型中
// 
// 参数:
//      collection -- 待插入的集合类型
//      key -- 待插入数据的键值
//      value -- 待插入数据的数值
// 返回:
//      一个 bool 类型值，表示是否插入成功
template <class Collection>
bool InsertIfNotPresent(
    Collection *const collection,
    const typename Collection::value_type::first_type &key,
    const typename Collection::value_type::second_type &value) {
  return InsertIfNotPresent(collection,
                            typename Collection::value_type(key, value));
}

// DOC:
// 将一组未出现过的数据插入到给定的集合类型中，若插入失败则输出异常并结束运行
// 
// 参数:
//      collection -- 待插入的集合类型
//      key -- 待插入数据的键值
//      data -- 待插入数据的数值
template <class Collection>
void InsertOrDie(Collection *const collection,
                 const typename Collection::value_type::first_type &key,
                 const typename Collection::value_type::second_type &data) {
  CHECK(InsertIfNotPresent(collection, key, data)) << "duplicate key";
}

// DOC:
// 三整数混合 Hash 函数 (采用移位实现)
// 
// 参数:
//      a -- 待计算的第一个整数
//      b -- 待计算的第二个整数
//      c -- 待计算的第三个整数
// hash
inline void mix(uint64 &a, uint64 &b, uint64 &c) {  // 64bit version
  a -= b;
  a -= c;
  a ^= (c >> 43);
  b -= c;
  b -= a;
  b ^= (a << 9);
  c -= a;
  c -= b;
  c ^= (b >> 8);
  a -= b;
  a -= c;
  a ^= (c >> 38);
  b -= c;
  b -= a;
  b ^= (a << 23);
  c -= a;
  c -= b;
  c ^= (b >> 5);
  a -= b;
  a -= c;
  a ^= (c >> 35);
  b -= c;
  b -= a;
  b ^= (a << 49);
  c -= a;
  c -= b;
  c ^= (b >> 11);
  a -= b;
  a -= c;
  a ^= (c >> 12);
  b -= c;
  b -= a;
  b ^= (a << 18);
  c -= a;
  c -= b;
  c ^= (b >> 22);
}

// DOC:
// 二整数 Hash 联结函数 (采用移位实现)
// 
// 参数:
//      x -- 待计算的第一个整数
//      y -- 待计算的第二个整数
inline uint64 FingerprintCat(uint64 x, uint64 y) {
  uint64 b = 0xe08c1d668b756f82;  // more of the golden ratio
  mix(x, b, y);
  return y;
}

// DOC:
// 针对普通类型的 unique_ptr 统一实现
// Trait to select overloads and return types for MakeUnique.
template <typename T>
struct MakeUniqueResult {
  using scalar = std::unique_ptr<T>;
};
// DOC:
// 针对数组类型的 unique_ptr 统一实现
template <typename T>
struct MakeUniqueResult<T[]> {
  using array = std::unique_ptr<T[]>;
};
// DOC:
// 针对定长数组类型的 unique_ptr 统一实现 (暂不支持)
template <typename T, size_t N>
struct MakeUniqueResult<T[N]> {
  using invalid = void;
};

// DOC:
// 针对早期版本 C++ 编译器实现的 make_unique 特性
// 
// 参数:
//      args -- 对象构造参数
// 返回:
//      给定数据类型的 unique_ptr 指针
// MakeUnique<T>(...) is an early implementation of C++14 std::make_unique.
// It is designed to be 100% compatible with std::make_unique so that the
// eventual switchover will be a simple renaming operation.
template <typename T, typename... Args>
typename MakeUniqueResult<T>::scalar MakeUnique(Args &&... args) {  // NOLINT
  return std::unique_ptr<T>(
      new T(std::forward<Args>(args)...));  // NOLINT(build/c++11)
}

// DOC:
// 针对未知边界数组实现的 MakeUnique 重载
// 
// 参数:
//      n -- 数组下标上限值
// 返回:
//      给定数据类型的 unique_ptr 指针
// Overload for array of unknown bound.
// The allocation of arrays needs to use the array form of new,
// and cannot take element constructor arguments.
template <typename T>
typename MakeUniqueResult<T>::array MakeUnique(size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

// DOC:
// 对于已知边界数组 拒绝 MakeUnique 构造
// Reject arrays of known bound.
template <typename T, typename... Args>
typename MakeUniqueResult<T>::invalid MakeUnique(Args &&... /* args */) =
    delete;  // NOLINT

// DOC:
// 针对 std::vector 实现的通用元素清空函数
// 
// 参数:
//      vec -- 待清空的 vector
template <typename T>
void STLDeleteElements(std::vector<T *> *vec) {
  for (auto item : *vec) {
    delete item;
  }
  vec->clear();
}
}  // namespace port

// DOC: 命名空间 sentencepiece::random
namespace random {

// DOC:
// 随机数发生器
// 
// 返回:
//      一个具有线程生存周期的 std::mt19937 随机数发生器
std::mt19937 *GetRandomGenerator();

// DOC:
// 水塘随机采样算法实现
template <typename T>
class ReservoirSampler {
 public:
  explicit ReservoirSampler(std::vector<T> *sampled, size_t size)
      : sampled_(sampled), size_(size), engine_(std::random_device{}()) {}
  explicit ReservoirSampler(std::vector<T> *sampled, size_t size, size_t seed)
      : sampled_(sampled), size_(size), engine_(seed) {}
  virtual ~ReservoirSampler() {}

  void Add(const T &item) {
    if (size_ == 0) return;

    ++total_;
    if (sampled_->size() < size_) {
      sampled_->push_back(item);
    } else {
      std::uniform_int_distribution<size_t> dist(0, total_ - 1);
      const size_t n = dist(engine_);
      if (n < sampled_->size()) (*sampled_)[n] = item;
    }
  }

  size_t total_size() const { return total_; }

 private:
  std::vector<T> *sampled_ = nullptr;
  size_t size_ = 0;
  size_t total_ = 0;
  std::mt19937 engine_;
};

}  // namespace random

// DOC: 命名空间 sentencepiece::util
namespace util {

// DOC:
// 路径合成函数 (单参数 主要用于转换数据类型)
// 
// 参数:
//      path -- 待合成的路径
// 返回:
//      以 std::string 类型表示的路径
inline std::string JoinPath(absl::string_view path) {
  return std::string(path.data(), path.size());
}

// DOC:
// 路径合成函数 (针对 Windows/Linux 分别适配)
// 
// 参数:
//      first -- 待合成的首个路径
//      rest -- 待合成的其余路径部分
// 返回:
//      以 std::string 类型表示的合成后的路径
template <typename... T>
inline std::string JoinPath(absl::string_view first, const T &... rest) {
#ifdef OS_WIN
  return JoinPath(first) + "\\" + JoinPath(rest...);
#else
  return JoinPath(first) + "/" + JoinPath(rest...);
#endif
}

// DOC:
// 将错误编码处理为提示信息文本并返回
// 
// 参数:
//      errnum -- 错误编码
// 返回:
//      包含错误编码的提示信息文本
std::string StrError(int errnum);

// DOC:
// 返回当前执行的状态
// 
// 返回:
//      一个 Status 型表示的状态
inline Status OkStatus() { return Status(); }

// DOC:
// 错误定义通用宏 构建不同类型错误与错误代码的映射
#define DECLARE_ERROR(FUNC, CODE)                          \
  inline util::Status FUNC##Error(absl::string_view str) { \
    return util::Status(error::CODE, str.data());          \
  }                                                        \
  inline bool Is##FUNC(const util::Status &status) {       \
    return status.code() == error::CODE;                   \
  }

// DOC:
// 错误定义过程 构建不同类型错误与错误代码的映射
DECLARE_ERROR(Cancelled, CANCELLED)
DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DECLARE_ERROR(NotFound, NOT_FOUND)
DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DECLARE_ERROR(Unavailable, UNAVAILABLE)
DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
DECLARE_ERROR(Internal, INTERNAL)
DECLARE_ERROR(Aborted, ABORTED)
DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DECLARE_ERROR(DataLoss, DATA_LOSS)
DECLARE_ERROR(Unknown, UNKNOWN)
DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)

// DOC:
// 状态构建器 用于生成当前状态信息
class StatusBuilder {
 public:
  explicit StatusBuilder(error::Code code) : code_(code) {}

  template <typename T>
  StatusBuilder &operator<<(const T &value) {
    os_ << value;
    return *this;
  }

  operator Status() const { return Status(code_, os_.str()); }

 private:
  error::Code code_;
  std::ostringstream os_;
};

// DOC:
// 变量条件检测宏 检测不通过则返回一个 StatusBuilder
#define CHECK_OR_RETURN(condition)                                     \
  if (condition) {                                                     \
  } else /* NOLINT */                                                  \
    return ::sentencepiece::util::StatusBuilder(util::error::INTERNAL) \
           << __FILE__ << "(" << __LINE__ << ") [" << #condition << "] "

// DOC:
// 变量检测条件定义
#define CHECK_EQ_OR_RETURN(a, b) CHECK_OR_RETURN((a) == (b))
#define CHECK_NE_OR_RETURN(a, b) CHECK_OR_RETURN((a) != (b))
#define CHECK_GE_OR_RETURN(a, b) CHECK_OR_RETURN((a) >= (b))
#define CHECK_LE_OR_RETURN(a, b) CHECK_OR_RETURN((a) <= (b))
#define CHECK_GT_OR_RETURN(a, b) CHECK_OR_RETURN((a) > (b))
#define CHECK_LT_OR_RETURN(a, b) CHECK_OR_RETURN((a) < (b))

}  // namespace util

// DOC: 命名空间 sentencepiece::thread
namespace thread {

// DOC:
// 线程池实现
class ThreadPool {
 public:
  ThreadPool() {}
  virtual ~ThreadPool() {
    for (auto &task : tasks_) {
      task.join();
    }
  }

  // DOC:
  // 线程调度实现
  void Schedule(std::function<void()> closure) { tasks_.emplace_back(closure); }

 private:
  std::vector<std::thread> tasks_;
};
}  // namespace thread
}  // namespace sentencepiece
#endif  // UTIL_H_
