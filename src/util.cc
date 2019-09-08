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

#include "util.h"
#include <iostream>

// DOC: 命名空间 sentencepiece::string_util
namespace sentencepiece {
namespace string_util {

// DOC:
// 内置文本分割模板函数
// 将以 delim 分隔的 str 切分为 vector 数组返回
// 
// 参数:
//      str -- 原始文本
//      delim -- 分隔符
//      allow_empty -- 切分后元素是否可空
// 返回:
//      切分后的 vector 数组
template <typename T>
std::vector<T> SplitInternal(const T &str, const T &delim, bool allow_empty) {
  std::vector<T> result;
  size_t current_pos = 0;
  size_t found_pos = 0;
  while ((found_pos = str.find_first_of(delim, current_pos)) != T::npos) {
    if ((allow_empty && found_pos >= current_pos) ||
        (!allow_empty && found_pos > current_pos)) {
      result.push_back(str.substr(current_pos, found_pos - current_pos));
    }
    current_pos = found_pos + 1;
  }
  // 末尾项的处理
  if (str.size() > current_pos) {
    result.push_back(str.substr(current_pos, str.size() - current_pos));
  }
  return result;
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
                               bool allow_empty) {
  return SplitInternal<std::string>(str, delim, allow_empty);
}

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
                                          bool allow_empty) {
  return SplitInternal<absl::string_view>(str, delim, allow_empty);
}

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
                 absl::string_view delim) {
  std::string result;
  if (!tokens.empty()) {
    result.append(tokens[0]);
  }
  for (size_t i = 1; i < tokens.size(); ++i) {
    result.append(delim.data(), delim.size());
    result.append(tokens[i]);
  }
  return result;
}

// DOC:
// 针对 int 实现的文本合并函数
// 将传入的 int 整形数组以 delim 作为分隔符合并为一个字符串
// 
// 参数:
//      tokens -- 整形数组
//      delim -- 分隔符
// 返回:
//      合并后的 std::string 类型字符串
std::string Join(const std::vector<int> &tokens, absl::string_view delim) {
  std::string result;
  char buf[32];
  if (!tokens.empty()) {
    const size_t len = Itoa(tokens[0], buf);
    result.append(buf, len);
  }
  for (size_t i = 1; i < tokens.size(); ++i) {
    result.append(delim.data(), delim.size());
    const size_t len = Itoa(tokens[i], buf);
    result.append(buf, len);
  }
  return result;
}

// DOC:
// 针对 absl::string_view 实现的子文本替换函数
// 调用内部方法 StringReplace 实现的带返回值版本
// 
// 参数:
//      s -- 原始文本
//      oldsub -- 待替换子串
//      newsub -- 用作替换的子串
//      replace_all -- 是否替换所有子串
// 返回:
//      替换后的 std::string 类型字符串
std::string StringReplace(absl::string_view s, absl::string_view oldsub,
                          absl::string_view newsub, bool replace_all) {
  std::string ret;
  StringReplace(s, oldsub, newsub, replace_all, &ret);
  return ret;
}

// DOC:
// 针对 absl::string_view 实现的子文本替换函数
// 将原始文本 s 中的 oldsub 子串替换为 newsub 子串
// 
// 参数:
//      s -- 原始文本
//      oldsub -- 待替换子串
//      newsub -- 用作替换的子串
//      replace_all -- 是否替换所有子串
//      res -- 保存替换后字符串的指针
void StringReplace(absl::string_view s, absl::string_view oldsub,
                   absl::string_view newsub, bool replace_all,
                   std::string *res) {
  if (oldsub.empty()) {
    res->append(s.data(), s.size());
    return;
  }

  absl::string_view::size_type start_pos = 0;
  do {
    const absl::string_view::size_type pos = s.find(oldsub, start_pos);
    if (pos == absl::string_view::npos) {
      break;
    }
    res->append(s.data() + start_pos, pos - start_pos);
    res->append(newsub.data(), newsub.size());
    start_pos = pos + oldsub.size();
  } while (replace_all);
  res->append(s.data() + start_pos, s.size() - start_pos);
}

// DOC:
// UTF-8 字符解码函数
// 
// 参数:
//      begin -- UTF-8 字符起始位置指针
//      end -- UTF-8 字符终止位置指针
//      mblen -- 保存解码后所使用的字节数变量的指针
// 返回:
//      一个 char32 字符，即 UTF-8 解码的结果
// mblen sotres the number of bytes consumed after decoding.
char32 DecodeUTF8(const char *begin, const char *end, size_t *mblen) {
  const size_t len = end - begin;

  if (static_cast<unsigned char>(begin[0]) < 0x80) {
    *mblen = 1;
    return static_cast<unsigned char>(begin[0]);
  } else if (len >= 2 && (begin[0] & 0xE0) == 0xC0) {
    const char32 cp = (((begin[0] & 0x1F) << 6) | ((begin[1] & 0x3F)));
    if (IsTrailByte(begin[1]) && cp >= 0x0080 && IsValidCodepoint(cp)) {
      *mblen = 2;
      return cp;
    }
  } else if (len >= 3 && (begin[0] & 0xF0) == 0xE0) {
    const char32 cp = (((begin[0] & 0x0F) << 12) | ((begin[1] & 0x3F) << 6) |
                       ((begin[2] & 0x3F)));
    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) && cp >= 0x0800 &&
        IsValidCodepoint(cp)) {
      *mblen = 3;
      return cp;
    }
  } else if (len >= 4 && (begin[0] & 0xf8) == 0xF0) {
    const char32 cp = (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3F) << 12) |
                       ((begin[2] & 0x3F) << 6) | ((begin[3] & 0x3F)));
    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) &&
        IsTrailByte(begin[3]) && cp >= 0x10000 && IsValidCodepoint(cp)) {
      *mblen = 4;
      return cp;
    }
  }

  // Invalid UTF-8.
  *mblen = 1;
  return kUnicodeError;
}

// DOC:
// 逐个字符地检测给定文本是否符合 UTF-8 规范
// 
// 参数:
//      str -- 待检测的文本
// 返回:
//      一个 bool 类型值，表示给定的文本是否符合 UTF-8 规范
bool IsStructurallyValid(absl::string_view str) {
  const char *begin = str.data();
  const char *end = str.data() + str.size();
  size_t mblen = 0;
  while (begin < end) {
    const char32 c = DecodeUTF8(begin, end, &mblen);
    if (c == kUnicodeError && mblen != 3) return false;
    if (!IsValidCodepoint(c)) return false;
    begin += mblen;
  }
  return true;
}

// DOC:
// 将字符编码为 UTF-8
// 
// 参数:
//      c -- 待编码字符
//      output -- 保存编码结果的字符串指针
// 返回:
//      一个 ULL 类型值，表示编码 UTF-8 字符后所占的字节数
size_t EncodeUTF8(char32 c, char *output) {
  if (c <= 0x7F) {
    *output = static_cast<char>(c);
    return 1;
  }

  if (c <= 0x7FF) {
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xC0 | c;
    return 2;
  }

  // if `c` is out-of-range, convert it to REPLACEMENT CHARACTER (U+FFFD).
  // This treatment is the same as the original runetochar.
  if (c > 0x10FFFF) c = kUnicodeError;

  if (c <= 0xFFFF) {
    output[2] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xE0 | c;
    return 3;
  }

  output[3] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[2] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[1] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[0] = 0xF0 | c;

  return 4;
}

// DOC:
// 将 Unicode 字符编码为 UTF-8 (调用 UnicodeTextToUTF8 实现)
// 
// 参数:
//      c -- 待编码 Unicode 字符
// 返回:
//      一个 std::string 类型，包含编码后的 UTF-8 数据
std::string UnicodeCharToUTF8(const char32 c) { return UnicodeTextToUTF8({c}); }

// DOC:
// 将 UTF-8 文本编码为 Unicode 文本
// 
// 参数:
//      utf8 -- 待编码的 UTF-8 文本
// 返回:
//      一个 UnicodeText 类型，包含编码后的 Unicode 数据
UnicodeText UTF8ToUnicodeText(absl::string_view utf8) {
  UnicodeText uc;
  const char *begin = utf8.data();
  const char *end = utf8.data() + utf8.size();
  while (begin < end) {
    size_t mblen;
    const char32 c = DecodeUTF8(begin, end, &mblen);
    uc.push_back(c);
    begin += mblen;
  }
  return uc;
}

// DOC:
// 将 Unicode 文本编码为 UTF-8 文本
// 
// 参数:
//      utext -- 待编码的 Unicode 文本
// 返回:
//      一个 std::string 类型，包含编码后的 UTF-8 数据
std::string UnicodeTextToUTF8(const UnicodeText &utext) {
  char buf[8];
  std::string result;
  for (const char32 c : utext) {
    const size_t mblen = EncodeUTF8(c, buf);
    result.append(buf, mblen);
  }
  return result;
}
}  // namespace string_util

// DOC: 命名空间 sentencepiece::random
// DOC:
// 将 Unicode 文本编码为 UTF-8 文本
// 
// 参数:
//      utext -- 待编码的 Unicode 文本
// 返回:
//      一个 std::string 类型，包含编码后的 UTF-8 数据
namespace random {
#ifdef SPM_NO_THREADLOCAL
namespace {
// DOC:
// 针对不具备 thread_local 特性编译器实现的线程周期存储类型
class RandomGeneratorStorage {
 public:
  RandomGeneratorStorage() {
    pthread_key_create(&key_, &RandomGeneratorStorage::Delete);
  }
  virtual ~RandomGeneratorStorage() { pthread_key_delete(key_); }

  std::mt19937 *Get() {
    auto *result = static_cast<std::mt19937 *>(pthread_getspecific(key_));
    if (result == nullptr) {
      result = new std::mt19937(std::random_device{}());
      pthread_setspecific(key_, result);
    }
    return result;
  }

 private:
  static void Delete(void *value) { delete static_cast<std::mt19937 *>(value); }
  pthread_key_t key_;
};
}  // namespace

std::mt19937 *GetRandomGenerator() {
  static RandomGeneratorStorage *storage = new RandomGeneratorStorage;
  return storage->Get();
}
#else
// DOC:
// 随机数发生器
// 
// 返回:
//      一个具有线程生存周期的 std::mt19937 随机数发生器
std::mt19937 *GetRandomGenerator() {
  thread_local static std::mt19937 mt(std::random_device{}());
  return &mt;
}
#endif
}  // namespace random

// DOC: 命名空间 sentencepiece::util
namespace util {

// DOC:
// 将错误编码处理为提示信息文本并返回
// 
// 参数:
//      errnum -- 错误编码
// 返回:
//      包含错误编码的提示信息文本
std::string StrError(int errnum) {
  constexpr int kStrErrorSize = 1024;
  char buffer[kStrErrorSize];
  char *str = nullptr;
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  str = strerror_r(errnum, buffer, kStrErrorSize - 1);
#elif defined(_WIN32)
  strerror_s(buffer, kStrErrorSize - 1, errnum);
  str = buffer;
#else
  strerror_r(errnum, buffer, kStrErrorSize - 1);
  str = buffer;
#endif
  std::ostringstream os;
  os << str << " Error #" << errnum;
  return os.str();
}
}  // namespace util

// DOC:
// 对于 Win32 系统，额外实现基于 Win32 API 的宽字节和 UTF-8 编码之间的转换
#ifdef OS_WIN
// DOC: 命名空间 sentencepiece::win32
namespace win32 {
// DOC:
// 将 UTF-8 编码文本转换为宽字节编码文本
// 
// 参数:
//      input -- 待转换的 UTF-8 编码文本
// 返回:
//      一个 std::wstring 类型的宽字节编码文本
std::wstring Utf8ToWide(const std::string &input) {
  int output_length =
      ::MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, nullptr, 0);
  output_length = output_length <= 0 ? 0 : output_length - 1;
  if (output_length == 0) {
    return L"";
  }
  std::unique_ptr<wchar_t[]> input_wide(new wchar_t[output_length + 1]);
  const int result = ::MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1,
                                           input_wide.get(), output_length + 1);
  std::wstring output;
  if (result > 0) {
    output.assign(input_wide.get());
  }
  return output;
}

// DOC:
// 将宽字节编码文本转换为 UTF-8 编码文本
// 
// 参数:
//      input -- 待转换的宽字节编码文本
// 返回:
//      一个 std::string 类型的 UTF-8 编码文本
std::string WideToUtf8(const std::wstring &input) {
  const int output_length = ::WideCharToMultiByte(CP_UTF8, 0, input.c_str(), -1,
                                                  nullptr, 0, nullptr, nullptr);
  if (output_length == 0) {
    return "";
  }

  std::unique_ptr<char[]> input_encoded(new char[output_length + 1]);
  const int result =
      ::WideCharToMultiByte(CP_UTF8, 0, input.c_str(), -1, input_encoded.get(),
                            output_length + 1, nullptr, nullptr);
  std::string output;
  if (result > 0) {
    output.assign(input_encoded.get());
  }
  return output;
}
}  // namespace win32
#endif
}  // namespace sentencepiece
