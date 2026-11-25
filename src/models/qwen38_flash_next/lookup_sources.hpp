#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_LOOKUP_SOURCES_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_LOOKUP_SOURCES_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Tool results often quote a file in a spelling the model does not write
/// back: a coding agent's Read numbers every line (`    12→text`,
/// `12\ttext`), and JSON results escape newlines and quotes. Prompt lookup
/// copies token runs, so such a source breaks every copy at each line start
/// or escape. These are the texts as the model would write them: the listing
/// without its numbers, and long JSON string literals unescaped (themselves
/// stripped when they hold a listing). Empty when the result has neither.
///
/// Proposals copied from these texts are verified like any other, so a
/// wrong guess costs only a rejected draft; the output never changes.
class LookupSources {
public:
  /// A listing needs this many numbered lines, and 3 in 4 non-empty lines
  /// numbered (a trailing note is not), so ordinary text with a few numbered
  /// items stays as it is.
  static constexpr std::size_t kMinNumberedLines = 3;
  /// Shorter literals rarely hold a file worth copying.
  static constexpr std::size_t kMinLiteralBytes = 64;

  [[nodiscard]] static std::vector<std::string> Texts(std::string_view text) {
    std::vector<std::string> out;
    std::string stripped;
    if (StripListing(text, &stripped))
      out.push_back(std::move(stripped));
    for (std::size_t i = 0; i < text.size(); ++i) {
      if (text[i] != '"')
        continue;
      std::string raw;
      std::size_t end = 0;
      bool escaped = false;
      if (!ReadLiteral(text, i + 1, &raw, &end, &escaped)) {
        continue;
      }
      i = end;
      if (!escaped || raw.size() < kMinLiteralBytes ||
          raw.find('\n') == std::string::npos)
        continue;
      std::string listing;
      out.push_back(StripListing(raw, &listing) ? std::move(listing)
                                                : std::move(raw));
    }
    return out;
  }

  /// `text` without its line numbers when it is a numbered listing.
  [[nodiscard]] static bool StripListing(std::string_view text,
                                         std::string* out) {
    std::size_t numbered = 0;
    std::size_t lines = 0;
    std::string body;
    for (std::size_t start = 0; start <= text.size();) {
      auto end = text.find('\n', start);
      if (end == std::string_view::npos)
        end = text.size();
      auto line = text.substr(start, end - start);
      if (!line.empty() && line.back() == '\r')
        line.remove_suffix(1);
      std::string_view rest;
      if (Numbered(line, &rest)) {
        if (numbered != 0)
          body.push_back('\n');
        body.append(rest);
        ++numbered;
      }
      if (line.find_first_not_of(" \t") != std::string_view::npos)
        ++lines;
      start = end + 1;
    }
    if (numbered < kMinNumberedLines || 4 * numbered < 3 * lines)
      return false;
    *out = std::move(body);
    return true;
  }

private:
  /// `   12→text` or `12\ttext`: optional spaces, digits, an arrow or a tab.
  static bool Numbered(std::string_view line, std::string_view* rest) {
    std::size_t i = line.find_first_not_of(' ');
    if (i == std::string_view::npos)
      return false;
    const auto digits = i;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9')
      ++i;
    if (i == digits)
      return false;
    constexpr std::string_view kArrow = "\xE2\x86\x92";
    if (line.substr(i).starts_with(kArrow)) {
      *rest = line.substr(i + kArrow.size());
      return true;
    }
    if (i < line.size() && line[i] == '\t') {
      *rest = line.substr(i + 1);
      return true;
    }
    return false;
  }

  static void AppendUtf8(std::uint32_t code, std::string* out) {
    if (code < 0x80) {
      out->push_back(static_cast<char>(code));
    } else if (code < 0x800) {
      out->push_back(static_cast<char>(0xC0 | (code >> 6)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code < 0x10000) {
      out->push_back(static_cast<char>(0xE0 | (code >> 12)));
      out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      out->push_back(static_cast<char>(0xF0 | (code >> 18)));
      out->push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  static bool Hex4(std::string_view text, std::size_t at, std::uint32_t* out) {
    if (at + 4 > text.size())
      return false;
    std::uint32_t value = 0;
    for (std::size_t k = 0; k < 4; ++k) {
      const char c = text[at + k];
      value <<= 4;
      if (c >= '0' && c <= '9')
        value |= static_cast<std::uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f')
        value |= static_cast<std::uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        value |= static_cast<std::uint32_t>(c - 'A' + 10);
      else
        return false;
    }
    *out = value;
    return true;
  }

  /// A JSON string literal starting after its opening quote at `begin`.
  /// False when it is not one (an unknown escape, a raw newline, no end).
  static bool ReadLiteral(std::string_view text, std::size_t begin,
                          std::string* raw, std::size_t* end, bool* escaped) {
    for (std::size_t i = begin; i < text.size(); ++i) {
      const char c = text[i];
      if (c == '"') {
        *end = i;
        return true;
      }
      if (c == '\n')
        return false;
      if (c != '\\') {
        raw->push_back(c);
        continue;
      }
      if (++i >= text.size())
        return false;
      *escaped = true;
      switch (text[i]) {
        case '"':
          raw->push_back('"');
          break;
        case '\\':
          raw->push_back('\\');
          break;
        case '/':
          raw->push_back('/');
          break;
        case 'b':
          raw->push_back('\b');
          break;
        case 'f':
          raw->push_back('\f');
          break;
        case 'n':
          raw->push_back('\n');
          break;
        case 'r':
          raw->push_back('\r');
          break;
        case 't':
          raw->push_back('\t');
          break;
        case 'u': {
          std::uint32_t code = 0;
          if (!Hex4(text, i + 1, &code))
            return false;
          i += 4;
          std::uint32_t low = 0;
          if (code >= 0xD800 && code < 0xDC00 && i + 6 < text.size() &&
              text[i + 1] == '\\' && text[i + 2] == 'u' &&
              Hex4(text, i + 3, &low) && low >= 0xDC00 && low < 0xE000) {
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            i += 6;
          }
          AppendUtf8(code, raw);
          break;
        }
        default:
          return false;
      }
    }
    return false;
  }
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_LOOKUP_SOURCES_HPP_
