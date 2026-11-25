#include "src/models/qwen38_flash_next/lookup_sources.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using gufo::models::qwen38_flash_next::LookupSources;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// A Read listing (arrow or tab after the number) comes back without numbers;
// a trailing note that is not numbered is dropped.
void CheckListing() {
  const std::string arrow =
      "     1→#include <cstdio>\n"
      "     2→\n"
      "     3→int main() {\n"
      "     4→  return 0;\n"
      "     5→}\n";
  auto texts = LookupSources::Texts(arrow);
  Require(texts.size() == 1, "an arrow listing gave no source");
  Require(texts[0] == "#include <cstdio>\n\nint main() {\n  return 0;\n}",
          "an arrow listing kept its numbers");
  texts = LookupSources::Texts(
      "1\tfirst\n2\tsecond\n3\tthird\n\n"
      "<system-reminder>note</system-reminder>\n");
  Require(texts.size() == 1 && texts[0] == "first\nsecond\nthird",
          "a tab listing was not stripped");
  // Prose with a couple of numbered items is not a listing.
  Require(LookupSources::Texts("Steps:\n1\tone\n2\ttwo\nthen more prose\n"
                               "and more\nand more\n")
              .empty(),
          "prose was treated as a listing");
}

// Long JSON literals with escapes are unescaped; short or plain ones are not
// sources, and a listing inside a literal is stripped too.
void CheckJson() {
  const std::string body =
      "line one of the file\n\"quoted\" \\ path\tand a tab\n"
      "caf\xC3\xA9 and \xF0\x9F\x98\x80 end\n";
  const std::string json =
      R"({"ok":true,"short":"a\nb","content":"line one of the file\n\"quoted\")"
      R"( \\ path\tand a tab\ncafé and 😀 end\n"})";
  auto texts = LookupSources::Texts(json);
  Require(texts.size() == 1, "a long escaped literal gave no source");
  Require(texts[0] == body, "a JSON literal was not unescaped exactly");
  const std::string listing =
      R"({"file":"     1→alpha beta gamma delta\n     2→epsilon zeta )"
      R"(eta theta\n     3→iota kappa lambda mu\n"})";
  texts = LookupSources::Texts(listing);
  Require(texts.size() == 1 &&
              texts[0] ==
                  "alpha beta gamma delta\nepsilon zeta eta theta\n"
                  "iota kappa lambda mu",
          "a listing inside a literal was not stripped");
  texts = LookupSources::Texts(
      R"({"s":"a lone \ud800 surrogate inside a long enough literal with\n)"
      R"(a newline so that it counts"})");
  Require(
      texts.size() == 1 && texts[0].find("\xEF\xBF\xBD") != std::string::npos,
      "a lone surrogate was not replaced");
  Require(LookupSources::Texts(R"({"a":"no escapes in this fairly long )"
                               R"(string value that has none at all"})")
              .empty(),
          "an unescaped literal became a source");
  Require(LookupSources::Texts(R"({"bad":"\q and more text to make it )"
                               R"(long enough to count\n\n\n"})")
              .empty(),
          "an invalid escape became a source");
}

}  // namespace

int main() {
  try {
    CheckListing();
    CheckJson();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
  std::cout << "qwen38_flash_next lookup_sources tests passed\n";
  return 0;
}
