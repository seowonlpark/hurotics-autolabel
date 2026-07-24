# Comment rules

How code comments are written in this repo. Applies to every source file.

## Length
- Never exceed two, maybe three lines. Most should be one line.
- Keep them short.

## Style
- Drop articles: "human" not "a human", "number" not "the number".
- No em dashes. ASCII only (cp949 consoles choke on non-ASCII).
- No divider dashes: write `# section name`, never `# --- section name ---`.

## Content
- A comment briefly describes the code for readability. That is all.
- Not a changelog. No "was X, now Y", no "dropped in Section 10.8", no history.
- No embedded metrics (`0.69->0.86`, AUC numbers, cost figures).
- No excessive design rationale. That lives in README / DOMAIN_NOTES / BUILDLOG; a
  short pointer (e.g. `Section 10`) is fine, the full argument is not.

## Quick check
If a comment reads like documentation, prose, or a commit message, cut it down to what
the next reader needs to follow the code.
