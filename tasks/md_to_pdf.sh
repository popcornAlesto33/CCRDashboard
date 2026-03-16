#!/bin/bash
# Convert summary_for_client.md to PDF using Chrome headless
# Usage: ./tasks/md_to_pdf.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="$SCRIPT_DIR/summary_for_client.md"
HTML="$SCRIPT_DIR/summary_for_client.html"
PDF="$SCRIPT_DIR/summary_for_client.pdf"

# Convert MD to standalone HTML with embedded images and inline CSS
# (cd into tasks/ so relative image paths resolve)
cd "$SCRIPT_DIR"
pandoc "summary_for_client.md" -o "$HTML" \
  --standalone \
  --embed-resources \
  --metadata title=" " \
  --variable maxwidth=none \
  --css <(cat <<'CSS'
body {
  font-family: -apple-system, Helvetica Neue, Arial, sans-serif;
  max-width: 720px;
  margin: 0 auto;
  padding: 0;
  font-size: 13px;
  line-height: 1.5;
  color: #222;
}
h1 { font-size: 22px; border-bottom: 2px solid #222; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 17px; margin-top: 24px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 14px; margin-top: 18px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12px; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background: #f5f5f5; font-weight: 600; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
pre { background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 12px; }
pre code { background: none; padding: 0; }
img { max-width: 100%; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
CSS
)

# Print to PDF with Chrome headless — no header/footer, moderate margins
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$PDF" \
  --print-to-pdf-no-header \
  "$HTML" 2>/dev/null

echo "PDF created: $PDF ($(du -h "$PDF" | cut -f1 | xargs))"
