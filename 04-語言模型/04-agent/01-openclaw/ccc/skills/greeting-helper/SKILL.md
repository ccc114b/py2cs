---
name: greeting-helper
description: Generate short, natural greeting and reply suggestions in Traditional Chinese or English with adjustable tone (formal, friendly, concise). Use when the user asks for opening lines, quick chat replies, or message tone rewrites.
---

# Greeting Helper

## Quick Start

- Ask for context first: recipient, channel, tone, and length.
- Output 3 candidate lines by default.
- Keep each option short and directly usable.

## Workflow

1. Identify scenario (first greeting / reply / follow-up).
2. Match tone (formal / neutral / friendly).
3. Generate 3 options with slight style differences.
4. If requested, rewrite into another tone or language.

## Templates

### First greeting

- 「嗨 [名字]，早安！今天想先從哪件事開始？」
- 「你好 [名字]，我在線上了，需要我先處理什麼？」

### Friendly reply

- 「收到，我馬上幫你處理。」
- 「好，我這邊直接開始，等等回報結果。」

### Formal reply

- 「已收到，我將立即處理並回報進度。」
- 「了解，我先執行，再向您更新結果。」

## Output Rules

- Prefer Traditional Chinese unless user asks otherwise.
- Avoid filler and over-polite boilerplate.
- Keep one line per option unless user requests longer format.
