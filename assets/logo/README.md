# mnem brand assets

Minimal, deliberately simple. Two colors, one geometric mark, one
wordmark. Everything else is composition.

## Files

| File | Purpose | Dimensions |
|---|---|---|
| `mnem-logo.svg` | The primary mark. Diamond graph with rust tilted-square joints. Use in README, docs, slides, favicons. | Tight 196×196 viewBox (scales to any size) |
| `mnem-social.svg` | Social / OpenGraph preview card with logo + wordmark + tagline + URL. Used as the image that appears when the repo link is shared on Twitter, Slack, Discord, LinkedIn, etc. | 1280×640 (GitHub's required aspect ratio) |
| `mnem-social.png` | Pre-rasterised copy of `mnem-social.svg` for direct upload to GitHub's Social Preview setting. | 1280×640 |

## Colors

| Role | Hex | Usage |
|---|---|---|
| Slate | `#1f2a36` | Structural lines, primary text |
| Rust  | `#d1642b` | Node joints, accents, URLs, the rust bar on the social card |
| Cream | `#f7f5f0` | Background for the social card (warm enough to read on light and dark chrome) |
| Dim   | `#556070` | Secondary text |

## Geometry

`mnem-logo.svg` is four slate line segments forming a diamond (each
line 20px wide, butt-capped at the apex), plus four rust tilted
squares whose edges align exactly with the stroke boundaries of the
two converging lines at each corner. That's it. No gradients, no
fills-inside-fills, no text - the whole brand stands on composition.

Scale freely. At 16×16 (favicon), the tilted squares still read as
joints. At 1024×1024 (print), the line widths and square proportions
stay right.

## Using the social preview

GitHub's "Social preview" feature only accepts **PNG / JPG**. If you
want the mnem social card to appear when someone pastes the repo URL
into Twitter / Slack / Discord / LinkedIn:

1. **Export** `mnem-social.svg` to a 1280×640 PNG. Any of these works:

   ```bash
   # rsvg-convert (librsvg). Fastest, no GUI.
   rsvg-convert -w 1280 -h 640 mnem-social.svg -o mnem-social.png

   # Inkscape CLI
   inkscape mnem-social.svg --export-type=png --export-width=1280 --export-height=640 --export-filename=mnem-social.png

   # ImageMagick
   magick mnem-social.svg -resize 1280x640 mnem-social.png
   ```

2. **Upload** the PNG via GitHub: `Settings → General → Social
   preview → Upload an image`. You need admin rights on the repo.

3. Sanity check by pasting `https://github.com/Uranid/mnem`
   into Twitter's compose box or a Slack DM. The rich-card preview
   should show the uploaded image within a minute.

There's no REST API endpoint for programmatic upload - GitHub keeps
this to the web UI.

## The three "logo" spots on a GitHub repo

Don't confuse them; they're filled in three different places.

| Where it shows | What it is | How to set |
|---|---|---|
| The small square next to `Uranid/mnem` at the top of the repo page | **Organization avatar** (shared across every repo the org owns). Defaults to the letters of the org name on a gray background if unset. | Org settings: `https://github.com/organizations/Uranid/settings/profile` - upload an image under "Profile picture". Re-use `mnem-logo.svg` rasterised to a 500×500 PNG if you want. |
| The card that shows when the repo URL is pasted into Twitter / Slack / Discord / LinkedIn / etc. | **Social preview** (per-repo, 1280×640). If unset, GitHub auto-generates a default card. | Repo settings: `https://github.com/Uranid/mnem/settings` → General → Social preview → Upload `assets/logo/mnem-social.png`. |
| Inline at the top of the README | Just a regular Markdown / HTML image. Only affects how the README renders. | Already done - the `<img src="assets/logo/mnem-logo.svg">` at the top of the root README. |

The GitHub REST API **does not expose** org-avatar upload or social-preview upload; both are web-UI only. You need the admin-ish permission corresponding to each (org admin / repo admin).

## Using the logo elsewhere

Embed in any markdown / HTML:

```html
<img src="https://raw.githubusercontent.com/Uranid/mnem/main/assets/logo/mnem-logo.svg"
     alt="mnem" width="128" />
```

Modify the stroke widths, but please keep the two-color palette and
the four-node diamond shape. If you need a single-color version,
set both `#1f2a36` and `#d1642b` to the same color.
