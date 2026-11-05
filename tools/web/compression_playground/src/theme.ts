// Copyright (c) Meta Platforms, Inc. and affiliates.

import {createSystem, defaultConfig, defineConfig, defineSlotRecipe} from '@chakra-ui/react';
import type {CompressorName} from './compressors.ts';

/**
 * Compression Playground palette, sourced from the Figma file. Components
 * reference the role names (`color="pg.ink"`, `bg="pg.surface"`) instead of
 * hardcoded hex values.
 *
 * Three constraints here are easy to break:
 *
 * - Colours belong under `semanticTokens`, not `tokens`. The latter is the
 *   primitive layer and holds one value per entry, so a palette defined there
 *   cannot express dark mode at all.
 * - Dark variants hang off `_osDark`, which Chakra maps to
 *   `@media (prefers-color-scheme: dark)`. `_dark` keys off a colour-mode class
 *   instead and needs a provider this app does not mount, so it would silently
 *   never match.
 * - The `pg` prefix keeps these clear of Chakra's own semantic colours, which
 *   already claim bare names such as `border`, `bg` and `fg`.
 *
 * Dark values are derived rather than designed -- the Figma file only covers
 * light mode. Every foreground/background pair clears WCAG AA against the
 * surface it sits on. `pg.muted` is deliberately darker than Figma's #94a3b8:
 * that value reaches only 2.56:1, and it carries the 12px uppercase section
 * headings, which are too small to qualify for the large-text threshold.
 */
const playgroundConfig = defineConfig({
  globalCss: {
    html: {scrollbarGutter: 'stable'},
  },
  theme: {
    slotRecipes: {
      helperTooltip: defineSlotRecipe({
        slots: ['trigger', 'positioner', 'content', 'arrow', 'arrowTip'],
        base: {
          content: {
            '--tooltip-bg': 'colors.pg.ink',
            bg: 'var(--tooltip-bg)',
            color: 'pg.onInk',
            borderWidth: '1px',
            borderColor: 'pg.ink',
            borderRadius: '6px',
            px: '10px',
            py: '8px',
            maxW: '300px',
            fontSize: '12px',
            fontWeight: 'normal',
            lineHeight: '1.5',
            boxShadow: 'md',
          },
          arrowTip: {borderColor: 'pg.ink'},
        },
      }),
    },
    semanticTokens: {
      colors: {
        pg: {
          // Surfaces
          pageBg: {value: {base: '#ffffff', _osDark: '#1f2129'}},
          surface: {value: {base: '#ffffff', _osDark: '#282b35'}},
          canvas: {value: {base: '#fafafb', _osDark: '#242731'}},
          chip: {value: {base: '#eceff1', _osDark: '#343844'}},
          callout: {value: {base: '#edf2ff', _osDark: '#252f4a'}},
          accentBg: {value: {base: '#eff6ff', _osDark: '#1e3a5f'}},
          border: {value: {base: '#e2e8f0', _osDark: '#424754'}},

          // Foregrounds. `ink` doubles as the step-badge fill, so it inverts
          // with the scheme and `onInk` inverts against it.
          ink: {value: {base: '#0f172a', _osDark: '#f1f5f9'}},
          onInk: {value: {base: '#ffffff', _osDark: '#0f172a'}},
          secondary: {value: {base: '#475569', _osDark: '#bfc1c6'}},
          muted: {value: {base: '#64748b', _osDark: '#93979f'}},
          faint: {value: {base: '#667385', _osDark: '#94a3b8'}},
          accent: {value: {base: '#2563eb', _osDark: '#60a5fa'}},
          success: {value: {base: '#059669', _osDark: '#34d399'}},
          // `pg.success` as 13px text reaches only 3.77:1 on white, so the
          // done line takes the darker green the design already specifies.
          successText: {value: {base: '#065f46', _osDark: '#6ee7a8'}},
          danger: {value: {base: '#b91c1c', _osDark: '#fca5a5'}},

          // Step 2/3 controls. The run button is indigo rather than the blue
          // accent, and the training note pairs the accent wash with a blue
          // border instead of the neutral one.
          // The dark variant is darker than Figma's #6366f1: that only reaches
          // 4.47:1 behind the button's 14px bold white label.
          primary: {value: {base: '#4f46e5', _osDark: '#5b54ee'}},
          primaryHover: {value: {base: '#4338ca', _osDark: '#4f46e5'}},
          // The run button while it works. Figma's #93c5fd carries bold white
          // at 1.80:1, and 14px bold is not large text, so this is darkened to
          // 4.94:1 -- which also keeps the button itself at 3:1 against the
          // card, the threshold for a shape rather than a label.
          primaryBusy: {value: {base: '#426fbe', _osDark: '#3f63a8'}},
          infoBorder: {value: {base: '#bfdbfe', _osDark: '#2f4a6b'}},

          // The WebAssembly-speed notice under the measurements. It sits on
          // `pg.callout`, which the design already uses at the same value.
          noticeBorder: {value: {base: '#ccdef7', _osDark: '#2f4a6b'}},
          noticeFg: {value: {base: '#2e4066', _osDark: '#cbd9f0'}},

          // One colour per codec, reached through `CODEC_COLOR` below.
          openzl: {value: {base: '#10b981', _osDark: '#34d399'}},
          zstd: {value: {base: '#3b82f6', _osDark: '#60a5fa'}},
          gzip: {value: {base: '#f97316', _osDark: '#fb923c'}},

          // Measurements table. The speed bars keep a fixed colour each while
          // the ratio bar takes the codec's, which is what ties a row to its
          // series on the charts.
          barTrack: {value: {base: '#f1f5f9', _osDark: '#3a3f4b'}},
          barCompress: {value: {base: '#2563eb', _osDark: '#60a5fa'}},
          barDecompress: {value: {base: '#8b5cf6', _osDark: '#a78bfa'}},

          // Frontier tags. `tagRatioFg` is darker than Figma's #16a34a for the
          // same reason `pg.muted` is: that value reaches only 3.00:1 on the
          // wash behind it, and the tag is 11px. Dark foregrounds are lifted
          // off Figma's too, which were picked against a light background.
          tagRatioBg: {value: {base: '#dcfce7', _osDark: '#14432a'}},
          tagRatioFg: {value: {base: '#15803d', _osDark: '#6ee7a8'}},
          tagBalancedBg: {value: {base: '#fff7ed', _osDark: '#43290f'}},
          tagBalancedFg: {value: {base: '#c2410c', _osDark: '#fdba74'}},
          tagSpeedBg: {value: {base: '#eff6ff', _osDark: '#1e3a5f'}},
          tagSpeedFg: {value: {base: '#1d4ed8', _osDark: '#93c5fd'}},
        },
      },
    },
    tokens: {
      fonts: {
        body: {
          value: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
        },
        heading: {
          value: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
        },
      },
    },
  },
});

export const playgroundSystem = createSystem(defaultConfig, playgroundConfig);

/**
 * The one place a codec's colour is named. The table, the charts and the
 * legend all read it from here, which is what keeps a row and its series the
 * same colour in both schemes.
 */
export const CODEC_COLOR: Record<CompressorName, string> = {
  OpenZL: 'pg.openzl',
  zstd: 'pg.zstd',
  gzip: 'pg.gzip',
};
