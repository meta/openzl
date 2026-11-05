// Copyright (c) Meta Platforms, Inc. and affiliates.

import {Box, Heading, Link, Text, VStack} from '@chakra-ui/react';
import {LuExternalLink, LuMonitor, LuTriangleAlert} from 'react-icons/lu';
import type {EngineBlocker} from '../engineSupport.ts';

/**
 * The full table, linked rather than reproduced, because it covers engines and
 * versions this screen has no room for.
 */
const MEMORY64_SUPPORT_URL = 'https://caniuse.com/wf-wasm-memory64';

/**
 * Named with versions rather than as "the latest", because these are the
 * releases that shipped Memory64 and the requirement does not move. "Latest"
 * would be vaguer and no more durable. Edge is Chromium, so it shipped Memory64
 * at the same version number as Chrome.
 *
 * The iPhone and iPad caveat is because Apple makes every iOS and iPadOS browser
 * use WebKit. Chrome and Firefox there fail exactly as Safari does, so without
 * it an iPhone user would install one and land back on this screen.
 */
const SUPPORTED_BROWSERS =
  'Chrome 133+, Edge 133+ and Firefox 134+ can run it, except on iPhone and iPad, where every browser uses Safari’s engine.';

interface UnsupportedBrowserProps {
  /** Which gate failed. See `engineBlocker` for why they read differently. */
  blocker: EngineBlocker;
}

/**
 * Shown in place of the playground when the WASM module cannot instantiate.
 *
 * This replaces the setup and results columns rather than sitting above them:
 * every control in those columns leads to a run, and none of them can do
 * anything here, so leaving them on screen would only offer dead ends.
 */
export default function UnsupportedBrowser({blocker}: UnsupportedBrowserProps) {
  // Missing threads means the isolation worker could not take effect, which
  // is rarely about the browser itself: service workers are commonly off in
  // private windows or when site data is blocked. Naming other browsers would
  // point away from the fixes that usually work, a reload or a normal window.
  const isolationFailed = blocker === 'no-threads';
  return (
    <Box
      as="section"
      // `alert` rather than `status`: this is the whole page content, not an
      // aside, so it should preempt whatever a screen reader is saying.
      role="alert"
      aria-labelledby="unsupported-browser-title"
      bg="pg.surface"
      borderWidth="1px"
      borderColor="pg.border"
      borderRadius="12px"
      p="32px">
      <VStack gap="16px" align="stretch">
        <Box display="flex" alignItems="center" gap="12px">
          <Box aria-hidden="true" display="inline-flex" flexShrink={0} color="pg.ink">
            <LuTriangleAlert size={20} />
          </Box>
          <Heading
            id="unsupported-browser-title"
            as="h2"
            flex="1"
            color="pg.ink"
            fontSize="20px"
            fontWeight="extrabold"
            m={0}>
            {isolationFailed ? 'The playground couldn’t start here' : 'Your browser can’t run the playground'}
          </Heading>
        </Box>

        {isolationFailed ? (
          <Text color="pg.secondary" fontSize="15px" lineHeight="1.5" m={0}>
            The playground runs WebAssembly threads, which browsers only allow on cross-origin isolated pages. This page
            sets that up with a service worker, and it didn&rsquo;t take effect. Service workers are often turned off in
            private windows, or by settings that block site data.
          </Text>
        ) : blocker === 'no-memory64' ? (
          <Text color="pg.secondary" fontSize="15px" lineHeight="1.5" m={0}>
            The playground needs 64-bit WebAssembly memory, which this browser doesn&rsquo;t have. See{' '}
            <Link
              href={MEMORY64_SUPPORT_URL}
              color="pg.accent"
              fontWeight="bold"
              // The docs site is same-origin with the playground, so an
              // off-site link is worth marking as one.
              target="_blank"
              rel="noreferrer">
              which browsers support it
              <LuExternalLink size={13} aria-hidden="true" />
            </Link>
            .
          </Text>
        ) : (
          // No WebAssembly at all. The Memory64 link would be beside the point
          // on a browser this old, and misleading about what is missing.
          <Text color="pg.secondary" fontSize="15px" lineHeight="1.5" m={0}>
            The playground needs WebAssembly, which this browser doesn&rsquo;t have.
          </Text>
        )}

        <Box as="aside" aria-label="What to use instead" bg="pg.callout" borderRadius="12px" p="20px">
          <Box display="flex" alignItems="flex-start" gap="12px">
            <Box aria-hidden="true" display="inline-flex" flexShrink={0} mt="2px" color="pg.ink">
              <LuMonitor size={18} />
            </Box>
            <Text color="pg.ink" fontSize="14px" lineHeight="1.5" opacity={0.9} m={0}>
              {isolationFailed ? 'Try reloading the page, or opening it in a normal window.' : SUPPORTED_BROWSERS}
            </Text>
          </Box>
        </Box>
      </VStack>
    </Box>
  );
}
