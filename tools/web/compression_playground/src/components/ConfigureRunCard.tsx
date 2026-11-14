// Copyright (c) Meta Platforms, Inc. and affiliates.

import {Fragment, useEffect, useMemo, useState} from 'react';
import {getBenchmarkProfiles} from '../runBenchmark.ts';
import {
  Box,
  Button,
  Checkbox,
  IconButton,
  NativeSelect,
  Popover,
  Portal,
  Select,
  Slider,
  Text,
  Tooltip,
  VStack,
  chakra,
  createListCollection,
  useSlotRecipe,
} from '@chakra-ui/react';
import {LuChevronDown, LuChevronRight, LuInfo, LuPlus, LuX} from 'react-icons/lu';
import StepCard from './StepCard.tsx';
import {
  COMPRESSOR_NAMES,
  ITERATION_TICKS,
  ITERATIONS_DEFAULT,
  ITERATIONS_MAX,
  ITERATIONS_MIN,
  ITERATIONS_THUMB_HALF_PX,
  OPENZL_DEFAULT_LEVEL,
  OPENZL_LEVELS,
  getOpenZlProfiles,
  TRAINED_CANDIDATE_COUNTS,
  isTrainableProfile,
  levelsFor,
  tickLabelLeft,
  type CompressorName,
  type CompressorRow,
  type GzipRow,
  type OpenZlProfile,
  type OpenZlRow,
  type ZstdRow,
} from '../compressors.ts';
import type {CompressorRows, PatchRow} from '../useCompressorRows.ts';
import {isBrowserSupportedProfile, profileDescription} from '../wasmProfiles.ts';

interface CompressorRowProps {
  row: CompressorRow;
  currentRows: readonly CompressorRow[];
  position: number;
  canRemove: boolean;
  onPatch: PatchRow;
  onCompressorChange: (id: number, compressor: CompressorName) => void;
  onRemove: (id: number) => void;
  candidateCounts: readonly number[];
  unavailableHeading: string;
}

interface OpenZlRowProps {
  row: OpenZlRow;
  position: number;
  onPatch: PatchRow;
}

interface ConfigureRunCardProps {
  onProfilesReady?: (ready: boolean) => void;
  compressors: CompressorRows;
  iterations: number;
  onIterationsChange: (iterations: number) => void;
  /** The candidate counts a trained row can pick; the browser trainer's own window unless the backend says otherwise. */
  candidateCounts?: readonly number[];
  /** What the picker says of a profile this build cannot run. */
  unavailableHeading?: string;
  /** Where the profiles come from: the WASM worker unless the desktop app supplies its own. */
  loadProfiles?: () => Promise<unknown>;
}

function OpenZlOptions({
  row,
  position,
  onPatch,
  candidateCounts,
}: OpenZlRowProps & {candidateCounts: readonly number[]}) {
  const helperTooltip = useSlotRecipe({key: 'helperTooltip'})();
  // A backstop for rows that arrive from elsewhere: the picker already refuses
  // pytorch, which the browser cannot run at all.
  const canTrain = isTrainableProfile(row.profile);
  const isTraining = row.trainRequested && canTrain;
  return (
    <Box borderRadius="6px" overflow="hidden">
      <chakra.button
        type="button"
        display="flex"
        gap="6px"
        alignItems="center"
        width="100%"
        px="10px"
        py="8px"
        bg="transparent"
        borderWidth="0"
        fontFamily="inherit"
        textAlign="left"
        cursor="pointer"
        aria-expanded={row.optionsOpen}
        aria-label={`OpenZL options for row ${position}`}
        onClick={() => onPatch(row.id, row.compressor, {optionsOpen: !row.optionsOpen})}>
        <Box aria-hidden="true" color="pg.secondary">
          {row.optionsOpen ? <LuChevronDown size={10} /> : <LuChevronRight size={10} />}
        </Box>
        <Text color="pg.secondary" fontSize="13px" fontWeight="medium" lineHeight="1" m={0}>
          OpenZL options
        </Text>
      </chakra.button>

      {row.optionsOpen && (
        <Box px="12px" py="10px">
          <VStack gap="10px" align="stretch">
            <Box display="flex" gap="8px" alignItems="center">
              <Text color="pg.faint" fontSize="11px" fontWeight="semibold" m={0}>
                LEVEL
              </Text>
              <Tooltip.Root openDelay={200} disabled={!isTraining} positioning={{placement: 'top', gutter: 8}}>
                <Tooltip.Trigger asChild>
                  <NativeSelect.Root
                    role="group"
                    aria-label={`OpenZL level for row ${position}`}
                    tabIndex={isTraining ? 0 : -1}
                    width="120px"
                    flexShrink={0}
                    disabled={isTraining}>
                    <NativeSelect.Field
                      aria-label={`OpenZL level for row ${position}`}
                      height="32px"
                      fontSize="13px"
                      bg="pg.surface"
                      borderColor="pg.border"
                      borderRadius="6px"
                      _disabled={{pointerEvents: 'none'}}
                      value={row.level}
                      onChange={(event) => onPatch(row.id, row.compressor, {level: Number(event.target.value)})}>
                      {OPENZL_LEVELS.map((level) => (
                        <option key={level} value={level}>
                          {level === OPENZL_DEFAULT_LEVEL ? `${level} (default)` : level}
                        </option>
                      ))}
                    </NativeSelect.Field>
                    <NativeSelect.Indicator />
                  </NativeSelect.Root>
                </Tooltip.Trigger>
                <Portal>
                  <Tooltip.Positioner>
                    <Tooltip.Content css={helperTooltip.content}>
                      <Tooltip.Arrow>
                        <Tooltip.ArrowTip css={helperTooltip.arrowTip} />
                      </Tooltip.Arrow>
                      Compression level has no effect while training is on.
                    </Tooltip.Content>
                  </Tooltip.Positioner>
                </Portal>
              </Tooltip.Root>
            </Box>

            <Box display="flex" alignItems="center" justifyContent="space-between" gap="8px">
              <Checkbox.Root
                checked={row.trainRequested && canTrain}
                disabled={!canTrain}
                onCheckedChange={(details) =>
                  onPatch(row.id, row.compressor, {trainRequested: details.checked === true})
                }>
                <Checkbox.HiddenInput />
                <Checkbox.Control
                  boxSize="16px"
                  borderRadius="4px"
                  bg="pg.surface"
                  borderColor="pg.border"
                  _checked={{bg: 'pg.accent', borderColor: 'pg.accent', color: 'white'}}
                />
                <Checkbox.Label color="pg.faint" fontSize="11px" fontWeight="semibold">
                  TRAIN
                </Checkbox.Label>
              </Checkbox.Root>

              <Box display="flex" alignItems="center" gap="8px">
                <Text color="pg.faint" fontSize="11px" fontWeight="semibold" textAlign="right" m={0}>
                  NUMBER OF TRAINED CANDIDATES
                </Text>
                <NativeSelect.Root width="72px" flexShrink={0} disabled={!row.trainRequested || !canTrain}>
                  <NativeSelect.Field
                    aria-label={`Number of trained candidates for row ${position}`}
                    height="28px"
                    fontSize="12px"
                    bg="pg.surface"
                    borderColor="pg.border"
                    borderRadius="6px"
                    value={row.candidates}
                    onChange={(event) => onPatch(row.id, row.compressor, {candidates: Number(event.target.value)})}>
                    {candidateCounts.map((count) => (
                      <option key={count} value={count}>
                        {count}
                      </option>
                    ))}
                  </NativeSelect.Field>
                  <NativeSelect.Indicator />
                </NativeSelect.Root>
              </Box>
            </Box>

            <Box
              display="flex"
              gap="8px"
              alignItems="center"
              bg="pg.accentBg"
              borderWidth="1px"
              borderColor="pg.infoBorder"
              borderRadius="6px"
              px="10px"
              py="8px">
              <Box aria-hidden="true" color="pg.accent" flexShrink={0}>
                <LuInfo size={14} />
              </Box>
              <Text color="pg.secondary" fontSize="11px" lineHeight="1.4" m={0}>
                {canTrain
                  ? 'Training builds a compression graph optimized for your data, using the selected profile as a starting point. This can take a while.'
                  : `Training is not supported for the ${row.profile} profile`}
              </Text>
            </Box>
          </VStack>
        </Box>
      )}
    </Box>
  );
}

/**
 * Ids for the tool tip text description. Scoped per row because ids must be unique
 * and every row renders its own listbox.
 */
function profileDescriptionId(rowId: number, profile: string): string {
  return `row-${rowId}-profile-${profile}-description`;
}

function ProfileSelect({row, position, onPatch, unavailableHeading}: OpenZlRowProps & {unavailableHeading: string}) {
  const helperTooltip = useSlotRecipe({key: 'helperTooltip'})();
  const triggerId = `row-${row.id}-profile-trigger`;
  const profiles = getOpenZlProfiles();
  /**
   * Runnable profiles first, the rest after, alphabetical within each group.
   * Both groups are derived from `isBrowserSupportedProfile`, so new profiles
   * sort themselves -- there is no second list to keep in step.
   *
   * Ordering rather than the CLI's own: most of the list cannot run here, and
   * `be-*` sorts to the front alphabetically, so the plain order opened on seven
   * greyed-out rows before the first selectable one.
   */
  const {runnableProfiles, profileCollection} = useMemo(() => {
    const runnableProfiles = profiles.filter(isBrowserSupportedProfile).sort();
    const unrunnableProfiles = profiles.filter((profile) => !isBrowserSupportedProfile(profile)).sort();

    const profileCollection = createListCollection<OpenZlProfile>({
      items: [...runnableProfiles, ...unrunnableProfiles],
      // Listed but unselectable, so the picker cannot produce a run the worker
      // would have to drop on the floor.
      isItemDisabled: (profile) => !isBrowserSupportedProfile(profile),
    });
    return {runnableProfiles, profileCollection};
  }, [profiles]);

  return (
    <Select.Root
      ids={{trigger: triggerId}}
      disabled={profiles.length === 0}
      collection={profileCollection}
      size="sm"
      width="140px"
      flexShrink={0}
      // Chakra defaults this to `sameWidth`, which pins the listbox to the
      // trigger's 140px: the group heading wraps and
      // `numeric-ml-selector-64` breaks mid-word. Off, the popper sizes it to
      // `max-content`, so the widest row decides and nothing has to be pinned
      // to the trigger's width by hand.
      positioning={{sameWidth: false}}
      value={[row.profile]}
      onValueChange={(details) => {
        const [next] = details.value;
        if (next != null) {
          // buildJobs checks training capability, preserve trainRequested so
          // switching back to a trainable profile restores the user's setting.
          onPatch(row.id, row.compressor, {profile: next});
        }
      }}>
      <Select.Label srOnly>Profile for row {position}</Select.Label>
      {/* The indicator group must sit beside the trigger inside the control:
          it is absolutely positioned and the control is the only positioned
          ancestor — nested in the trigger, the arrow flies off-box. */}
      <Tooltip.Root ids={{trigger: triggerId}} openDelay={200} positioning={{placement: 'top', gutter: 8}}>
        <Select.Control>
          <Tooltip.Trigger asChild>
            <Select.Trigger bg="pg.surface" borderColor="pg.border" borderRadius="6px" px="12px" color="pg.ink">
              <Select.ValueText />
            </Select.Trigger>
          </Tooltip.Trigger>
          <Select.IndicatorGroup>
            <Select.Indicator />
          </Select.IndicatorGroup>
        </Select.Control>
        <Portal>
          <Tooltip.Positioner>
            <Tooltip.Content css={helperTooltip.content}>
              <Tooltip.Arrow>
                <Tooltip.ArrowTip css={helperTooltip.arrowTip} />
              </Tooltip.Arrow>
              Match the profile to your data so OpenZL can use its structure for better compression.
            </Tooltip.Content>
          </Tooltip.Positioner>
        </Portal>
      </Tooltip.Root>
      <Select.Positioner>
        <Select.Content color="pg.ink">
          {profileCollection.items.map((profile, index) => (
            <Fragment key={profile}>
              {/* One heading, before the first profile that cannot run. Without
                  it the greyed half reads as broken rather than unavailable,
                  and the reason is only reachable by hovering each one. */}
              {index === runnableProfiles.length && (
                <Box
                  px="8px"
                  pt="8px"
                  pb="4px"
                  mt="4px"
                  borderTopWidth="1px"
                  borderColor="pg.border"
                  color="pg.faint"
                  fontSize="11px"
                  fontWeight="semibold"
                  textTransform="uppercase">
                  {unavailableHeading}
                </Box>
              )}
              <Select.Item
                item={profile}
                // zag drives this listbox with aria-activedescendant, so options
                // never take DOM focus and the tooltip below can only ever fire
                // on hover. This is what carries the description to everyone else.
                aria-describedby={profileDescriptionId(row.id, profile)}>
                <Tooltip.Root openDelay={200} positioning={{placement: 'right', gutter: 8}}>
                  <Tooltip.Trigger asChild>
                    <Select.ItemText>{profile}</Select.ItemText>
                  </Tooltip.Trigger>
                  {/* Portalled out: the listbox scrolls, so inline content would
                    be clipped by its overflow. */}
                  <Portal>
                    <Tooltip.Positioner>
                      <Tooltip.Content css={helperTooltip.content}>
                        <Tooltip.Arrow>
                          <Tooltip.ArrowTip css={helperTooltip.arrowTip} />
                        </Tooltip.Arrow>
                        {profileDescription(profile)}
                      </Tooltip.Content>
                    </Tooltip.Positioner>
                  </Portal>
                </Tooltip.Root>
                <Select.ItemIndicator />
              </Select.Item>
            </Fragment>
          ))}
        </Select.Content>
      </Select.Positioner>
      {/* Outside the listbox on purpose: nested in an option these would join
          its accessible name, so "csv" would read as the whole help sentence. */}
      <Box srOnly>
        {profileCollection.items.map((profile) => (
          <Box as="span" key={profile} id={profileDescriptionId(row.id, profile)}>
            {profileDescription(profile)}
          </Box>
        ))}
      </Box>
    </Select.Root>
  );
}

/** The id of the message below, so the trigger can point at what is wrong. */
function levelsErrorId(rowId: number): string {
  return `row-${rowId}-levels-error`;
}

/**
 * The levels a zstd or gzip row will be measured at. A multi-select rather than
 * a dropdown because each level is one point on the charts, so a row is a curve
 * rather than a value.
 *
 * Clearing every level is allowed and flagged rather than prevented: a picker
 * whose `None` leaves one box ticked reads as broken, and the row has to be
 * emptied before a single level can be chosen without unticking five first.
 */
function LevelsPicker({row, position, onPatch}: {row: ZstdRow | GzipRow; position: number; onPatch: PatchRow}) {
  const helperTooltip = useSlotRecipe({key: 'helperTooltip'})();
  const all = levelsFor(row.compressor);
  const selected = new Set(row.levels);
  const empty = row.levels.length === 0;
  const levelsLabel = empty ? 'No levels' : row.levels.join(', ');
  const setLevels = (levels: readonly number[]) => {
    onPatch(row.id, row.compressor, {levels: [...levels].sort((a, b) => a - b)});
  };

  return (
    <Tooltip.Root openDelay={200} disabled={empty} positioning={{placement: 'top', gutter: 8}}>
      <Popover.Root positioning={{placement: 'bottom-end'}}>
        <Tooltip.Trigger asChild>
          <Box as="span" display="inline-flex" flexShrink={0}>
            <Popover.Trigger asChild>
              <Button
                // The levels lead so the accessible name contains the visible text:
                // a voice-control user says what they can see. A static label would
                // override it and leave them with nothing to say.
                aria-label={`${levelsLabel}, row ${position}`}
                aria-invalid={empty || undefined}
                aria-describedby={empty ? levelsErrorId(row.id) : undefined}
                width="140px"
                flexShrink={0}
                height="36px"
                px="12px"
                justifyContent="space-between"
                fontSize="14px"
                fontWeight="normal"
                color={empty ? 'pg.danger' : 'pg.ink'}
                bg="pg.surface"
                borderWidth="1px"
                borderColor={empty ? 'pg.danger' : 'pg.border'}
                borderRadius="6px"
                _hover={{bg: 'pg.surface'}}>
                <Text as="span" flex="1" minW={0} truncate textAlign="left">
                  {levelsLabel}
                </Text>
                <Box as="span" aria-hidden="true" display="inline-flex" flexShrink={0}>
                  <LuChevronDown size={12} />
                </Box>
              </Button>
            </Popover.Trigger>
          </Box>
        </Tooltip.Trigger>
        <Portal>
          <Popover.Positioner>
            <Popover.Content width="220px" maxH="260px" overflowY="auto">
              <Box
                display="flex"
                alignItems="center"
                justifyContent="space-between"
                px="12px"
                py="10px"
                borderBottomWidth="1px"
                borderColor="pg.border">
                <Text fontSize="11px" fontWeight="bold" color="pg.secondary" textTransform="uppercase" m={0}>
                  {row.compressor} levels
                </Text>
                <Box display="flex" gap="8px">
                  <chakra.button
                    type="button"
                    fontSize="12px"
                    fontWeight="medium"
                    color="pg.accent"
                    onClick={() => setLevels(all)}>
                    All
                  </chakra.button>
                  <chakra.button
                    type="button"
                    fontSize="12px"
                    fontWeight="medium"
                    color="pg.accent"
                    onClick={() => setLevels([])}>
                    None
                  </chakra.button>
                </Box>
              </Box>
              {/* A column, because `Checkbox.Root` is inline-flex: as direct
                children of the content they stretched, and inside a plain box
                they would sit two to a line. */}
              <Box role="group" aria-label={`${row.compressor} levels`} display="flex" flexDirection="column">
                {all.map((level) => {
                  const checked = selected.has(level);
                  return (
                    <Checkbox.Root
                      key={level}
                      px="12px"
                      py="8px"
                      checked={checked}
                      onCheckedChange={() =>
                        setLevels(checked ? row.levels.filter((l) => l !== level) : [...row.levels, level])
                      }>
                      <Checkbox.HiddenInput />
                      <Checkbox.Control />
                      <Checkbox.Label fontSize="13px" fontWeight="normal" color="pg.ink">
                        Level {level}
                      </Checkbox.Label>
                    </Checkbox.Root>
                  );
                })}
              </Box>
            </Popover.Content>
          </Popover.Positioner>
        </Portal>
      </Popover.Root>
      <Portal>
        <Tooltip.Positioner>
          <Tooltip.Content css={helperTooltip.content}>
            <Tooltip.Arrow>
              <Tooltip.ArrowTip css={helperTooltip.arrowTip} />
            </Tooltip.Arrow>
            {levelsLabel}
          </Tooltip.Content>
        </Tooltip.Positioner>
      </Portal>
    </Tooltip.Root>
  );
}

function CompressorRowCard({
  row,
  currentRows,
  position,
  canRemove,
  onPatch,
  onCompressorChange,
  onRemove,
  candidateCounts,
  unavailableHeading,
}: CompressorRowProps) {
  const helperTooltip = useSlotRecipe({key: 'helperTooltip'})();
  const triggerId = `row-${row.id}-compressor`;
  const collection = createListCollection({
    items: COMPRESSOR_NAMES.map((name) => ({
      label: name,
      value: name,
      disabled: name !== 'OpenZL' && currentRows.some((other) => other.id !== row.id && other.compressor === name),
      description: `${name} is already selected. Select multiple levels in its existing row.`,
      descriptionId: `row-${row.id}-compressor-${name}-description`,
    })),
    isItemDisabled: (item) => item.disabled,
  });
  return (
    <Box
      bg="pg.canvas"
      borderWidth="1px"
      borderColor="pg.border"
      borderRadius="8px"
      px="12px"
      pt="12px"
      pb="8px"
      display="flex"
      flexDirection="column"
      gap="4px">
      <Box display="flex" gap="8px" alignItems="end">
        <VStack flex="1" minW={0} gap="6px" align="stretch">
          <chakra.label htmlFor={triggerId} color="pg.faint" fontSize="11px" fontWeight="semibold" m={0}>
            COMPRESSOR
          </chakra.label>
          <Select.Root
            ids={{trigger: triggerId}}
            collection={collection}
            size="sm"
            value={[row.compressor]}
            onValueChange={({value}) => {
              const [name] = value;
              if (name !== undefined) {
                onCompressorChange(row.id, name as CompressorName);
              }
            }}>
            <Select.Label srOnly>Compressor for row {position}</Select.Label>
            <Select.Control>
              <Select.Trigger
                height="36px"
                fontSize="14px"
                color="pg.ink"
                bg="pg.surface"
                borderColor="pg.border"
                borderRadius="6px">
                <Select.ValueText />
              </Select.Trigger>
              <Select.IndicatorGroup>
                <Select.Indicator />
              </Select.IndicatorGroup>
            </Select.Control>
            <Select.Positioner>
              <Select.Content color="pg.ink" bg="pg.surface">
                {collection.items.map((item) => (
                  <Select.Item
                    key={item.value}
                    item={item}
                    _highlighted={{bg: 'pg.chip'}}
                    _disabled={{pointerEvents: 'auto'}}
                    aria-describedby={item.disabled ? item.descriptionId : undefined}>
                    <Tooltip.Root
                      disabled={!item.disabled}
                      openDelay={200}
                      positioning={{placement: 'right', gutter: 8}}>
                      <Tooltip.Trigger asChild>
                        <Select.ItemText>{item.label}</Select.ItemText>
                      </Tooltip.Trigger>
                      <Portal>
                        <Tooltip.Positioner>
                          <Tooltip.Content css={helperTooltip.content}>
                            <Tooltip.Arrow>
                              <Tooltip.ArrowTip css={helperTooltip.arrowTip} />
                            </Tooltip.Arrow>
                            {item.description}
                          </Tooltip.Content>
                        </Tooltip.Positioner>
                      </Portal>
                    </Tooltip.Root>
                    <Select.ItemIndicator />
                  </Select.Item>
                ))}
              </Select.Content>
            </Select.Positioner>
            <Box srOnly>
              {collection.items
                .filter((item) => item.disabled)
                .map((item) => (
                  <Box as="span" key={item.value} id={item.descriptionId}>
                    {item.description}
                  </Box>
                ))}
            </Box>
          </Select.Root>
        </VStack>

        <VStack gap="6px" align="stretch" flexShrink={0}>
          <Text color="pg.faint" fontSize="11px" fontWeight="semibold" m={0}>
            {row.compressor === 'OpenZL' ? 'PROFILE' : 'LEVEL'}
          </Text>
          {row.compressor === 'OpenZL' ? (
            <ProfileSelect row={row} position={position} onPatch={onPatch} unavailableHeading={unavailableHeading} />
          ) : (
            <LevelsPicker row={row} position={position} onPatch={onPatch} />
          )}
        </VStack>

        <IconButton
          aria-label={`Remove row ${position}`}
          variant="ghost"
          size="2xs"
          flexShrink={0}
          mb="6px"
          color="pg.muted"
          _hover={{color: 'pg.accent'}}
          disabled={!canRemove}
          onClick={() => onRemove(row.id)}>
          <LuX aria-hidden="true" />
        </IconButton>
      </Box>

      {row.compressor !== 'OpenZL' && row.levels.length === 0 && (
        <Text id={levelsErrorId(row.id)} color="pg.danger" fontSize="11px" m={0}>
          Pick at least one level to measure
        </Text>
      )}

      {row.compressor === 'OpenZL' && (
        <OpenZlOptions row={row} position={position} onPatch={onPatch} candidateCounts={candidateCounts} />
      )}
    </Box>
  );
}

export default function ConfigureRunCard({
  compressors: {rows, addRow, removeRow, patchRow, changeCompressor},
  iterations,
  onIterationsChange,
  onProfilesReady,
  candidateCounts = TRAINED_CANDIDATE_COUNTS,
  unavailableHeading = 'Not available in the browser',
  loadProfiles = getBenchmarkProfiles,
}: ConfigureRunCardProps) {
  const [profileStatus, setProfileStatus] = useState<'loading' | 'ready' | Error>('loading');
  useEffect(() => {
    let active = true;
    loadProfiles()
      .then(() => {
        if (active) {
          setProfileStatus('ready');
          onProfilesReady?.(true);
        }
      })
      .catch((error: unknown) => {
        if (active) setProfileStatus(error instanceof Error ? error : new Error(String(error)));
      });
    return () => {
      active = false;
    };
  }, [loadProfiles, onProfilesReady]);
  return (
    <StepCard number={2} title="Configure the run" subtitle="Select which compressors to compare">
      <VStack gap="10px" align="stretch" aria-busy={profileStatus === 'loading'}>
        {profileStatus instanceof Error && (
          <Text role="alert">Unable to load compression profiles: {profileStatus.message}</Text>
        )}
        {rows.map((row, index) => (
          <CompressorRowCard
            key={row.id}
            row={row}
            currentRows={rows}
            position={index + 1}
            canRemove={rows.length > 1}
            onPatch={patchRow}
            onCompressorChange={changeCompressor}
            onRemove={removeRow}
            candidateCounts={candidateCounts}
            unavailableHeading={unavailableHeading}
          />
        ))}
      </VStack>

      <Box>
        <Button
          type="button"
          variant="outline"
          size="sm"
          bg="pg.surface"
          borderColor="pg.border"
          color="pg.secondary"
          fontSize="13px"
          fontWeight="semibold"
          borderRadius="6px"
          _hover={{bg: 'pg.canvas'}}
          onClick={addRow}>
          <LuPlus aria-hidden="true" />
          Add compressor
        </Button>
      </Box>

      <Box height="1px" bg="pg.border" />

      <VStack gap="10px" align="stretch">
        <Box display="flex" gap="8px" alignItems="baseline" flexWrap="wrap">
          <Text color="pg.ink" fontSize="11px" fontWeight="bold" m={0}>
            ITERATIONS
          </Text>
          <Text color="pg.muted" fontSize="12px" m={0}>
            repeat each measurement, report the mean
          </Text>
          <Box flex="1" />
          <Text color="pg.ink" fontSize="12px" fontWeight="bold" data-testid="iterations-value" m={0}>
            {iterations}
          </Text>
        </Box>
        <VStack gap="12px" align="stretch">
          <Slider.Root
            min={ITERATIONS_MIN}
            max={ITERATIONS_MAX}
            step={1}
            value={[iterations]}
            onValueChange={(details) => onIterationsChange(details.value[0] ?? ITERATIONS_DEFAULT)}
            width="100%">
            {/* The thumb is named through this label: zag always sets
                aria-labelledby to the label id, which shadows aria-label. */}
            <Slider.Label srOnly>Iterations</Slider.Label>
            {/* The track and thumb must live inside the control: the root is a
                column flex, so a track placed directly in it collapses — its
                flex-basis of 0% overrides the explicit height. */}
            <Slider.Control data-testid="iterations-control">
              <Slider.Track bg="pg.border" height="4px" borderRadius="2px">
                <Slider.Range bg="pg.accent" />
              </Slider.Track>
              <Slider.Thumb
                index={0}
                bg="pg.accent"
                borderColor="pg.accent"
                boxSize={`${ITERATIONS_THUMB_HALF_PX * 2}px`}
                borderRadius={`${ITERATIONS_THUMB_HALF_PX}px`}
                shadow="sm">
                <Slider.HiddenInput />
              </Slider.Thumb>
            </Slider.Control>
          </Slider.Root>
          <Box position="relative" height="16px" data-testid="iteration-ticks">
            {ITERATION_TICKS.map((tick) => (
              <Text
                key={tick}
                position="absolute"
                top={0}
                left={tickLabelLeft(tick)}
                transform="translateX(-50%)"
                color="pg.muted"
                fontSize="11px"
                lineHeight="16px"
                m={0}>
                {tick}
              </Text>
            ))}
          </Box>
        </VStack>
      </VStack>
    </StepCard>
  );
}
