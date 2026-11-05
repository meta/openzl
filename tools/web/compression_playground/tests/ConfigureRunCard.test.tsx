// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import {describe, it, expect, afterEach} from 'vitest';
import {fireEvent, render, screen, cleanup, within, waitFor, type RenderOptions} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {ChakraProvider} from '@chakra-ui/react';
import React, {useState} from 'react';
import ConfigureRunCard from '../src/components/ConfigureRunCard.tsx';
import {
  ITERATIONS_DEFAULT,
  getOpenZlProfiles,
  getOpenZlProfileDescriptions,
  defaultLevelsFor,
  createCompressorRow,
  tickLabelLeft,
  type CompressorName,
  type CompressorRow,
  type OpenZlProfile,
} from '../src/compressors.ts';
import {isBrowserSupportedProfile, profileDescription} from '../src/wasmProfiles.ts';
import {useCompressorRows} from '../src/useCompressorRows.ts';
import {playgroundSystem} from '../src/theme.ts';

function renderWithPlaygroundTheme(ui: React.ReactElement, options?: Omit<RenderOptions, 'wrapper'>) {
  return render(ui, {
    wrapper: ({children}) => <ChakraProvider value={playgroundSystem}>{children}</ChakraProvider>,
    ...options,
  });
}

function ControlledConfigureRunCard({initialRows}: {initialRows?: readonly CompressorRow[]}) {
  // The same hook App drives the card with, so these tests exercise the real
  // row and id handling rather than a stand-in.
  const compressors = useCompressorRows(initialRows);
  const [iterations, setIterations] = useState(ITERATIONS_DEFAULT);
  return (
    <>
      <ConfigureRunCard compressors={compressors} iterations={iterations} onIterationsChange={setIterations} />
      <output data-testid="compressor-ids">{compressors.rows.map((row) => row.id).join(',')}</output>
    </>
  );
}

function renderConfigureRunCard(initialRows?: readonly CompressorRow[]) {
  return renderWithPlaygroundTheme(<ControlledConfigureRunCard initialRows={initialRows} />);
}

/**
 * The trigger's accessible name leads with its visible text ("1, 5, 8"), so
 * it changes as levels are picked. Match the row instead of the whole name.
 */
function levelsTrigger(position: number): HTMLElement {
  return screen.getByRole('button', {name: new RegExp(`, row ${String(position)}$`)});
}

/**
 * Opens a row's levels popover and returns a scope for its own content. Two
 * rows' popovers can be open at once, so the scope comes from the trigger's
 * aria-controls rather than from whichever content mounted first.
 */
async function openLevels(position: number) {
  const trigger = levelsTrigger(position);
  fireEvent.click(trigger);
  const id = trigger.getAttribute('aria-controls') ?? '';
  const content = await waitFor(() => {
    const element = document.getElementById(id);
    expect(element).not.toBeNull();
    return element as HTMLElement;
  });
  return within(content);
}

function optionValues(select: HTMLElement): string[] {
  return within(select)
    .getAllByRole('option')
    .map((option) => (option as HTMLOptionElement).value);
}

async function selectCompressor(position: number, compressor: CompressorName) {
  const trigger = screen.getByRole('combobox', {name: `Compressor for row ${position}`});
  fireEvent.click(trigger);
  const listbox = await screen.findByRole('listbox');
  fireEvent.click(within(listbox).getByRole('option', {name: compressor}));
  await waitFor(() => expect(trigger).toHaveTextContent(compressor));
}

describe('ConfigureRunCard', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders the default OpenZL, zstd and gzip rows', () => {
    renderConfigureRunCard();

    const optionsToggle = screen.getByRole('button', {name: 'OpenZL options for row 1'});
    if (optionsToggle.getAttribute('aria-expanded') === 'false') {
      fireEvent.click(optionsToggle);
    }

    expect(screen.getAllByText('COMPRESSOR')).toHaveLength(3);
    expect(screen.getByText('PROFILE')).toBeInTheDocument();
    // zstd and gzip each have a LEVEL label. The expanded OpenZL options add a third, note
    // that if the OpenZL options panel is collapsed, only two LEVEl labels remain
    expect(screen.getAllByText('LEVEL')).toHaveLength(3);

    expect(screen.getByRole('combobox', {name: 'Compressor for row 1'})).toHaveTextContent('OpenZL');
    expect(screen.getByRole('combobox', {name: 'Profile for row 1'})).toHaveTextContent('serial');
    expect(screen.getByRole('combobox', {name: 'Compressor for row 2'})).toHaveTextContent('zstd');
    expect(levelsTrigger(2)).toHaveTextContent('1, 5, 8, 12, 15, 19');
    expect(screen.getByRole('combobox', {name: 'Compressor for row 3'})).toHaveTextContent('gzip');
    expect(levelsTrigger(3)).toHaveTextContent('1, 3, 4, 6, 7, 9');
  });

  it('keeps the dropdown arrow inside the positioned control', () => {
    renderConfigureRunCard();

    const trigger = screen.getByRole('combobox', {name: 'Profile for row 1'});
    // The chevron must sit beside the trigger, not in it: nested in the
    // trigger it positions against a distant ancestor and vanishes.
    expect(trigger.querySelector('svg')).toBeNull();
    expect(trigger.parentElement?.querySelector('svg')).not.toBeNull();
  });

  it('describes the profile picker on keyboard focus and restores focus after closing it', async () => {
    const user = userEvent.setup();
    renderConfigureRunCard();
    const trigger = screen.getByRole('combobox', {name: 'Profile for row 1'});

    await user.tab();
    await user.tab();
    expect(trigger).toHaveFocus();
    await waitFor(() => {
      expect(trigger).toHaveAccessibleDescription(
        'Match the profile to your data so OpenZL can use its structure for better compression.',
      );
    });

    await user.keyboard('{ArrowDown}');
    const listbox = await screen.findByRole('listbox');
    await waitFor(() => expect(listbox).toHaveFocus());
    await user.keyboard('{Escape}');
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it('returns focus to the compressor trigger after closing or selecting', async () => {
    const user = userEvent.setup();
    renderConfigureRunCard();
    const trigger = screen.getByRole('combobox', {name: 'Compressor for row 2'});

    await user.click(trigger);
    const listbox = await screen.findByRole('listbox');
    await waitFor(() => expect(listbox).toHaveFocus());
    await user.keyboard('{Escape}');
    await waitFor(() => expect(trigger).toHaveFocus());

    await user.click(trigger);
    const reopened = await screen.findByRole('listbox');
    await user.click(within(reopened).getByRole('option', {name: 'OpenZL'}));
    await waitFor(() => {
      expect(trigger).toHaveTextContent('OpenZL');
      expect(trigger).toHaveFocus();
    });
  });

  it('offers every compressor and OpenZL profile', async () => {
    renderConfigureRunCard();

    const compressorTrigger = screen.getByRole('combobox', {name: 'Compressor for row 1'});
    fireEvent.click(compressorTrigger);
    const compressors = await screen.findByRole('listbox');
    expect(
      within(compressors)
        .getAllByRole('option')
        .map((option) => option.textContent),
    ).toEqual(['OpenZL', 'zstd', 'gzip']);
    fireEvent.click(compressorTrigger);

    fireEvent.click(screen.getByRole('combobox', {name: 'Profile for row 1'}));
    const listbox = await screen.findByRole('listbox', {name: 'Profile for row 1'});
    const listed = within(listbox)
      .getAllByRole('option')
      .map((option) => option.textContent);

    // Every profile is offered, but runnable ones first: `be-*` sorts to the
    // front alphabetically, so the CLI's own order opened on seven greyed-out
    // rows before the first selectable one.
    expect([...listed].sort()).toEqual([...getOpenZlProfiles()].sort());
    const firstUnrunnable = listed.findIndex((p) => !isBrowserSupportedProfile(p as OpenZlProfile));
    expect(listed.slice(0, firstUnrunnable).every((p) => isBrowserSupportedProfile(p as OpenZlProfile))).toBe(true);
    expect(listed.slice(firstUnrunnable).some((p) => isBrowserSupportedProfile(p as OpenZlProfile))).toBe(false);
  });

  it.each(['zstd', 'gzip'] as const)(
    'disables an existing %s choice and explains multiple levels on hover',
    async (compressor) => {
      const user = userEvent.setup();
      renderConfigureRunCard();
      await user.click(screen.getByRole('button', {name: 'Add compressor'}));
      const trigger = screen.getByRole('combobox', {name: 'Compressor for row 4'});
      await user.click(trigger);
      const listbox = await screen.findByRole('listbox');
      const option = within(listbox).getByRole('option', {name: compressor});

      expect(option).toHaveAttribute('aria-disabled', 'true');
      expect(within(listbox).getByRole('option', {name: 'OpenZL'})).not.toHaveAttribute('aria-disabled', 'true');
      const description = `${compressor} is already selected. Select multiple levels in its existing row.`;
      const describedBy = option.getAttribute('aria-describedby') ?? '';
      expect(document.getElementById(describedBy)).toHaveTextContent(description);

      await user.hover(within(option).getByText(compressor));
      expect(await screen.findByRole('tooltip')).toHaveTextContent(description);
      await user.click(option);
      expect(trigger).toHaveTextContent('OpenZL');
      expect(trigger).toHaveAttribute('aria-expanded', 'true');
    },
  );

  it.each(['zstd', 'gzip'] as const)(
    'allows selecting %s again after its existing row is removed',
    async (compressor) => {
      renderConfigureRunCard();
      fireEvent.click(screen.getByRole('button', {name: `Remove row ${compressor === 'zstd' ? 2 : 3}`}));
      const trigger = screen.getByRole('combobox', {name: 'Compressor for row 1'});
      fireEvent.click(trigger);
      const listbox = await screen.findByRole('listbox');
      const option = within(listbox).getByRole('option', {name: compressor});
      expect(option).not.toHaveAttribute('aria-disabled', 'true');
      fireEvent.click(option);
      await waitFor(() => expect(trigger).toHaveTextContent(compressor));
      expect(levelsTrigger(1)).toBeEnabled();

      fireEvent.click(trigger);
      const reopened = await screen.findByRole('listbox');
      expect(within(reopened).getByRole('option', {name: compressor})).not.toHaveAttribute('aria-disabled', 'true');
    },
  );

  it('names the levels trigger with the text a voice user can see', async () => {
    // WCAG 2.5.3: a static aria-label would override the selected levels and leave
    // someone saying it with nothing to activate.
    renderConfigureRunCard();

    expect(levelsTrigger(2)).toHaveAccessibleName('1, 5, 8, 12, 15, 19, row 2');
    expect(levelsTrigger(2)).toHaveTextContent('1, 5, 8, 12, 15, 19');

    const levels = await openLevels(2);
    fireEvent.click(levels.getAllByRole('checkbox')[0]);
    await waitFor(() => {
      expect(levelsTrigger(2)).toHaveAccessibleName('5, 8, 12, 15, 19, row 2');
    });
  });

  it('shows the chosen level after clearing and selecting one level', async () => {
    renderConfigureRunCard();

    const levels = await openLevels(3);
    fireEvent.click(levels.getByText('None'));
    await waitFor(() => expect(levelsTrigger(3)).toHaveTextContent('No levels'));
    fireEvent.click(levels.getByRole('checkbox', {name: 'Level 6'}));

    await waitFor(() => expect(levelsTrigger(3)).toHaveTextContent(/^6$/));
    expect(levelsTrigger(3)).toHaveAccessibleName('6, row 3');
    expect(levelsTrigger(3)).not.toHaveAttribute('aria-invalid');
  });

  it.each(['hover', 'focus'])('shows the full selected levels on %s', async (interaction) => {
    renderConfigureRunCard();
    const trigger = levelsTrigger(2);

    if (interaction === 'hover') {
      fireEvent.pointerOver(trigger);
    } else {
      fireEvent.focus(trigger);
    }

    const tip = await screen.findByRole('tooltip');
    expect(tip).toHaveTextContent(/^1, 5, 8, 12, 15, 19$/);
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
  });

  it('returns focus to the level button when the dropdown is dismissed', async () => {
    const user = userEvent.setup();
    renderConfigureRunCard();
    const trigger = levelsTrigger(2);

    await user.click(trigger);
    await waitFor(() => expect(trigger).toHaveAttribute('aria-expanded', 'true'));
    const dialog = await screen.findByRole('dialog');
    await waitFor(() => expect(dialog).toContainElement(document.activeElement as HTMLElement));
    await user.keyboard('{Escape}');

    await waitFor(() => {
      expect(trigger).toHaveAttribute('aria-expanded', 'false');
      expect(trigger).toHaveFocus();
    });
  });

  it('keeps the trigger chevron out of the accessible name', async () => {
    renderConfigureRunCard();
    expect(levelsTrigger(2).querySelector('[aria-hidden="true"] svg')).not.toBeNull();
  });

  it('heads the unavailable profiles so the greyed half is not a mystery', async () => {
    renderConfigureRunCard();

    fireEvent.click(screen.getByRole('combobox', {name: 'Profile for row 1'}));
    const listbox = await screen.findByRole('listbox');
    expect(within(listbox).getByText('Not available in the browser')).toBeInTheDocument();
  });

  it('explains each profile when hovered in the list', async () => {
    renderConfigureRunCard();

    fireEvent.click(screen.getByRole('combobox', {name: 'Profile for row 1'}));
    const listbox = await screen.findByRole('listbox');
    const option = within(listbox).getByRole('option', {name: 'parquet'});

    fireEvent.pointerOver(within(option).getByText('parquet'));
    const tip = await screen.findByText(profileDescription('parquet'));
    expect(tip).toBeInTheDocument();
    // Portalled out of the scrolling listbox, which would otherwise clip it.
    expect(tip.closest('[role="listbox"]')).toBeNull();
  });

  it('ranges levels per compressor with OpenZL defaulting to 6', () => {
    renderConfigureRunCard();

    const openZlLevel = screen.getByRole('combobox', {name: 'OpenZL level for row 1'});
    expect(optionValues(openZlLevel).map(Number)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(openZlLevel).toHaveValue('6');
    expect(within(openZlLevel).getByRole('option', {name: '6 (default)'})).toBeInTheDocument();
  });

  it('offers every level of the compressor in the picker, six of them chosen', async () => {
    renderConfigureRunCard();

    const zstd = (await openLevels(2)).getAllByRole('checkbox');
    expect(zstd).toHaveLength(19);
    expect(zstd.filter((box) => (box as HTMLInputElement).checked)).toHaveLength(6);

    const gzip = (await openLevels(3)).getAllByRole('checkbox');
    expect(gzip).toHaveLength(9);
    expect(gzip.filter((box) => (box as HTMLInputElement).checked)).toHaveLength(6);
  });

  it('clears every level and says the row measures nothing', async () => {
    // `None` that leaves one box ticked reads as broken, so the empty row is
    // allowed and flagged instead: the trigger says so, points at the message
    // with `aria-describedby`, and `buildJobs` rejects the row if a run
    // reaches it.
    renderConfigureRunCard();

    const levels = await openLevels(3);
    fireEvent.click(levels.getByText('None'));

    await waitFor(() => {
      expect(levelsTrigger(3)).toHaveTextContent('No levels');
    });
    expect(levels.getAllByRole('checkbox').filter((box) => (box as HTMLInputElement).checked)).toHaveLength(0);
    expect(levelsTrigger(3)).toHaveAttribute('aria-invalid', 'true');
    expect(levelsTrigger(3)).toHaveAccessibleDescription('Pick at least one level to measure');
  });

  it('groups the level checkboxes under the compressor they belong to', async () => {
    // Without it a screen reader announces "Level 1, checkbox" with none of
    // the context the heading above the list gives everyone else.
    renderConfigureRunCard();

    const levels = await openLevels(3);
    expect(levels.getByRole('group', {name: 'gzip levels'})).toBeInTheDocument();
  });

  it('selects and clears levels from the picker', async () => {
    renderConfigureRunCard();

    const levels = await openLevels(3);

    fireEvent.click(levels.getByText('All'));
    await waitFor(() => {
      expect(levelsTrigger(3)).toHaveTextContent('1, 2, 3, 4, 5, 6, 7, 8, 9');
    });

    // The checkbox machine notifies parents asynchronously, so wait for the
    // selection rather than asserting straight after the click.
    fireEvent.click(levels.getByRole('checkbox', {name: 'Level 1'}));
    await waitFor(() => {
      expect(levelsTrigger(3)).toHaveTextContent('2, 3, 4, 5, 6, 7, 8, 9');
    });
  });

  it('describes every OpenZL profile', () => {
    for (const profile of getOpenZlProfiles()) {
      expect(getOpenZlProfileDescriptions()[profile].length).toBeGreaterThan(0);
    }
  });

  it('reaches every profile description without a pointer', async () => {
    // zag drives the listbox with aria-activedescendant, so options never take
    // focus and the hover tooltip is unreachable by keyboard. Each option
    // points at its description instead.
    renderConfigureRunCard();
    fireEvent.click(screen.getByRole('combobox', {name: 'Profile for row 1'}));
    const listbox = await screen.findByRole('listbox');

    for (const profile of getOpenZlProfiles()) {
      const describedBy = within(listbox).getByRole('option', {name: profile}).getAttribute('aria-describedby');
      expect(describedBy).toBeTruthy();
      expect(document.getElementById(describedBy ?? '')).toHaveTextContent(getOpenZlProfileDescriptions()[profile]);
    }
  });

  it('keeps each option name free of its description', async () => {
    // The description lives outside the listbox: nested in an option it would
    // join the accessible name and "csv" would read as the whole sentence.
    renderConfigureRunCard();
    fireEvent.click(screen.getByRole('combobox', {name: 'Profile for row 1'}));
    const listbox = await screen.findByRole('listbox');

    expect(within(listbox).getByRole('option', {name: 'csv'})).toBeInTheDocument();
  });

  it('selects a profile from the list', async () => {
    renderConfigureRunCard();

    fireEvent.click(screen.getByRole('combobox', {name: 'Profile for row 1'}));
    const listbox = await screen.findByRole('listbox');
    fireEvent.click(within(listbox).getByRole('option', {name: 'le-u32'}));

    await waitFor(() => {
      expect(screen.getByRole('combobox', {name: 'Profile for row 1'})).toHaveTextContent('le-u32');
    });
  });

  it('lists the profiles the browser cannot run but refuses to select them', async () => {
    renderConfigureRunCard();

    fireEvent.click(screen.getByRole('combobox', {name: 'Profile for row 1'}));
    const listbox = await screen.findByRole('listbox');

    for (const profile of getOpenZlProfiles().filter((name) => !isBrowserSupportedProfile(name))) {
      expect(within(listbox).getByRole('option', {name: profile})).toHaveAttribute('aria-disabled', 'true');
    }

    fireEvent.click(within(listbox).getByRole('option', {name: 'parquet'}));

    // Still the default: a profile with no WASM compressor must not become a
    // run the worker would then have to drop.
    expect(screen.getByRole('combobox', {name: 'Profile for row 1'})).toHaveTextContent('serial');
  });

  it('offers only the candidate counts the trainer honours', () => {
    renderConfigureRunCard();

    // Asserted as bounds rather than against the constant: the binding raises
    // anything under 6 and throws over 25, so an option outside that window is
    // a run the user cannot actually get.
    const candidates = screen.getByRole('combobox', {name: 'Number of trained candidates for row 1'});
    const counts = optionValues(candidates).map(Number);
    expect(counts.at(0)).toBe(6);
    expect(counts.at(-1)).toBe(25);
    expect(candidates).toHaveValue('6');
  });

  it('swaps the second dropdown when the compressor changes', async () => {
    renderConfigureRunCard([createCompressorRow(1, 'OpenZL'), createCompressorRow(2, 'zstd')]);

    await selectCompressor(2, 'OpenZL');
    expect(screen.getByRole('combobox', {name: 'Profile for row 2'})).toHaveTextContent('serial');
    // Two OpenZL rows now, each with its own TRAIN checkbox.
    expect(screen.getAllByRole('checkbox', {name: 'TRAIN'})).toHaveLength(2);

    await selectCompressor(1, 'gzip');
    expect(levelsTrigger(1)).toHaveTextContent('1, 3, 4, 6, 7, 9');
    expect(screen.queryByRole('combobox', {name: 'OpenZL level for row 1'})).not.toBeInTheDocument();
  });

  it('resets dependent fields when the compressor changes', async () => {
    renderConfigureRunCard([createCompressorRow(1, 'OpenZL'), createCompressorRow(2, 'zstd')]);

    const levels = await openLevels(2);
    fireEvent.click(levels.getByRole('checkbox', {name: 'Level 19'}));
    await waitFor(() => {
      expect(levelsTrigger(2)).toHaveTextContent('1, 5, 8, 12, 15');
    });

    // gzip has no level 19, so the row falls back to its own default set.
    await selectCompressor(2, 'gzip');
    expect(levelsTrigger(2)).toHaveTextContent('1, 3, 4, 6, 7, 9');
  });

  it('leaves the other rows alone when the trained candidate count changes', async () => {
    // Each compressor is configured on its own, so a pick made on the zstd row
    // has to survive a change on the OpenZL one.
    renderConfigureRunCard();

    const levels = await openLevels(2);
    fireEvent.click(levels.getByRole('checkbox', {name: 'Level 19'}));
    await waitFor(() => {
      expect(levelsTrigger(2)).toHaveTextContent('1, 5, 8, 12, 15');
    });

    fireEvent.change(screen.getByRole('combobox', {name: 'Number of trained candidates for row 1'}), {
      target: {value: '12'},
    });
    expect(screen.getByRole('combobox', {name: 'Number of trained candidates for row 1'})).toHaveValue('12');

    expect(levelsTrigger(2)).toHaveTextContent('1, 5, 8, 12, 15');
    expect(levels.getByRole('checkbox', {name: 'Level 19'})).not.toBeChecked();
    expect(levelsTrigger(3)).toHaveTextContent('1, 3, 4, 6, 7, 9');
  });

  it('adds and removes compressor rows', () => {
    renderConfigureRunCard();

    fireEvent.click(screen.getByRole('button', {name: 'Add compressor'}));
    expect(screen.getByRole('combobox', {name: 'Compressor for row 4'})).toHaveTextContent('OpenZL');
    expect(screen.getByTestId('compressor-ids')).toHaveTextContent(/^1,2,3,4$/);

    fireEvent.click(screen.getByRole('button', {name: 'Remove row 4'}));
    expect(screen.queryByRole('combobox', {name: 'Compressor for row 4'})).not.toBeInTheDocument();
    expect(screen.getByTestId('compressor-ids')).toHaveTextContent(/^1,2,3$/);

    fireEvent.click(screen.getByRole('button', {name: 'Add compressor'}));
    expect(screen.getByRole('combobox', {name: 'Compressor for row 4'})).toHaveTextContent('OpenZL');
    expect(screen.getByTestId('compressor-ids')).toHaveTextContent(/^1,2,3,5$/);
  });

  it('keeps at least one row', () => {
    renderConfigureRunCard();

    fireEvent.click(screen.getByRole('button', {name: 'Remove row 3'}));
    fireEvent.click(screen.getByRole('button', {name: 'Remove row 2'}));
    expect(screen.getByRole('button', {name: 'Remove row 1'})).toBeDisabled();
  });

  it('collapses the OpenZL options section', () => {
    renderConfigureRunCard();
    const toggle = screen.getByRole('button', {name: 'OpenZL options for row 1'});

    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('combobox', {name: 'OpenZL level for row 1'})).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(screen.getByRole('combobox', {name: 'OpenZL level for row 1'})).toBeInTheDocument();
  });

  it('only allows changing the OpenZL level while training is off', async () => {
    const user = userEvent.setup();
    renderConfigureRunCard();
    const train = screen.getByRole('checkbox', {name: 'TRAIN'});
    const level = screen.getByRole('combobox', {name: 'OpenZL level for row 1'});

    // Unable to change level if training is on
    expect(train).toBeChecked();
    expect(level).toBeDisabled();
    expect(levelsTrigger(2)).toBeEnabled();
    expect(levelsTrigger(3)).toBeEnabled();
    await user.selectOptions(level, '3');
    expect(level).toHaveValue('6');

    // able to update level is training is off
    fireEvent.click(train);
    await waitFor(() => expect(level).toBeEnabled());
    await user.selectOptions(level, '3');
    expect(level).toHaveValue('3');

    // toggling on training again does not cause any issues
    fireEvent.click(train);
    await waitFor(() => expect(level).toBeDisabled());
    await user.selectOptions(level, '9');
    expect(level).toHaveValue('3');
    expect(levelsTrigger(2)).toBeEnabled();
    expect(levelsTrigger(3)).toBeEnabled();
  });

  it('disables the candidate count while training is off', async () => {
    renderConfigureRunCard();
    const train = screen.getByRole('checkbox', {name: 'TRAIN'});
    const candidates = screen.getByRole('combobox', {name: 'Number of trained candidates for row 1'});

    expect(train).toBeChecked();
    expect(candidates).toBeEnabled();

    fireEvent.click(train);

    // The state machine notifies parents asynchronously, so wait for the
    // consequence instead of asserting synchronously.
    await waitFor(() => {
      expect(train.closest('label')).toHaveAttribute('data-state', 'unchecked');
      expect(candidates).toBeDisabled();
    });
  });

  it('builds non-OpenZL rows without the OpenZL-only fields', () => {
    // `toEqual` rather than `toMatchObject`: the point is that nothing extra is
    // there. Reading `.trainRequested` off these rows is now a compile error,
    // so this only guards the runtime shape the run seam will consume.
    expect(createCompressorRow(2, 'zstd')).toEqual({id: 2, compressor: 'zstd', levels: defaultLevelsFor('zstd')});
    expect(createCompressorRow(3, 'gzip')).toEqual({id: 3, compressor: 'gzip', levels: defaultLevelsFor('gzip')});
    expect(createCompressorRow(1, 'OpenZL').trainRequested).toBe(true);
  });

  it('locks training off for a pytorch row', () => {
    // Injected rather than picked: the picker refuses pytorch, so this covers
    // the lock as the backstop it now is for rows built elsewhere.
    renderConfigureRunCard([{...createCompressorRow(1, 'OpenZL'), profile: 'pytorch'}]);

    const train = screen.getByRole('checkbox', {name: 'TRAIN'});
    expect(train).toBeDisabled();
    expect(train.closest('label')).toHaveAttribute('data-state', 'unchecked');
    expect(screen.getByRole('combobox', {name: 'OpenZL level for row 1'})).toBeEnabled();
    expect(screen.getByText('Training is not supported for the pytorch profile')).toBeInTheDocument();
  });

  it('pins tick labels to the dot travel with the thumb inset', () => {
    // Endpoints sit one half-thumb in from each edge; midpoints are
    // proportional to their value, matching the contain-alignment travel.
    expect(tickLabelLeft(1)).toBe('calc(8px + 0 * (100% - 16px))');
    expect(tickLabelLeft(5)).toBe('calc(8px + 0.2857 * (100% - 16px))');
    expect(tickLabelLeft(10)).toBe('calc(8px + 0.6429 * (100% - 16px))');
    expect(tickLabelLeft(15)).toBe('calc(8px + 1 * (100% - 16px))');
  });

  it('adjusts the iteration count with the slider', async () => {
    renderConfigureRunCard();
    // The thumb stays visibility-hidden under jsdom, where the size
    // measurement that unhides it never completes, so it can only be queried
    // unnamed; the label association is asserted explicitly instead.
    const slider = screen.getByRole('slider', {hidden: true});
    const label = screen.getByText('Iterations');
    expect(slider).toHaveAttribute('aria-labelledby', label.getAttribute('id'));
    // The track collapses when composed outside the control, so pin the
    // structure, not just the behavior.
    const inControl = within(screen.getByTestId('iterations-control'));
    expect(inControl.getByRole('slider', {hidden: true})).toBe(slider);

    expect(slider).toHaveAttribute('aria-valuenow', '5');
    expect(screen.getByTestId('iterations-value')).toHaveTextContent('5');
    const ticks = within(screen.getByTestId('iteration-ticks'));
    for (const tick of ['1', '5', '10', '15']) {
      const label = ticks.getByText(tick);
      expect(label).toBeInTheDocument();
      // Centered under the dot's position for that value, not spread evenly.
      expect(label).toHaveStyle({position: 'absolute', transform: 'translateX(-50%)'});
    }

    // Keyboard input only reaches the slider once the thumb has focus, the
    // same Tab-then-arrows flow a keyboard user follows.
    fireEvent.focus(slider);
    fireEvent.keyDown(slider, {key: 'ArrowRight'});
    await waitFor(() => {
      expect(slider).toHaveAttribute('aria-valuenow', '6');
      expect(screen.getByTestId('iterations-value')).toHaveTextContent('6');
    });
  });
});
