// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import {describe, it, expect, afterEach, vi} from 'vitest';
import {createEvent, fireEvent, render, screen, cleanup, within, type RenderOptions} from '@testing-library/react';
import {ChakraProvider} from '@chakra-ui/react';
import React from 'react';
import UploadCard from '../src/components/UploadCard.tsx';
import {playgroundSystem} from '../src/theme.ts';

function renderWithPlaygroundTheme(ui: React.ReactElement, options?: Omit<RenderOptions, 'wrapper'>) {
  return render(ui, {
    wrapper: ({children}) => <ChakraProvider value={playgroundSystem}>{children}</ChakraProvider>,
    ...options,
  });
}

const textFile = (name: string, bytes: number) =>
  new File(['x'.repeat(bytes)], name, {type: 'application/octet-stream'});

describe('UploadCard', () => {
  afterEach(() => {
    cleanup();
  });

  it('prompts for a file when none is selected', () => {
    renderWithPlaygroundTheme(<UploadCard file={null} onFileChange={vi.fn()} />);

    expect(screen.getByText('Drag & drop a file here')).toBeInTheDocument();
    expect(screen.getByLabelText(/browse your computer/)).toHaveAttribute('type', 'file');
  });

  it('renders the selected file from props', () => {
    renderWithPlaygroundTheme(<UploadCard file={textFile('sample.bin', 5_000_000)} onFileChange={vi.fn()} />);

    // Scoped to the drop zone: the live region repeats the name and size.
    const dropzone = within(screen.getByTestId('upload-dropzone'));
    expect(dropzone.getByText('sample.bin')).toBeInTheDocument();
    expect(dropzone.getByText(/5 MB/)).toBeInTheDocument();
    expect(dropzone.getByText('Choose another file')).toBeInTheDocument();
  });

  it('reports a file chosen through the picker without storing it', () => {
    const onFileChange = vi.fn();
    renderWithPlaygroundTheme(<UploadCard file={null} onFileChange={onFileChange} />);

    const picked = textFile('picked.bin', 2048);
    fireEvent.change(screen.getByLabelText(/browse your computer/), {target: {files: [picked]}});

    expect(onFileChange).toHaveBeenCalledWith(picked);
    // Controlled: the parent did not update props, so nothing renders.
    expect(screen.queryByText('picked.bin')).not.toBeInTheDocument();
  });

  it('reports a dropped file', () => {
    const onFileChange = vi.fn();
    renderWithPlaygroundTheme(<UploadCard file={null} onFileChange={onFileChange} />);

    const dropped = textFile('dropped.bin', 9);
    fireEvent.drop(screen.getByTestId('upload-dropzone'), {dataTransfer: {files: [dropped]}});

    expect(onFileChange).toHaveBeenCalledWith(dropped);
  });

  it('accepts a drop released over the remove button', () => {
    const onFileChange = vi.fn();
    renderWithPlaygroundTheme(<UploadCard file={textFile('old.bin', 9)} onFileChange={onFileChange} />);

    const replacement = textFile('new.bin', 9);
    fireEvent.drop(screen.getByRole('button', {name: 'Remove old.bin'}), {
      dataTransfer: {files: [replacement]},
    });

    expect(onFileChange).toHaveBeenCalledWith(replacement);
  });

  it('ignores a dropped directory', () => {
    const onFileChange = vi.fn();
    renderWithPlaygroundTheme(<UploadCard file={null} onFileChange={onFileChange} />);

    fireEvent.drop(screen.getByTestId('upload-dropzone'), {
      dataTransfer: {
        items: [{webkitGetAsEntry: () => ({isFile: false})}],
        files: [textFile('a-folder', 0)],
      },
    });

    expect(onFileChange).not.toHaveBeenCalled();
  });

  it('keeps the drag highlight while the pointer moves onto a child', () => {
    renderWithPlaygroundTheme(<UploadCard file={null} onFileChange={vi.fn()} />);
    const dropzone = screen.getByTestId('upload-dropzone');
    const label = screen.getByLabelText(/browse your computer/).closest('label') as HTMLElement;

    // relatedTarget is an accessor on MouseEvent, so fireEvent's init object
    // cannot set it; define it on the event before dispatching.
    const leaveTowards = (target: Node) => {
      const event = createEvent.dragLeave(dropzone);
      Object.defineProperty(event, 'relatedTarget', {value: target});
      fireEvent(dropzone, event);
    };

    fireEvent.dragEnter(dropzone);
    expect(dropzone).toHaveAttribute('data-dragging', 'true');

    leaveTowards(label);
    expect(dropzone).toHaveAttribute('data-dragging', 'true');

    leaveTowards(document.body);
    expect(dropzone).not.toHaveAttribute('data-dragging');
  });

  it('takes no file while disabled, by the picker, a drop or the remove button', () => {
    const onFileChange = vi.fn();
    renderWithPlaygroundTheme(<UploadCard file={textFile('data.bin', 1)} onFileChange={onFileChange} disabled />);

    expect(screen.getByLabelText(/Choose another file/)).toBeDisabled();
    expect(screen.getByRole('button', {name: 'Remove data.bin'})).toBeDisabled();

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.dragEnter(dropzone);
    expect(dropzone).not.toHaveAttribute('data-dragging');
    fireEvent.drop(dropzone, {dataTransfer: {files: [textFile('dropped.bin', 9)]}});
    expect(onFileChange).not.toHaveBeenCalled();
  });

  it('reports removal through onFileChange', () => {
    const onFileChange = vi.fn();
    renderWithPlaygroundTheme(<UploadCard file={textFile('data.bin', 1)} onFileChange={onFileChange} />);

    fireEvent.click(screen.getByRole('button', {name: 'Remove data.bin'}));

    expect(onFileChange).toHaveBeenCalledWith(null);
  });
});
