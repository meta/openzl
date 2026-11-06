// Copyright (c) Meta Platforms, Inc. and affiliates.

import {useRef, useState} from 'react';
import {Box, IconButton, Input, Text, VStack} from '@chakra-ui/react';
import {LuCloudUpload, LuX} from 'react-icons/lu';
import StepCard from './StepCard.tsx';

const FILE_SIZE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const;

/** Decimal units, not binary ones: a 5,000,000-byte file reads back as 5 MB rather than 4.8. */
function formatFileSize(bytes: number): string {
  let size = bytes;
  let unit = 0;
  while (size >= 1000 && unit < FILE_SIZE_UNITS.length - 1) {
    size /= 1000;
    unit += 1;
  }
  const rounded = unit === 0 || size >= 10 ? Math.round(size) : Math.round(size * 10) / 10;
  return `${rounded} ${FILE_SIZE_UNITS[unit]}`;
}

interface UploadCardProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  /** Set while a run is in flight, so the file cannot change under it. */
  disabled?: boolean;
}

export default function UploadCard({file, onFileChange, disabled = false}: UploadCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // The input holds onto the last file it picked, so a selection changed from
  // anywhere else has to clear it or re-picking that file fires no onChange.
  const selectFile = (next: File | null) => {
    onFileChange(next);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  return (
    <StepCard
      number={1}
      title="Choose your data"
      subtitle="Choose data to compress, files are not uploaded to a server">
      {/* The drop target is this row rather than the label below it, so releasing
          over the remove button still lands a file instead of letting the browser
          navigate away from the page. */}
      <Box
        data-testid="upload-dropzone"
        // Exposed so the drag highlight, which is otherwise only themed
        // background and border colours, can be asserted in tests.
        data-dragging={isDragging ? 'true' : undefined}
        display="flex"
        alignItems="center"
        gap="8px"
        bg={isDragging ? 'pg.accentBg' : 'pg.canvas'}
        borderWidth="1px"
        borderStyle="dashed"
        borderColor={isDragging ? 'pg.accent' : 'pg.border'}
        borderRadius="8px"
        pr={file ? '12px' : '0'}
        opacity={disabled ? 0.6 : 1}
        transition="background-color 120ms ease, border-color 120ms ease"
        _hover={disabled ? undefined : {borderColor: 'pg.accent'}}
        _focusWithin={{outline: '2px solid', outlineColor: 'pg.accent', outlineOffset: '2px'}}
        // A disabled card still takes the drag events, only to refuse the file:
        // leaving them unhandled would let the browser navigate to it instead.
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(!disabled);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = disabled ? 'none' : 'copy';
        }}
        onDragLeave={(event) => {
          // Moving onto a child fires dragleave on this element too; only clear
          // the highlight once the pointer has actually left the row.
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setIsDragging(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (disabled) {
            return;
          }
          // Chrome hands over a dropped directory as a zero-byte File, so the
          // entry API is the only way to tell it from a genuinely empty file.
          const entry = event.dataTransfer.items?.[0]?.webkitGetAsEntry?.();
          if (entry != null && !entry.isFile) {
            return;
          }
          const dropped = event.dataTransfer.files?.[0];
          if (dropped) {
            selectFile(dropped);
          }
        }}>
        <VStack
          as="label"
          flex="1"
          minW={0}
          gap="8px"
          px="16px"
          py="32px"
          cursor={disabled ? 'not-allowed' : 'pointer'}>
          <Input
            ref={inputRef}
            type="file"
            disabled={disabled}
            srOnly
            onChange={(event) => {
              onFileChange(event.target.files?.[0] ?? null);
            }}
          />
          <Box aria-hidden="true" color="pg.muted">
            <LuCloudUpload size={32} />
          </Box>
          <Text
            as="span"
            maxW="100%"
            color="pg.ink"
            fontSize="14px"
            fontWeight="semibold"
            lineHeight="1.4"
            overflowWrap="anywhere"
            m={0}>
            {file?.name ?? 'Drag & drop a file here'}
          </Text>
          <Text as="span" color="pg.secondary" fontSize="13px" lineHeight="1.4" m={0}>
            {file === null ? (
              <>
                or{' '}
                <Text as="span" color="pg.accent" textDecoration="underline" textUnderlinePosition="from-font">
                  browse your computer
                </Text>
              </>
            ) : (
              <>
                {formatFileSize(file.size)} ·{' '}
                <Text as="span" color="pg.accent" textDecoration="underline" textUnderlinePosition="from-font">
                  Choose another file
                </Text>
              </>
            )}
          </Text>
        </VStack>

        {file !== null && (
          <IconButton
            aria-label={`Remove ${file.name}`}
            variant="ghost"
            size="xs"
            flexShrink={0}
            color="pg.muted"
            disabled={disabled}
            _hover={{color: 'pg.accent'}}
            onClick={() => selectFile(null)}>
            <LuX />
          </IconButton>
        )}
      </Box>

      {/* Announced separately: a live region inside the label would compete with
          the label text for the file input's accessible name. */}
      <Box srOnly aria-live="polite">
        {file === null ? 'No file selected' : `${file.name}, ${formatFileSize(file.size)} selected`}
      </Box>
    </StepCard>
  );
}
